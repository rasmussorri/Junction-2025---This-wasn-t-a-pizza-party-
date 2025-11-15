#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import time
from collections import deque
import cv2
import numpy as np
from scipy import signal

# We need these from the 'evio' library
try:
    from evio.core.pacer import Pacer
    from evio.source.dat_file import BatchRange, DatFileSource
    # Import the low-level function that DatFileSource uses internally
    from evio.core.recording import open_dat
except ImportError:
    print("ERROR: evio library not found.")
    print("Please clone the repo and run 'uv sync' as per the hackathon plan.")
    exit(1)


# --- UNCHANGED ---
def get_window(
    event_words: np.ndarray,  # This is src.event_words
    timestamps: np.ndarray,   # This is the raw_timestamps array
    time_order: np.ndarray,   # This is src.order
    win_start: int,
    win_stop: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract event coordinates, polarities, AND TIMESTAMPS
    by slicing the parallel evio arrays.
    """
    event_indexes = time_order[win_start:win_stop]
    ts_slice = timestamps[event_indexes].astype(np.int64, copy=False)
    words = event_words[event_indexes].astype(np.uint32, copy=False)
    
    x_coords = (words & 0x3FFF).astype(np.int32, copy=False)
    y_coords = ((words >> 14) & 0x3FFF).astype(np.int32, copy=False)
    polarities_on = ((words >> 28) & 0xF) > 0

    return x_coords, y_coords, polarities_on, ts_slice


# --- UNCHANGED ---
def get_frame(
    window: tuple[np.ndarray, np.ndarray, np.ndarray],
    width: int = 1280,
    height: int = 720,
    *,
    base_color: tuple[int, int, int] = (0, 0, 0),
    on_color: tuple[int, int, int] = (255, 255, 255),
    off_color: tuple[int, int, int] = (100, 100, 100),
) -> np.ndarray:
    x_coords, y_coords, polarities_on = window
    frame = np.full((height, width, 3), base_color, np.uint8)
    if len(x_coords) > 0:
        frame[y_coords[polarities_on], x_coords[polarities_on]] = on_color
        frame[y_coords[~polarities_on], x_coords[~polarities_on]] = off_color
    return frame


# --- UNCHANGED ---
def find_drone_cluster(
    x_coords: np.ndarray, 
    y_coords: np.ndarray, 
    polarities_on: np.ndarray, 
    grid_size: int = 30, 
    min_density: int = 20,
    score_threshold: float = 3000.0
) -> tuple[tuple[int, int] | None, np.ndarray | None, int]:
    if len(x_coords) == 0:
        return None, None, 0
    bins_x = np.arange(0, 1280 + grid_size, grid_size)
    bins_y = np.arange(0, 720 + grid_size, grid_size)
    H_on, xedges, yedges = np.histogram2d(
        x_coords[polarities_on], y_coords[polarities_on],
        bins=[bins_x, bins_y]
    )
    H_off, _, _ = np.histogram2d(
        x_coords[~polarities_on], y_coords[~polarities_on],
        bins=[bins_x, bins_y]
    )
    H_density = H_on + H_off
    with np.errstate(divide='ignore', invalid='ignore'):
        H_balance = (np.minimum(H_on, H_off) + 1) / (np.maximum(H_on, H_off) + 1)
        H_balance[np.isnan(H_balance)] = 0
    H_score = H_density * H_balance
    H_score[H_density < min_density] = 0
    valid_cells = H_score >= score_threshold
    num_propellers = int(np.sum(valid_cells))
    if num_propellers == 0:
        return None, None, 0
    combined_cluster_mask = np.zeros(len(x_coords), dtype=bool)
    valid_indices = np.argwhere(valid_cells)
    for idx in valid_indices:
        idx = tuple(idx)
        x_min, x_max = xedges[idx[0]], xedges[idx[0] + 1]
        y_min, y_max = yedges[idx[1]], yedges[idx[1] + 1]
        cell_mask = (x_coords >= x_min) & (x_coords < x_max) & \
                    (y_coords >= y_min) & (y_coords < y_max)
        combined_cluster_mask |= cell_mask
    if np.sum(combined_cluster_mask) == 0:
        return None, None, 0
    centroid = (
        int(np.mean(x_coords[combined_cluster_mask])),
        int(np.mean(y_coords[combined_cluster_mask]))
    )
    return centroid, combined_cluster_mask, num_propellers


# --- #################################### ---
# --- THIS IS THE ONLY MODIFIED SECTION ---
# --- #################################### ---
class RPMTracker:
    """
    Tracks RPM using a research-based approach.
    This version is optimized for a 2-blade propeller, per user constraints.
    """
    
    def __init__(self, history_duration_ms: int = 250):
        self.history_duration_us = history_duration_ms * 1000
        self.event_history = deque()
        self.current_rpm = 0.0
        self.current_confidence = 0.0
        
        # --- HARDCODED BLADE COUNT (Based on user constraint) ---
        # This resolves the primary ambiguity of Method 1 (FFT) 
        self.num_blades = 2
        
        self.last_calc_time = 0
        
        # --- HIGHLY-CONSTRAINED FREQUENCY RANGE (Improves Reliability) ---
        # Based on research  and user constraints:
        # User RPM Range: 5500-6500 RPM
        # Rotational Freq (Hz) = RPM / 60
        #   - Min: 5500 / 60 = 91.7 Hz
        #   - Max: 6500 / 60 = 108.3 Hz
        # Blade-Pass Freq (f) = Rotational Freq * N_Blades [cite: 133]
        #   - Min f: 91.7 * 2 = 183.4 Hz
        #   - Max f: 108.3 * 2 = 216.7 Hz
        # We scan a slightly padded range (180-220 Hz) for robustness.
        # This 40 Hz window is far more reliable than the original 200 Hz window.
        self.min_freq_hz = 180.0
        self.max_freq_hz = 220.0
        
        self.scan_frequencies = np.linspace(
            self.min_freq_hz, self.max_freq_hz, 500 # 500 steps is sufficient
        )
        self.angular_frequencies = self.scan_frequencies * 2 * np.pi

    def add_events(self,
                   timestamps_us: np.ndarray,
                   polarities_on: np.ndarray,
                   cluster_mask: np.ndarray | None):
        """
        More efficient event addition.
        Filters events by the cluster mask, converts polarity to +/- 1,
        and adds them to the history deque in one operation.
        """
        if cluster_mask is None or len(timestamps_us) == 0:
            return
            
        ts_cluster = timestamps_us[cluster_mask]
        pol_cluster = polarities_on[cluster_mask]
        
        if len(ts_cluster) == 0:
            return

        # Convert boolean polarities (True/False) to float values (+1.0 / -1.0)
        # for the Lomb-Scargle algorithm
        y_values = np.where(pol_cluster, 1.0, -1.0)
        
        # Combine timestamps and y_values into one (N, 2) array
        # This is more efficient than stacking 'on' and 'off' events separately
        new_data = np.column_stack((ts_cluster, y_values))
        
        # Add the new batch of data to the history
        self.event_history.extend(new_data)
            
        # Prune old events from the history
        latest_ts = self.event_history[-1][0]
        prune_time = latest_ts - self.history_duration_us
        while self.event_history and self.event_history[0][0] < prune_time:
            self.event_history.popleft()
            
        # Trigger RPM calculation every 50ms
        if latest_ts - self.last_calc_time > 50_000:
            self._calculate_rpm() # Renamed function
            self.last_calc_time = latest_ts

    def _calculate_rpm(self):
        """
        Calculates RPM assuming a 2-blade propeller.
        Finds the strongest peak in the *narrowed* Blade-Pass Frequency (BPF) range.
        """
        if len(self.event_history) < 200:
            self._reset_rpm()
            return
            
        # Efficiently convert deque to numpy array for processing
        # This is the main computational cost, but necessary for numpy operations
        try:
            data = np.array(self.event_history, dtype=np.float64)
            ts_us = data[:, 0]
            y_values = data[:, 1]
        except (IndexError, ValueError):
            #Deque was modified during conversion, or is empty
            self._reset_rpm()
            return
            
        # Normalize timestamps to seconds (from first event) for Lomb-Scargle
        ts_sec = (ts_us - ts_us[0]) / 1_000_000.0
        
        # Need at least 0.1s of data for a reliable frequency reading
        if ts_sec[-1] < 0.1:
            self._reset_rpm()
            return
            
        try:
            # Lomb-Scargle is a periodogram for unevenly sampled data.
            # This implements the "Frequency-Domain Analysis" 
            # without the "anti-pattern" of binning[cite: 62].
            power = signal.lombscargle(
                ts_sec, y_values, self.angular_frequencies, normalize=True
            )
        except ValueError:
            # Can happen with malformed data
            self._reset_rpm()
            return
            
        power_mean = np.mean(power)
        power_std = np.std(power)
        
        # 1. Find the *single strongest peak* in our narrow band
        peak_index = np.argmax(power)
        peak_freq_hz = self.scan_frequencies[peak_index]
        peak_power = power[peak_index]

        # 2. Calculate confidence (Signal-to-Noise Ratio of the peak)
        snr = (peak_power - power_mean) / (power_std + 1e-6)
        confidence = float(min(1.0, snr / 10.0)) # Heuristic
        
        # 3. Calculate RPM using the known blade count
        # Formula from research: RPM = (f * 60) / N [cite: 136]
        rpm = float((peak_freq_hz / self.num_blades) * 60.0)

        # 4. Validate and store
        # Confidence threshold and a basic sanity check on the RPM value
        if confidence > 0.2 and 5000 < rpm < 7000:
            self.current_rpm = rpm
            self.current_confidence = confidence
        else:
            self._reset_rpm() # Not a confident or valid reading

    def _reset_rpm(self):
        """Resets the current RPM reading."""
        self.current_rpm = 0.0
        self.current_confidence = 0.0
        # self.num_blades is fixed, no need to reset

    def get_rpm(self) -> tuple[float, float, int]:
        """Returns: (rpm, confidence, num_blades)"""
        return self.current_rpm, self.current_confidence, self.num_blades

# --- #################################### ---
# --- END OF MODIFIED SECTION ---
# --- #################################### ---


# --- UNCHANGED ---
def calculate_speed(
    current_centroid, prev_centroid, 
    current_time_us, prev_time_us
) -> tuple[float, tuple[float, float]]:
    if current_centroid is None or prev_centroid is None or prev_time_us is None:
        return 0.0, (0.0, 0.0)
    dx = current_centroid[0] - prev_centroid[0]
    dy = current_centroid[1] - prev_centroid[1]
    dt_seconds = (current_time_us - prev_time_us) / 1e6
    if dt_seconds == 0:
        return 0.0, (0.0, 0.0)
    vx = dx / dt_seconds
    vy = dy / dt_seconds
    speed = np.sqrt(vx**2 + vy**2)
    return speed, (vx, vy)


# --- UNCHANGED ---
def draw_tracking_overlay(
    frame: np.ndarray, 
    centroid: tuple[int, int] | None, 
    speed: float,
    avg_propellers: int = 0,
    rpm: float = 0.0,
    rpm_confidence: float = 0.0,
    num_blades: int = 0,
    # blade_count_assumed has been removed
    cluster_mask: np.ndarray | None = None, 
    x_coords: np.ndarray | None = None, 
    y_coords: np.ndarray | None = None
) -> None:
    if centroid is None:
        cv2.putText(
            frame, "DRONE: LOST", (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA
        )
        return
    x, y = centroid
    if cluster_mask is not None and x_coords is not None and y_coords is not None:
        cluster_x = x_coords[cluster_mask]
        cluster_y = y_coords[cluster_mask]
        if len(cluster_x) > 0:
            min_x, max_x = int(np.min(cluster_x)), int(np.max(cluster_x))
            min_y, max_y = int(np.min(cluster_y)), int(np.max(cluster_y))
            padding = 20
            min_x = max(0, min_x - padding)
            min_y = max(0, min_y - padding)
            max_x = min(frame.shape[1] - 1, max_x + padding)
            max_y = min(frame.shape[0] - 1, max_y + padding)
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
    crosshair_size = 20
    cv2.line(frame, (x - crosshair_size, y), (x + crosshair_size, y), (0, 255, 0), 2)
    cv2.line(frame, (x, y - crosshair_size), (x, y + crosshair_size), (0, 255, 0), 2)
    cv2.putText(
        frame, f"DRONE: ({x}, {y})", (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA
    )
    cv2.putText(
        frame, f"SPEED: {speed:.1f} px/s", (10, 110),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA
    )
    if avg_propellers > 0:
        cv2.putText(
            frame, f"PROPELLERS: {avg_propellers} detected", (10, 140),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA
        )
        
    # --- SIMPLIFIED BLADE/RPM DISPLAY ---
    if rpm > 0 and num_blades > 0:
        rpm_color = (0, 255, 0)
        if rpm_confidence < 0.7:
             rpm_color = (0, 165, 255)
        elif rpm_confidence < 0.4:
             rpm_color = (0, 0, 255)
             
        # No longer need to check if 'assumed'
        blade_text = f"BLADES: {num_blades} (Fixed)"
         
        cv2.putText(
            frame, blade_text, (10, 170),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, rpm_color, 2, cv2.LINE_AA
        )
        cv2.putText(
            frame, f"RPM: {rpm:.0f} (conf: {rpm_confidence:.2f})", (10, 200),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, rpm_color, 2, cv2.LINE_AA
        )
    else:
        cv2.putText(
            frame, "BLADES: 2 (Fixed)", (10, 170),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA
        )
        cv2.putText(
            frame, "RPM: ---", (10, 200),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA
        )


# --- UNCHANGED ---
def draw_hud(
    frame: np.ndarray,
    pacer: Pacer,
    batch_range: BatchRange,
    *,
    color: tuple[int, int, int] = (255, 255, 0),
) -> None:
    if pacer._t_start is None or pacer._e_start is None:
        return
    wall_time_s = time.perf_counter() - pacer._t_start
    rec_time_s = max(0.0, (batch_range.end_ts_us - pacer._e_start) / 1e6)
    first_row_str = f"(target) speed={pacer.speed:.2f}x"
    if pacer.force_speed:
        first_row_str = (
            f"speed={pacer.speed:.2f}x"
            f"  avg(drops/ms)={pacer.average_drop_rate:.2f}"
        )
    second_row_str = f"wall={wall_time_s:7.3f}s   rec={rec_time_s:7.3f}s"
    cv2.putText(
        frame, first_row_str, (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
    )
    cv2.putText(
        frame, second_row_str, (8, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
    )


# --- UNCHANGED ---
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track drone in event data using propeller signature analysis"
    )
    parser.add_argument(
        "dat",
        nargs='?',
        default="C:\\Users\\Henri\\Downloads\\Junction\\Data\\drone_moving-20251114T191633Z-1-002\\drone_moving\\drone_moving.dat", # PLEASE UPDATE THIS PATH
        help="Path to .dat file (e.g., path/to/drone_moving.dat)"
    )
    parser.add_argument(
        "--window", type=float, default=2.0, 
        help="Window duration in ms. (default: 2.0). "
    )
    parser.add_argument(
        "--speed", type=float, default=1.0, 
        help="Playback speed (1.0 is real time)"
    )
    parser.add_argument(
        "--force-speed", action="store_true",
        help="Force the playback speed by dropping windows",
    )
    parser.add_argument(
        "--grid_size", type=int, default=30,
        help="Size (in pixels) of the analysis grid cells."
    )
    parser.add_argument(
        "--min_density", type=int, default=20,
        help="Minimum events in a cell to be considered a candidate."
    )
    parser.add_argument(
        "--score_threshold", type=float, default=300.0,
        help="Minimum 'propeller score' (density * balance) to trigger a detection."
    )
    args = parser.parse_args()

    if args.window > 10.0:
        print(f"WARNING: Window size of {args.window}ms is likely too large")
        print("to detect drone propeller RPM (~200 Hz / 5ms period).")
        print("Recommend using '--window 2.0' or similar.")

    print(f"Loading {args.dat}...")
    try:
        src = DatFileSource(
            args.dat, 
            width=1280, 
            height=720, 
            window_length_us=args.window * 1000 
        )
        
        print("Opening low-level recording to access timestamps...")
        rec = open_dat(args.dat, width=src.width, height=src.height)
        
        t_raw = getattr(rec, "timestamps", getattr(rec, "ts_us", None))
        if t_raw is None:
             raise AttributeError("Low-level recording 'rec' object is missing timestamps.")
        
        raw_timestamps = np.asarray(t_raw).astype(np.int64, copy=False)
        
        if raw_timestamps.shape[0] != src.event_words.shape[0]:
            raise ValueError("Timestamp and event word array lengths do not match.")
        
        print(f"File loaded. Resolution: {src.width}x{src.height}")
        print(f"Loaded 'event_words' (from src) and 'raw_timestamps' (from rec)")
        
    except Exception as e:
        print(f"ERROR: Could not load file. {e}")
        print("Please ensure the path in the 'default' argument is correct.")
        return

    pacer = Pacer(speed=args.speed, force_speed=args.force_speed)
    # --- MODIFIED WINDOW TITLE ---
    cv2.namedWindow("Drone Tracker (2-Blade Mode)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Drone Tracker (2-Blade Mode)", 1280, 720)

    prev_centroid = None
    prev_time_us = None
    propeller_history = deque(maxlen=50)
    
    rpm_tracker = RPMTracker(history_duration_ms=250)

    print("Starting tracking loop... Press 'q' or ESC to quit.")
    print(f"Using settings: grid_size={args.grid_size}, "
          f"min_density={args.min_density}, "
          f"score_threshold={args.score_threshold}")
    print("--- RPM Tracker set to 2-BLADE mode (180-220 Hz) ---")
    
    for batch_range in pacer.pace(src.ranges()):
        
        x_coords, y_coords, polarities_on, timestamps_us = get_window(
            src.event_words,
            raw_timestamps,
            src.order,
            batch_range.start,
            batch_range.stop,
        )
        
        centroid, cluster_mask, num_propellers = find_drone_cluster(
            x_coords, y_coords, polarities_on, 
            grid_size=args.grid_size, 
            min_density=args.min_density,
            score_threshold=args.score_threshold
        )
        
        current_time_us = batch_range.end_ts_us
        
        speed, velocity = calculate_speed(
            centroid, prev_centroid, 
            current_time_us, prev_time_us
        )
        
        if num_propellers > 0:
            propeller_history.append(num_propellers)
        
        avg_propellers = int(np.ceil(np.mean(propeller_history))) if propeller_history else 0
        
        if centroid is not None and cluster_mask is not None:
            rpm_tracker.add_events(
                timestamps_us,
                polarities_on,
                cluster_mask
            )
        
        # --- SIMPLIFIED GETTER CALL ---
        rpm, rpm_confidence, num_blades = rpm_tracker.get_rpm()
        
        if centroid is not None:
            prev_centroid = centroid
            prev_time_us = current_time_us
        else:
            prev_centroid = None
            prev_time_us = None
        
        frame = get_frame((x_coords, y_coords, polarities_on))
        draw_hud(frame, pacer, batch_range)
        
        # --- SIMPLIFIED DRAW CALL ---
        draw_tracking_overlay(
            frame, centroid, speed, avg_propellers, 
            rpm, rpm_confidence, num_blades,
            cluster_mask, x_coords, y_coords
        )
        
        cv2.imshow("Drone Tracker (2-Blade Mode)", frame)

        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
            break
    
    cv2.destroyAllWindows()
    print("Demo finished.")


if __name__ == "__main__":
    main()