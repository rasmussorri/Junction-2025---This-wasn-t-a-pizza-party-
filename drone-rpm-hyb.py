import argparse
import time
from enum import Enum
from collections import deque

import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

# We need these from the 'evio' library
try:
    from evio.core.pacer import Pacer
    from evio.source.dat_file import BatchRange, DatFileSource
    from evio.core.recording import open_dat
except ImportError:
    print("ERROR: evio library not found.")
    print("Please clone the repo and run 'uv sync' as per the hackathon plan.")
    exit(1)


# --- State Machine Enum ---
class TrackingStatus(Enum):
    SEARCHING = 1
    TRACKING = 2


# --- Lightweight FFT-based RPM Tracker ---
class FFT_RPMTracker:
    """
    Calculates RPM using a lightweight "Binning + FFT" method.
    This avoids the computationally heavy Lomb-Scargle periodogram.
    """
    
    # --- PLEASE-TUNE THESE ASSUMPTIONS FOR YOUR DRONE ---
    N_BLADES: int = 2
    EXPECTED_RPM_RANGE: tuple[int, int] = (1000, 10000)
    # ---
    
    def __init__(self, history_ms: int = 250, sample_rate_hz: int = 10000):
        self.history_duration_us = history_ms * 1000
        self.sample_rate_hz = sample_rate_hz
        self.bin_size_us = 1_000_000 // self.sample_rate_hz
        self.num_bins = (history_ms * sample_rate_hz) // 1000
        
        self.event_history = deque()
        
        self.current_rpm = 0.0
        self.current_confidence = 0.0
        self.last_calc_time = 0
        
        # --- Pre-calculate FFT frequencies for speed ---
        self.fft_frequencies = np.fft.fftfreq(
            self.num_bins, d=1/self.sample_rate_hz
        )
        
        min_freq_hz = (self.EXPECTED_RPM_RANGE[0] / 60) * self.N_BLADES
        max_freq_hz = (self.EXPECTED_RPM_RANGE[1] / 60) * self.N_BLADES
        
        print(f"[FFT_RPMTracker] Init: {self.num_bins} bins, "
              f"Sample Rate: {self.sample_rate_hz} Hz")
        print(f"[FFT_RPMTracker] Blade-Pass Freq Range: "
              f"{min_freq_hz:.1f} Hz to {max_freq_hz:.1f} Hz")
        
        self.valid_fft_mask = (self.fft_frequencies >= min_freq_hz) & \
                              (self.fft_frequencies <= max_freq_hz)
        
        if not np.any(self.valid_fft_mask):
            print("WARNING: [FFT_RPMTracker] No valid FFT bins in range. "
                  "Check sample rate and RPM range.")
            self.valid_frequencies = np.array([])
        else:
            self.valid_frequencies = self.fft_frequencies[self.valid_fft_mask]
    
    def add_events(self, timestamps_us: np.ndarray, polarities_on: np.ndarray,
                   cluster_mask: np.ndarray | None):
        if cluster_mask is None or len(timestamps_us) == 0:
            return
        
        ts_cluster = timestamps_us[cluster_mask]
        pol_cluster = polarities_on[cluster_mask]
        
        if len(ts_cluster) == 0:
            return
        
        self.event_history.extend(zip(ts_cluster, pol_cluster))
        latest_ts = self.event_history[-1][0]
        
        prune_time = latest_ts - self.history_duration_us
        while self.event_history and self.event_history[0][0] < prune_time:
            self.event_history.popleft()
        
        if latest_ts - self.last_calc_time > 50_000 and len(self.event_history) > 200:
            self._calculate_rpm(latest_ts)
            self.last_calc_time = latest_ts
    
    def _calculate_rpm(self, latest_ts: int):
        if len(self.valid_frequencies) == 0:
            return
        
        signal_bins = np.zeros(self.num_bins)
        
        for ts, pol in self.event_history:
            time_ago_us = latest_ts - ts
            if time_ago_us >= self.history_duration_us:
                continue
            
            bin_index = (self.num_bins - 1) - (time_ago_us // self.bin_size_us)
            
            if 0 <= bin_index < self.num_bins:
                signal_bins[int(bin_index)] += 1 if pol else -1
        
        fft_result = np.fft.fft(signal_bins)
        power_in_range = np.abs(fft_result[self.valid_fft_mask])
        
        if len(power_in_range) == 0:
            self._reset_rpm()
            return
        
        peak_index_in_range = np.argmax(power_in_range)
        peak_freq = self.valid_frequencies[peak_index_in_range]
        peak_power = power_in_range[peak_index_in_range]
        
        self.current_rpm = (peak_freq / self.N_BLADES) * 60.0
        
        mean_power = np.mean(power_in_range)
        snr = peak_power / (mean_power + 1e-6)
        self.current_confidence = min(1.0, (snr - 1.0) / 10.0)
        
        #if self.current_confidence < 0.2:
            #self._reset_rpm()
    
    def _reset_rpm(self):
        self.current_rpm = 0.0
        self.current_confidence = 0.0
    
    def get_rpm(self) -> tuple[float, float]:
        return self.current_rpm, self.current_confidence


def get_window(event_words: np.ndarray, timestamps: np.ndarray,
               time_order: np.ndarray, win_start: int, win_stop: int
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    event_indexes = time_order[win_start:win_stop]
    ts_slice = timestamps[event_indexes].astype(np.int64, copy=False)
    words = event_words[event_indexes].astype(np.uint32, copy=False)
    
    x_coords = (words & 0x3FFF).astype(np.int32, copy=False)
    y_coords = ((words >> 14) & 0x3FFF).astype(np.int32, copy=False)
    polarities_on = ((words >> 28) & 0xF) > 0
    
    return x_coords, y_coords, polarities_on, ts_slice


def get_frame(window: tuple[np.ndarray, np.ndarray, np.ndarray], width: int,
              height: int, *, base_color: tuple[int, int, int] = (0, 0, 0),
              on_color: tuple[int, int, int] = (255, 255, 255),
              off_color: tuple[int, int, int] = (100, 100, 100)) -> np.ndarray:
    x_coords, y_coords, polarities_on = window
    frame = np.full((height, width, 3), base_color, np.uint8)
    if len(x_coords) > 0:
        frame[y_coords[polarities_on], x_coords[polarities_on]] = on_color
        frame[y_coords[~polarities_on], x_coords[~polarities_on]] = off_color
    return frame


def find_drone_cluster(x_coords: np.ndarray, y_coords: np.ndarray,
                       polarities_on: np.ndarray, width: int, height: int,
                       grid_size: int = 30, min_density: int = 20,
                       score_threshold: float = 3000.0,
                       prev_centroid: tuple[int, int] | None = None,
                       search_region_size: int = 300
                       ) -> tuple[tuple[int, int] | None, np.ndarray | None, int]:
    if len(x_coords) == 0:
        return None, None, 0
    
    search_mask = np.ones(len(x_coords), dtype=bool)
    if prev_centroid is not None:
        px, py = prev_centroid
        distances = np.sqrt((x_coords - px)**2 + (y_coords - py)**2)
        search_mask = distances <= search_region_size
        
        if np.sum(search_mask) >= 50:
            x_coords_search = x_coords[search_mask]
            y_coords_search = y_coords[search_mask]
            polarities_search = polarities_on[search_mask]
        else:
            x_coords_search = x_coords
            y_coords_search = y_coords
            polarities_search = polarities_on
            search_mask = np.ones(len(x_coords), dtype=bool)
    else:
        x_coords_search = x_coords
        y_coords_search = y_coords
        polarities_search = polarities_on
    
    total_events = len(x_coords_search)
    frame_area = width * height
    if frame_area > 0:
        avg_density = total_events / (frame_area / (grid_size ** 2))
    else:
        avg_density = 0
    adaptive_min_density = max(3, int(min_density * min(1.0, (avg_density / 5.0) ** 0.7)))
    
    best_centroid = None
    best_mask = None
    best_score = 0
    best_num_propellers = 0
    
    for scale_factor in [0.7, 1.0, 1.5]:
        current_grid_size = int(grid_size * scale_factor)
        bins_x = np.arange(0, width + current_grid_size, current_grid_size)
        bins_y = np.arange(0, height + current_grid_size, current_grid_size)
        
        H_on, xedges, yedges = np.histogram2d(
            x_coords_search[polarities_search],
            y_coords_search[polarities_search],
            bins=[bins_x, bins_y]
        )
        H_off, _, _ = np.histogram2d(
            x_coords_search[~polarities_search],
            y_coords_search[~polarities_search],
            bins=[bins_x, bins_y]
        )
        
        H_density = H_on + H_off
        with np.errstate(divide='ignore', invalid='ignore'):
            H_balance = (np.minimum(H_on, H_off) + 1) / (np.maximum(H_on, H_off) + 1)
            H_balance[np.isnan(H_balance)] = 0
        H_score = H_density * (H_balance ** 1.5)
        
        adaptive_score_threshold = max(
            score_threshold * 0.1,
            np.percentile(H_score[H_score > 0], 90) if np.any(H_score > 0) else 0
        )
        valid_density = H_density >= float(adaptive_min_density)
        valid_balance = H_balance >= 0.3
        valid_cells = valid_density & valid_balance & (H_score >= float(adaptive_score_threshold))
        
        if np.any(valid_cells):
            try:
                labeled_array, num_features = ndimage.label(valid_cells)  # type: ignore
                
                if num_features > 0:
                    cluster_sizes = ndimage.sum(valid_cells, labeled_array, range(1, num_features + 1))
                    num_clusters_to_use = min(4, num_features)
                    top_cluster_indices = np.argsort(cluster_sizes)[-num_clusters_to_use:]
                    
                    valid_cells = np.zeros_like(labeled_array, dtype=bool)
                    for cluster_idx in top_cluster_indices:
                        cluster_label = cluster_idx + 1
                        valid_cells |= (labeled_array == cluster_label)
            except Exception:
                pass
        
        num_propellers = int(np.sum(valid_cells))
        if num_propellers == 0:
            continue
        
        combined_cluster_mask_local = np.zeros(len(x_coords_search), dtype=bool)
        valid_indices = np.argwhere(valid_cells)
        
        for idx in valid_indices:
            idx = tuple(idx)
            x_min, x_max = xedges[idx[0]], xedges[idx[0] + 1]
            y_min, y_max = yedges[idx[1]], yedges[idx[1] + 1]
            cell_mask = (x_coords_search >= x_min) & (x_coords_search < x_max) & \
                        (y_coords_search >= y_min) & (y_coords_search < y_max)
            combined_cluster_mask_local |= cell_mask
        
        if np.sum(combined_cluster_mask_local) == 0:
            continue
        
        centroid = (
            int(np.mean(x_coords_search[combined_cluster_mask_local])),
            int(np.mean(y_coords_search[combined_cluster_mask_local]))
        )
        
        detection_score = np.sum(H_score[valid_cells]) * num_propellers
        
        if prev_centroid is not None:
            distance = np.sqrt((centroid[0] - prev_centroid[0])**2 +
                               (centroid[1] - prev_centroid[1])**2)
            proximity_bonus = max(0, 1.0 - distance / search_region_size) * detection_score * 0.5
            detection_score += proximity_bonus
        
        if detection_score > best_score:
            best_score = detection_score
            best_centroid = centroid
            best_num_propellers = num_propellers
            
            best_mask = np.zeros(len(x_coords), dtype=bool)
            best_mask[search_mask] = combined_cluster_mask_local
    
    return best_centroid, best_mask, best_num_propellers


def calculate_speed(current_centroid, prev_centroid, current_time_us,
                    prev_time_us) -> tuple[float, tuple[float, float]]:
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


def draw_tracking_overlay(frame: np.ndarray, status: TrackingStatus,
                          centroid: tuple[int, int] | None, speed: float,
                          avg_propellers: int = 0, rpm: float = 0.0,
                          rpm_confidence: float = 0.0,
                          cluster_mask: np.ndarray | None = None,
                          x_coords: np.ndarray | None = None,
                          y_coords: np.ndarray | None = None,
                          search_roi: tuple[int, int, int, int] | None = None) -> None:
    if status == TrackingStatus.SEARCHING:
        cv2.putText(frame, "STATUS: SEARCHING", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "Looking for drone...", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        return
    
    if centroid is None:
        cv2.putText(frame, "STATUS: TRACKING (LOST)", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        return
    
    x, y = centroid
    color = (0, 0, 255)
    
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
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), color, 2)
    
    crosshair_size = 20
    cv2.line(frame, (x - crosshair_size, y), (x + crosshair_size, y), color, 2)
    cv2.line(frame, (x, y - crosshair_size), (x, y + crosshair_size), color, 2)
    
    if search_roi:
        cv2.rectangle(frame, (search_roi[0], search_roi[1]),
                      (search_roi[2], search_roi[3]), (255, 255, 0), 1)
    
    cv2.putText(frame, f"STATUS: TRACKING", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    if avg_propellers > 0:
        cv2.putText(frame, f"PROPELLER ESTIMATE: {avg_propellers}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    
    if rpm > 0:
        rpm_color = (0, 255, 0) if rpm_confidence >= 0.7 else (0, 165, 255)
        cv2.putText(frame, f"RPM ESTIMATE: {rpm:.0f}", (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, rpm_color, 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "CALCULATING RPM...:", (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, pacer: Pacer, batch_range: BatchRange, *,
             color: tuple[int, int, int] = (255, 255, 0)) -> None:
    if pacer._t_start is None or pacer._e_start is None:
        return
    
    wall_time_s = time.perf_counter() - pacer._t_start
    rec_time_s = max(0.0, (batch_range.end_ts_us - pacer._e_start) / 1e6)
    first_row_str = f"(target) speed={pacer.speed:.2f}x"
    if pacer.force_speed:
        first_row_str = (
            f"speed={pacer.speed:.2f}x"
            f" avg(drops/ms)={pacer.average_drop_rate:.2f}"
        )
    second_row_str = f"wall={wall_time_s:7.3f}s   rec={rec_time_s:7.3f}s"
    cv2.putText(frame, first_row_str, (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.putText(frame, second_row_str, (8, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid drone tracker (Robust Detection + Speedy Tracking + FFT RPM)"
    )
    parser.add_argument(
        "dat", nargs='?',
        default="C:\\Users\\Henri\\Downloads\\Junction\\Data\\drone_moving-20251114T191633Z-1-002\\drone_moving\\drone_moving.dat",
        help="Path to .dat file (e.g., path/to/drone_moving.dat)"
    )
    parser.add_argument("--window", type=float, default=5.0,
                        help="Window duration in ms (default: 5.0)")
    parser.add_argument("--speed", type=float, default=0.1,
                        help="Playback speed (1.0 is real time)")
    parser.add_argument("--force-speed", action="store_true", default=True,
                        help="Force the playback speed by dropping windows")
    parser.add_argument("--grid_size", type=int, default=20,
                        help="Size (in pixels) of the analysis grid cells.")
    parser.add_argument("--min_density", type=int, default=5,
                        help="Minimum events in a cell to be considered a candidate.")
    parser.add_argument("--score_threshold", type=float, default=1000.0,
                        help="Minimum 'propeller score' (density * balance) to trigger a detection.")
    parser.add_argument("--roi_size", type=int, default=300,
                        help="Size of the search box (ROI) when tracking. (default: 300)")
    parser.add_argument("--track_loss_threshold", type=int, default=5,
                        help="Frames to wait before reverting to SEARCHING mode (default: 5)")
    
    args = parser.parse_args()
    
    print(f"Loading {args.dat}...")
    try:
        src = DatFileSource(args.dat, width=1280, height=720,
                            window_length_us=args.window * 1000)
        
        print("Opening low-level recording to access timestamps...")
        rec = open_dat(args.dat, width=src.width, height=src.height)
        t_raw = getattr(rec, "timestamps", getattr(rec, "ts_us", None))
        if t_raw is None:
            raise AttributeError("Low-level 'rec' object is missing timestamps.")
        
        raw_timestamps = np.asarray(t_raw).astype(np.int64, copy=False)
        
        if raw_timestamps.shape[0] != src.event_words.shape[0]:
            raise ValueError("Timestamp and event word array lengths do not match.")
        
        print(f"File loaded. Resolution: {src.width}x{src.height}")
    except Exception as e:
        print(f"ERROR: Could not load file. {e}")
        return
    
    pacer = Pacer(speed=args.speed, force_speed=args.force_speed)
    cv2.namedWindow("Hybrid Tracker with FFT RPM", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hybrid Tracker with FFT RPM", src.width, src.height)
    
    status = TrackingStatus.SEARCHING
    last_centroid = None
    frames_since_last_seen = 0
    prev_time_us = None
    
    propeller_history = []
    propeller_timestamps = []
    rolling_window_us = 10000000
    
    rpm_tracker = FFT_RPMTracker(history_ms=250, sample_rate_hz=1000)
    
    rpm_history = []
    rpm_timestamps = []
    all_rpm_values = []  # Store all RPM measurements for histogram
    
    print("Starting hybrid tracking loop... Press 'q' or ESC to quit.")
    print(f"Using settings: grid_size={args.grid_size}, score_threshold={args.score_threshold}")
    print(f"Tracking settings: roi_size={args.roi_size}, track_loss={args.track_loss_threshold} frames")
    
    for batch_range in pacer.pace(src.ranges()):
        x_coords, y_coords, polarities, timestamps_us = get_window(
            src.event_words, raw_timestamps, src.order,
            batch_range.start, batch_range.stop
        )
        
        if status == TrackingStatus.SEARCHING:
            centroid_to_use_for_search = None
            current_roi_size = src.width
        else:
            centroid_to_use_for_search = last_centroid
            current_roi_size = args.roi_size
        
        centroid, cluster_mask, num_propellers = find_drone_cluster(
            x_coords, y_coords, polarities, width=src.width, height=src.height,
            grid_size=args.grid_size, min_density=args.min_density,
            score_threshold=args.score_threshold,
            prev_centroid=centroid_to_use_for_search,
            search_region_size=current_roi_size
        )
        
        current_time_us = batch_range.end_ts_us
        speed, velocity = calculate_speed(
            centroid, last_centroid, current_time_us, prev_time_us
        )
        
        if num_propellers > 0:
            propeller_history.append(num_propellers)
            propeller_timestamps.append(current_time_us)
            while propeller_timestamps and (current_time_us - propeller_timestamps[0]) > rolling_window_us:
                propeller_history.pop(0)
                propeller_timestamps.pop(0)
        
        avg_propellers = int(np.ceil(np.mean(propeller_history))) if propeller_history else 0
        
        if centroid and cluster_mask is not None:
            rpm_tracker.add_events(timestamps_us, polarities, cluster_mask)
        
        rpm, rpm_confidence = rpm_tracker.get_rpm()
        
        if rpm > 0 and rpm_confidence > 0.001:
            rpm_history.append(rpm)
            rpm_timestamps.append(current_time_us)
            all_rpm_values.append(rpm)  # Store for histogram
            while rpm_timestamps and (current_time_us - rpm_timestamps[0]) > rolling_window_us:
                rpm_history.pop(0)
                rpm_timestamps.pop(0)
        
        avg_rpm = float(np.mean(rpm_history)) if rpm_history else 0.0
        avg_rpm_confidence = rpm_confidence if rpm_history else 0.0
        
        if centroid:
            status = TrackingStatus.TRACKING
            last_centroid = centroid
            prev_time_us = current_time_us
            frames_since_last_seen = 0
        else:
            frames_since_last_seen += 1
            if frames_since_last_seen > args.track_loss_threshold:
                status = TrackingStatus.SEARCHING
                last_centroid = None
        
        frame = get_frame((x_coords, y_coords, polarities), src.width, src.height)
        draw_hud(frame, pacer, batch_range)
        
        search_roi_box = None
        if status == TrackingStatus.TRACKING and last_centroid:
            half_roi = args.roi_size // 2
            x1 = max(0, last_centroid[0] - half_roi)
            y1 = max(0, last_centroid[1] - half_roi)
            x2 = min(src.width, last_centroid[0] + half_roi)
            y2 = min(src.height, last_centroid[1] + half_roi)
            search_roi_box = (x1, y1, x2, y2)
        
        draw_tracking_overlay(
            frame, status, last_centroid, speed, avg_propellers,
            avg_rpm, avg_rpm_confidence, cluster_mask, x_coords, y_coords,
            search_roi_box
        )
        
        cv2.imshow("Hybrid Tracker with FFT RPM", frame)
        
        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
            break
    
    cv2.destroyAllWindows()
    print("Demo finished.")
    
    # Plot RPM histogram
    if all_rpm_values:
        print(f"\nPlotting RPM histogram with {len(all_rpm_values)} measurements...")
        plt.figure(figsize=(12, 6))
        
        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(all_rpm_values, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        plt.xlabel('RPM', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'RPM Distribution\n({len(all_rpm_values)} measurements)', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Statistics
        mean_rpm = np.mean(all_rpm_values)
        median_rpm = np.median(all_rpm_values)
        std_rpm = np.std(all_rpm_values)
        min_rpm = np.min(all_rpm_values)
        max_rpm = np.max(all_rpm_values)
        
        stats_text = f'Mean: {mean_rpm:.1f} RPM\n'
        stats_text += f'Median: {median_rpm:.1f} RPM\n'
        stats_text += f'Std Dev: {std_rpm:.1f} RPM\n'
        stats_text += f'Min: {min_rpm:.1f} RPM\n'
        stats_text += f'Max: {max_rpm:.1f} RPM'
        
        plt.text(0.98, 0.97, stats_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Time series
        plt.subplot(1, 2, 2)
        plt.plot(all_rpm_values, linewidth=0.8, alpha=0.7)
        plt.xlabel('Sample Index', fontsize=12)
        plt.ylabel('RPM', fontsize=12)
        plt.title('RPM over Time', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=float(mean_rpm), color='r', linestyle='--', label=f'Mean: {mean_rpm:.1f}', linewidth=1.5)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        print(f"\nRPM Statistics:")
        print(f"  Mean:   {mean_rpm:.1f} RPM")
        print(f"  Median: {median_rpm:.1f} RPM")
        print(f"  Std:    {std_rpm:.1f} RPM")
        print(f"  Range:  {min_rpm:.1f} - {max_rpm:.1f} RPM")
    else:
        print("\nNo valid RPM measurements collected.")


if __name__ == "__main__":
    main()
