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
except ImportError:
    print("ERROR: evio library not found.")
    print("Please clone the repo and run 'uv sync' as per the hackathon plan.")
    exit(1)


def get_window(
    event_words: np.ndarray,
    time_order: np.ndarray,
    win_start: int,
    win_stop: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract event coordinates and polarities for the given window.
    This is a low-level, high-speed function.
    """
    # Get indexes corresponding to events within the window
    event_indexes = time_order[win_start:win_stop]
    words = event_words[event_indexes].astype(np.uint32, copy=False)
    
    # Decode the 32-bit event word using bitwise operations
    x_coords = (words & 0x3FFF).astype(np.int32, copy=False)
    y_coords = ((words >> 14) & 0x3FFF).astype(np.int32, copy=False)
    # Polarity is > 0 for ON, 0 for OFF
    polarities_on = ((words >> 28) & 0xF) > 0

    return x_coords, y_coords, polarities_on


def get_frame(
    window: tuple[np.ndarray, np.ndarray, np.ndarray],
    width: int = 1280,
    height: int = 720,
    *,
    base_color: tuple[int, int, int] = (0, 0, 0),  # Black background
    on_color: tuple[int, int, int] = (255, 255, 255),  # White
    off_color: tuple[int, int, int] = (100, 100, 100),  # Gray
) -> np.ndarray:
    """Render events into a frame for visualization."""
    x_coords, y_coords, polarities_on = window
    
    # Start with a black frame
    frame = np.full((height, width, 3), base_color, np.uint8)
    
    # Draw events only if they exist
    if len(x_coords) > 0:
        # Use numpy "fancy indexing" for high speed
        frame[y_coords[polarities_on], x_coords[polarities_on]] = on_color
        frame[y_coords[~polarities_on], x_coords[~polarities_on]] = off_color

    return frame


def find_drone_cluster(
    x_coords: np.ndarray, 
    y_coords: np.ndarray, 
    polarities_on: np.ndarray, 
    grid_size: int = 30, 
    min_density: int = 20,
    score_threshold: float = 3000.0
) -> tuple[tuple[int, int] | None, np.ndarray | None, int]:
    """
    Finds the drone by identifying ALL spatial grid cells with
    a strong "propeller signature" (4+ propellers).

    A propeller signature is a high-density, high-frequency signal,
    which we proxy as a dense region with a balanced mix of
    ON and OFF polarity events.

    Args:
        x_coords: All x coordinates in the window
        y_coords: All y coordinates in the window
        polarities_on: Boolean mask for ON (True) vs OFF (False) events
        grid_size: The side length (in pixels) of our analysis cells.
        min_density: The minimum number of events a cell must have.
        score_threshold: The minimum H_score to be considered a valid detection.

    Returns:
        (centroid, cluster_mask, num_propellers) or (None, None, 0)
    """
    
    # 0. Handle empty frames
    if len(x_coords) == 0:
        return None, None, 0

    # 1. Define the spatial grid boundaries
    bins_x = np.arange(0, 1280 + grid_size, grid_size)
    bins_y = np.arange(0, 720 + grid_size, grid_size)

    # 2. Create density histograms for ON and OFF events
    H_on, xedges, yedges = np.histogram2d(
        x_coords[polarities_on], y_coords[polarities_on],
        bins=[bins_x, bins_y]
    )
    H_off, _, _ = np.histogram2d(
        x_coords[~polarities_on], y_coords[~polarities_on],
        bins=[bins_x, bins_y]
    )

    # 3. Calculate "Propeller Signature" Score
    H_density = H_on + H_off
    
    with np.errstate(divide='ignore', invalid='ignore'):
        H_balance = (np.minimum(H_on, H_off) + 1) / (np.maximum(H_on, H_off) + 1)
        H_balance[np.isnan(H_balance)] = 0

    H_score = H_density * H_balance
    
    # 4. Find ALL cells above threshold (multiple propellers)
    
    # Ignore cells with very low density (background noise)
    H_score[H_density < min_density] = 0
    
    # Find ALL cells that meet the threshold
    valid_cells = H_score >= score_threshold
    
    # Count the number of propellers detected
    num_propellers = int(np.sum(valid_cells))
    
    # If no cells are good enough, exit
    if num_propellers == 0:
        return None, None, 0
    
    # 5. Extract events from ALL valid cells (all propellers combined)
    
    # Initialize combined cluster mask
    combined_cluster_mask = np.zeros(len(x_coords), dtype=bool)
    
    # Get indices of all valid cells
    valid_indices = np.argwhere(valid_cells)
    
    for idx in valid_indices:
        idx = tuple(idx)
        
        # Get the pixel boundaries of this cell
        x_min, x_max = xedges[idx[0]], xedges[idx[0] + 1]
        y_min, y_max = yedges[idx[1]], yedges[idx[1] + 1]
        
        # Add events from this cell to the combined mask
        cell_mask = (x_coords >= x_min) & (x_coords < x_max) & \
                    (y_coords >= y_min) & (y_coords < y_max)
        
        combined_cluster_mask |= cell_mask
    
    # Safety check
    if np.sum(combined_cluster_mask) == 0:
        return None, None, 0

    # 6. Calculate the precise centroid of ALL propellers combined
    # This gives us the center of the entire drone
    centroid = (
        int(np.mean(x_coords[combined_cluster_mask])),
        int(np.mean(y_coords[combined_cluster_mask]))
    )

    return centroid, combined_cluster_mask, num_propellers


class RPMTracker:
    """Tracks RPM of propellers using event count analysis."""
    
    def __init__(self, roi_radius: int = 40, min_windows: int = 10, num_blades: int = 2):
        self.roi_radius = roi_radius
        self.min_windows = min_windows
        self.num_blades = num_blades
        
        # History buffers
        self.event_counts = deque(maxlen=100)
        self.timestamps = deque(maxlen=100)
        self.window_durations = deque(maxlen=100)
        
    def add_window(self, centroid: tuple[int, int] | None, x_coords: np.ndarray, 
                   y_coords: np.ndarray, timestamp_us: int, window_duration_us: int):
        """Add a new window of event data."""
        if centroid is None:
            return
        
        # Count events in circular ROI
        cx, cy = centroid
        distances = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
        event_count = np.sum(distances <= self.roi_radius)
        
        self.event_counts.append(event_count)
        self.timestamps.append(timestamp_us)
        self.window_durations.append(window_duration_us)
    
    def calculate_rpm_peak_detection(self) -> tuple[float, float]:
        """Calculate RPM using peak detection on event count signal."""
        if len(self.event_counts) < self.min_windows:
            return 0.0, 0.0
        
        counts = np.array(self.event_counts)
        times = np.array(self.timestamps) / 1e6  # Convert to seconds
        
        # Smooth the signal
        if len(counts) > 5:
            # Ensure window length is odd and <= array size
            window_length = min(len(counts) // 2 * 2 + 1, 11)
            if window_length > len(counts):
                window_length = len(counts) if len(counts) % 2 == 1 else len(counts) - 1
            if window_length >= 4:  # Minimum for polyorder 3
                counts = signal.savgol_filter(counts, window_length, min(3, window_length - 1))
        
        # Find peaks
        mean_count = np.mean(counts)
        peaks, _ = signal.find_peaks(counts, height=mean_count * 0.2, distance=2)
        
        if len(peaks) < 2:
            return 0.0, 0.0
        
        # Measure time between peaks
        peak_times = times[peaks]
        intervals = np.diff(peak_times)
        
        # Filter to realistic RPM range (100-10000 RPM)
        valid_intervals = intervals[(intervals > 60 / 10000) & (intervals < 60 / 100)]
        
        if len(valid_intervals) == 0:
            return 0.0, 0.0
        
        # Use median interval
        blade_period = np.median(valid_intervals)
        rpm = 60 / (blade_period * self.num_blades)
        
        # Confidence based on consistency
        variance = np.std(valid_intervals) / np.mean(valid_intervals) if len(valid_intervals) > 1 else 1.0
        confidence = max(0.0, 1.0 - variance)
        
        return float(rpm), float(confidence)
    
    def calculate_rpm_fft(self) -> tuple[float, float]:
        """Calculate RPM using FFT analysis."""
        if len(self.event_counts) < self.min_windows:
            return 0.0, 0.0
        
        counts = np.array(self.event_counts)
        
        # Normalize
        counts_norm = (counts - np.mean(counts)) / (np.std(counts) + 1e-6)
        
        # Apply Hanning window
        window = np.hanning(len(counts_norm))
        counts_windowed = counts_norm * window
        
        # Compute FFT
        fft = np.fft.rfft(counts_windowed)
        power = np.abs(fft) ** 2
        
        # Get frequencies
        window_durations = np.array(self.window_durations)
        avg_dt = np.mean(window_durations) / 1e6  # seconds
        freqs = np.fft.rfftfreq(len(counts_windowed), d=avg_dt)
        
        # Find dominant frequency (skip DC component)
        if len(power) < 2:
            return 0.0, 0.0
        
        dominant_idx = np.argmax(power[1:]) + 1
        dominant_freq = freqs[dominant_idx]
        
        # Convert to RPM
        rpm = (dominant_freq * 60) / self.num_blades
        
        # Validate range
        if rpm < 100 or rpm > 10000:
            return 0.0, 0.0
        
        return float(rpm), 0.3  # Fixed confidence for FFT
    
    def get_rpm(self) -> tuple[float, float]:
        """Get current RPM estimate with confidence."""
        rpm_peak, conf_peak = self.calculate_rpm_peak_detection()
        rpm_fft, conf_fft = self.calculate_rpm_fft()
        
        # Prefer peak detection if confident
        if conf_peak > 0.5:
            return rpm_peak, conf_peak
        elif rpm_fft > 0:
            return rpm_fft, conf_fft
        else:
            return 0.0, 0.0


def calculate_speed(
    current_centroid, prev_centroid, 
    current_time_us, prev_time_us
) -> tuple[float, tuple[float, float]]:
    """Calculate speed between two centroids."""
    # Check if either centroid is None
    if current_centroid is None or prev_centroid is None or prev_time_us is None:
        return 0.0, (0.0, 0.0)
    
    # Calculate displacement
    dx = current_centroid[0] - prev_centroid[0]
    dy = current_centroid[1] - prev_centroid[1]
    
    # Calculate time difference in seconds
    dt_seconds = (current_time_us - prev_time_us) / 1e6
    
    if dt_seconds == 0:
        return 0.0, (0.0, 0.0)
    
    # Calculate velocity (pixels/second)
    vx = dx / dt_seconds
    vy = dy / dt_seconds
    
    # Calculate scalar speed
    speed = np.sqrt(vx**2 + vy**2)
    
    return speed, (vx, vy)


def draw_tracking_overlay(
    frame: np.ndarray, 
    centroid: tuple[int, int] | None, 
    speed: float,
    avg_propellers: int = 0,
    rpm: float = 0.0,
    rpm_confidence: float = 0.0,
    roi_radius: int = 40,
    cluster_mask: np.ndarray | None = None, 
    x_coords: np.ndarray | None = None, 
    y_coords: np.ndarray | None = None
) -> None:
    """Draw tracking visualization on the frame."""
    
    if centroid is None:
        # Draw "LOST" message
        cv2.putText(
            frame, "DRONE: LOST", (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA
        )
        return
    
    x, y = centroid
    
    # Draw ROI circle for RPM tracking
    cv2.circle(frame, (x, y), roi_radius, (0, 255, 255), 2)
    
    # Draw a bounding box around the cluster
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
    
    # Draw crosshair at centroid
    crosshair_size = 20
    cv2.line(frame, (x - crosshair_size, y), (x + crosshair_size, y), (0, 255, 0), 2)
    cv2.line(frame, (x, y - crosshair_size), (x, y + crosshair_size), (0, 255, 0), 2)
    
    # Display position and speed
    cv2.putText(
        frame, f"DRONE: ({x}, {y})", (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA
    )
    cv2.putText(
        frame, f"SPEED: {speed:.1f} px/s", (10, 110),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA
    )
    # Display average number of propellers detected
    if avg_propellers > 0:
        cv2.putText(
            frame, f"PREDICTED NUM OF PROPELLERS: {avg_propellers}", (10, 140),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA
        )
    
    # Display RPM
    if rpm > 0:
        rpm_color = (0, 255, 0) if rpm_confidence > 0.5 else (0, 165, 255)  # Green if confident, orange if low
        cv2.putText(
            frame, f"AVG RPM: {rpm:.0f} (conf: {rpm_confidence:.2f})", (10, 170),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, rpm_color, 2, cv2.LINE_AA
        )


def draw_hud(
    frame: np.ndarray,
    pacer: Pacer,
    batch_range: BatchRange,
    *,
    color: tuple[int, int, int] = (255, 255, 0),  # Cyan
) -> None:
    """Overlay timing info: wall time, recording time, and playback speed."""
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track drone in event data using propeller signature analysis"
    )
    parser.add_argument(
        "dat",
        nargs='?',
        default="C:\\Users\\Henri\\Downloads\\Junction\\Data\\drone_moving-20251114T191633Z-1-002\\drone_moving\\drone_moving.dat",
        #"C:\\Users\\Henri\\Downloads\\Junction\\Data\\fred-1-20251114T194658Z-1-001\\fred-1\\Event\\events.dat",
        #"C:\\Users\\Henri\\Downloads\\Junction\\Data\\drone_moving-20251114T191633Z-1-002\\drone_moving\\drone_moving.dat",
        help="Path to .dat file (e.g., path/to/drone_moving.dat)"
    )
    parser.add_argument(
        "--window", type=float, default=20.0, 
        help="Window duration in ms (default: 20.0)"
    )
    parser.add_argument(
        "--speed", type=float, default=1.0, 
        help="Playback speed (1.0 is real time)"
    )
    parser.add_argument(
        "--force-speed", action="store_true",
        help="Force the playback speed by dropping windows",
    )
    # --- New Tunable Parameters ---
    parser.add_argument(
        "--grid_size", type=int, default=30,
        help="Size (in pixels) of the analysis grid cells."
    )
    parser.add_argument(
        "--min_density", type=int, default=20,
        help="Minimum events in a cell to be considered a candidate."
    )
    # --- NEW THRESHOLD ARGUMENT ---
    parser.add_argument(
        "--score_threshold", type=float, default=1000.0,
        help="Minimum 'propeller score' (density * balance) to trigger a detection. (default: 1000.0)"
    )
    args = parser.parse_args()

    # 1. Load the data file
    print(f"Loading {args.dat}...")
    try:
        src = DatFileSource(
            args.dat, 
            width=1280, 
            height=720, 
            window_length_us=args.window * 1000
        )
        print(f"File loaded. Resolution: {src.width}x{src.height}")
    except Exception as e:
        print(f"ERROR: Could not load file. {e}")
        return

    # 2. Setup playback and visualization
    pacer = Pacer(speed=args.speed, force_speed=args.force_speed)
    cv2.namedWindow("Drone Tracker (Propeller Signature)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Drone Tracker (Propeller Signature)", 1280, 720)

    # 3. Initialize tracking state
    prev_centroid = None
    prev_time_us = None
    
    # Propeller count tracking (rolling average over last 10 seconds)
    propeller_history = []
    propeller_timestamps = []
    rolling_window_us = 100000000  # 10 seconds in microseconds
    
    # RPM tracking
    rpm_tracker = RPMTracker(roi_radius=40, min_windows=10, num_blades=2)

    print("Starting tracking loop... Press 'q' or ESC to quit.")
    print(f"Using settings: grid_size={args.grid_size}, min_density={args.min_density}, score_threshold={args.score_threshold}")
    
    # 4. Main Loop
    for batch_range in pacer.pace(src.ranges()):
        
        x_coords, y_coords, polarities = get_window(
            src.event_words,
            src.order,
            batch_range.start,
            batch_range.stop,
        )
        
        centroid, cluster_mask, num_propellers = find_drone_cluster(
            x_coords, y_coords, polarities, 
            grid_size=args.grid_size, 
            min_density=args.min_density,
            score_threshold=args.score_threshold
        )
        
        current_time_us = batch_range.end_ts_us
        speed, velocity = calculate_speed(
            centroid, prev_centroid, 
            current_time_us, prev_time_us
        )
        
        # Update propeller count history
        if num_propellers > 0:
            propeller_history.append(num_propellers)
            propeller_timestamps.append(current_time_us)
            
            # Remove old entries outside the rolling window
            while propeller_timestamps and (current_time_us - propeller_timestamps[0]) > rolling_window_us:
                propeller_history.pop(0)
                propeller_timestamps.pop(0)
        
        # Calculate average propeller count (rounded up)
        avg_propellers = int(np.ceil(np.mean(propeller_history))) if propeller_history else 0
        
        # Update RPM tracker
        window_duration_us = int(args.window * 1000)
        rpm_tracker.add_window(centroid, x_coords, y_coords, current_time_us, window_duration_us)
        rpm, rpm_confidence = rpm_tracker.get_rpm()
        
        if centroid is not None:
            prev_centroid = centroid
            prev_time_us = current_time_us
        else:
            # If drone is lost, reset the state
            prev_centroid = None
            prev_time_us = None
        
        frame = get_frame((x_coords, y_coords, polarities))
        draw_hud(frame, pacer, batch_range)
        draw_tracking_overlay(frame, centroid, speed, avg_propellers, rpm, rpm_confidence, 
                            rpm_tracker.roi_radius, cluster_mask, x_coords, y_coords)
        
        cv2.imshow("Drone Tracker (Propeller Signature)", frame)

        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
            break
    
    cv2.destroyAllWindows()
    print("Demo finished.")


if __name__ == "__main__":
    main()