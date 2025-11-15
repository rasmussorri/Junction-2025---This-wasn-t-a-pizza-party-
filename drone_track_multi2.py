import argparse
from enum import Enum
import cv2
import numpy as np
from collections import deque
from scipy.signal import find_peaks

try:
    from evio.core.pacer import Pacer
    from evio.source.dat_file import DatFileSource
    from evio.core.recording import open_dat
except ImportError:
    print("ERROR: evio library not found. Please install it as per the README.")
    exit(1)

# --- Algorithm & Tracking Constants ---
GRID_SIZE = 20
MIN_DENSITY = 10
SCORE_THRESHOLD = 20.0
ROI_SIZE = 150
TRACK_LOSS_THRESHOLD = 10

# --- RPM Calculation Constants ---
RPM_UPDATE_INTERVAL = 5  # Calculate RPM every N frames

class TrackingStatus(Enum):
    """Defines the current state of the tracker."""
    SEARCHING = 1
    TRACKING = 2

class RPMCalculator:
    def __init__(self):
        self.last_rpm = 0
        self.frame_count = 0

    def calculate(self, x, y, p, ts, centroid, roi_size):
        self.frame_count += 1
        if self.frame_count % RPM_UPDATE_INTERVAL != 0:
            return self.last_rpm

        half_roi = roi_size // 2
        cx, cy = centroid
        x1, y1 = cx - half_roi, cy - half_roi
        x2, y2 = cx + half_roi, cy + half_roi

        mask = (x >= x1) & (x < x2) & (y >= y1) & (y < y2)
        
        propeller_ts = ts[mask]
        if len(propeller_ts) < 50: # Need enough events for FFT
            return self.last_rpm

        # Use FFT to find the frequency of blade passes
        propeller_ts = np.sort(propeller_ts)
        time_diffs = np.diff(propeller_ts)
        
        # We need to work with a regularly sampled signal for FFT
        # So we'll histogram the timestamps
        if len(time_diffs) == 0:
            return self.last_rpm
            
        # Bin timestamps to create a signal
        # The bin size is crucial. Let's try to make it adaptive.
        # A high RPM (e.g., 7000 RPM) means ~116 rotations/sec. For a 2-blade prop, that's 233 Hz.
        # Nyquist theorem says we need to sample at > 466 Hz.
        # Let's use a bin size that gives us a sampling rate of ~1000 Hz.
        t_start, t_end = propeller_ts[0], propeller_ts[-1]
        duration_s = (t_end - t_start) / 1e6  # Timestamps are in microseconds
        if duration_s == 0:
            return self.last_rpm

        num_bins = int(duration_s * 2000) # Aim for 2kHz sampling rate
        if num_bins < 2:
            return self.last_rpm

        hist, bin_edges = np.histogram(propeller_ts, bins=num_bins)
        
        # Perform FFT
        fft_result = np.fft.fft(hist)
        fft_freq = np.fft.fftfreq(len(hist), d=duration_s / num_bins)

        # Find the peak frequency (ignoring the DC component at index 0)
        # We are looking for a peak in a plausible RPM range (e.g., 1000-10000 RPM)
        # which corresponds to a frequency range for a 2-blade prop.
        # 1000 RPM = 16.6 RPS = 33.3 Hz for 2 blades
        # 10000 RPM = 166.6 RPS = 333.3 Hz for 2 blades
        min_freq = 30
        max_freq = 400
        
        idx = np.where((fft_freq > min_freq) & (fft_freq < max_freq))
        if len(idx[0]) == 0:
            return self.last_rpm

        peak_idx = idx[0][np.argmax(np.abs(fft_result[idx]))]
        peak_freq = fft_freq[peak_idx]
        
        # Frequency corresponds to blade passes. For a 2-blade prop, divide by 2 for rotations.
        rotations_per_second = peak_freq / 2.0
        rpm = rotations_per_second * 60

        # Simple smoothing
        if self.last_rpm == 0:
            self.last_rpm = rpm
        else:
            self.last_rpm = (self.last_rpm * 0.9) + (rpm * 0.1)
        
        return self.last_rpm

def get_window_events(event_words: np.ndarray, time_order: np.ndarray, win_start: int, win_stop: int, timestamps: np.ndarray):
    """Extracts and decodes a slice of event data."""
    event_indexes = time_order[win_start:win_stop]
    words = event_words[event_indexes].astype(np.uint32, copy=False)
    ts = timestamps[event_indexes]
    x = (words & 0x3FFF).astype(np.int32, copy=False)
    y = ((words >> 14) & 0x3FFF).astype(np.int32, copy=False)
    p = ((words >> 28) & 0xF) > 0
    return x, y, p, ts

def find_propellers(x, y, p, width, height):
    """
    Analyzes events to find the top 4 propeller hotspots based on score.
    Returns a list of centroids for each detected propeller.
    """
    if len(x) == 0:
        return []

    bins_x = np.arange(0, width + GRID_SIZE, GRID_SIZE)
    bins_y = np.arange(0, height + GRID_SIZE, GRID_SIZE)

    H_on, xedges, yedges = np.histogram2d(x[p], y[p], bins=[bins_x, bins_y])
    H_off, _, _ = np.histogram2d(x[~p], y[~p], bins=[bins_x, bins_y])

    H_density = H_on + H_off
    with np.errstate(divide='ignore', invalid='ignore'):
        balance_num = np.minimum(H_on, H_off) + 1
        balance_den = np.maximum(H_on, H_off) + 1
        H_balance = np.nan_to_num(balance_num / balance_den)

    H_score = np.sqrt(H_density) * H_balance
    H_score[H_density < MIN_DENSITY] = 0

    # Find all cells that meet the score threshold
    hotspot_indices = np.argwhere(H_score >= SCORE_THRESHOLD)
    if len(hotspot_indices) == 0:
        return []

    # Get the scores of each hotspot and sort them to find the best ones
    hotspot_scores = H_score[hotspot_indices[:, 0], hotspot_indices[:, 1]]
    
    # Sort indices by score in descending order and take the top 4
    sorted_hotspot_indices = hotspot_indices[np.argsort(hotspot_scores)[::-1]]
    top_hotspots = sorted_hotspot_indices[:4]

    centroids = []
    for idx in top_hotspots:
        # Note: The indices from argwhere are (row, col), which corresponds to (y, x)
        # in the histogram, but the histogram itself was created with (x, y).
        # So idx[0] is for the x-axis bins and idx[1] is for the y-axis bins.
        x_min, x_max = xedges[idx[0]], xedges[idx[0] + 1]
        y_min, y_max = yedges[idx[1]], yedges[idx[1] + 1]
        
        mask = (x >= x_min) & (x < x_max) & (y >= y_min) & (y < y_max)
        if np.any(mask):
            centroid = (int(np.mean(x[mask])), int(np.mean(y[mask])))
            centroids.append(centroid)
            
    return centroids

def render_frame(x, y, p, width, height):
    """Renders events into a visual frame."""
    frame = np.zeros((height, width, 3), np.uint8)
    frame[y[p], x[p]] = (255, 255, 255)
    frame[y[~p], x[~p]] = (100, 100, 100)
    return frame

def draw_hud(frame, status, propellers, overall_centroid, roi, rpm):
    """Draws the HUD, marking all propellers and the overall center."""
    if status == TrackingStatus.TRACKING and propellers:
        color = (0, 0, 255)
        text = f"STATUS: TRACKING (propeller estimate: {len(propellers)})"
        if roi:
            cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (255, 255, 0), 1)
        
        # Draw a small circle on each propeller
        for prop_center in propellers:
            cv2.circle(frame, prop_center, 10, (0, 0, 255), 2)

        # Draw a large crosshair on the overall center of the drone
        if overall_centroid:
            x, y = overall_centroid
            cv2.line(frame, (x - 20, y), (x + 20, y), (0, 0, 255), 2)
            cv2.line(frame, (x, y - 20), (x, y + 20), (0, 0, 255), 2)
    else:
        color = (255, 100, 0)
        text = "STATUS: SEARCHING"
    
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Display RPM below status
    rpm_text = f"RPM ESTIMATE: {int(rpm)}"
    cv2.putText(frame, rpm_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def main():
    # Hardcoded file path
    dat_file = r"C:\\Users\\Henri\\Downloads\\Junction\\Data\\drone_moving-20251114T191633Z-1-002\\drone_moving\\drone_moving.dat"
    
    parser = argparse.ArgumentParser(description="Multi-propeller drone tracker.")
    parser.add_argument("--dat", default=dat_file, help="Path to .dat file")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed")
    parser.add_argument("--window", type=float, default=20.0, help="Window duration in ms")
    parser.add_argument("--force-speed", action="store_true", default=True, help="Force playback speed")
    args = parser.parse_args()

    # Load recording to get timestamps
    rec = open_dat(args.dat, width=1280, height=720)
    timestamps = rec.timestamps
    
    src = DatFileSource(args.dat, width=1280, height=720, window_length_us=args.window * 1000)
    pacer = Pacer(speed=args.speed, force_speed=args.force_speed)
    
    cv2.namedWindow("Drone Tracker - Multi", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Drone Tracker - Multi", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    status = TrackingStatus.SEARCHING
    last_overall_centroid = None
    frames_since_last_seen = 0
    
    rpm_calculator = RPMCalculator()
    current_rpm = 0
    tracked_propeller_idx = 0 # Track the first propeller for RPM

    print("Starting tracking loop... Press 'q' or ESC to quit.")
    for batch in pacer.pace(src.ranges()):
        x_full, y_full, p_full, ts_full = get_window_events(src.event_words, src.order, batch.start, batch.stop, timestamps)

        search_roi = None
        x_roi, y_roi, p_roi = x_full, y_full, p_full

        if status == TrackingStatus.TRACKING and last_overall_centroid:
            half_roi = ROI_SIZE // 2
            x1 = max(0, last_overall_centroid[0] - half_roi)
            y1 = max(0, last_overall_centroid[1] - half_roi)
            x2 = min(src.width, last_overall_centroid[0] + half_roi)
            y2 = min(src.height, last_overall_centroid[1] + half_roi)
            search_roi = (x1, y1, x2, y2)
            
            mask = (x_full >= x1) & (x_full < x2) & (y_full >= y1) & (y_full < y2)
            x_roi, y_roi, p_roi = x_full[mask], y_full[mask], p_full[mask]

        propellers = find_propellers(x_roi, y_roi, p_roi, src.width, src.height)
        
        overall_centroid = None
        if propellers:
            status = TrackingStatus.TRACKING
            # Calculate the center of mass of all detected propellers
            overall_centroid = (
                int(np.mean([p[0] for p in propellers])),
                int(np.mean([p[1] for p in propellers]))
            )
            last_overall_centroid = overall_centroid
            frames_since_last_seen = 0

            # --- RPM Calculation ---
            if tracked_propeller_idx < len(propellers):
                prop_centroid = propellers[tracked_propeller_idx]
                # Use a smaller ROI for RPM calculation to isolate the propeller
                current_rpm = rpm_calculator.calculate(x_full, y_full, p_full, ts_full, prop_centroid, roi_size=50)
            else:
                # If the tracked propeller is lost, reset RPM
                current_rpm = 0

        else:
            frames_since_last_seen += 1
            if frames_since_last_seen > TRACK_LOSS_THRESHOLD:
                status = TrackingStatus.SEARCHING
                last_overall_centroid = None
                current_rpm = 0

        frame = render_frame(x_full, y_full, p_full, src.width, src.height)
        draw_hud(frame, status, propellers, last_overall_centroid, search_roi, current_rpm)
        cv2.imshow("Drone Tracker - Multi", frame)

        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
            break
            
    cv2.destroyAllWindows()
    print("Demo finished.")

if __name__ == "__main__":
    main()
