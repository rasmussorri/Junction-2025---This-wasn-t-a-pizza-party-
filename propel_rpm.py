import argparse
import cv2
import numpy as np
from scipy.fft import fft, fftfreq
import sys

sys.path.insert(0, './src')

try:
    from evio.core.pacer import Pacer
    from evio.source.dat_file import DatFileSource
except ImportError:
    print("ERROR: evio library not found. Please install it as per the README.")
    exit(1)

# Import common functions from drone_track_multi
def get_window_events(event_words: np.ndarray, all_timestamps: np.ndarray, time_order: np.ndarray, win_start: int, win_stop: int):
    """Extracts and decodes a slice of event data including timestamps."""
    event_indexes = time_order[win_start:win_stop]
    words = event_words[event_indexes].astype(np.uint32, copy=False)
    timestamps = all_timestamps[event_indexes]
    x = (words & 0x3FFF).astype(np.int32, copy=False)
    y = ((words >> 14) & 0x3FFF).astype(np.int32, copy=False)
    p = ((words >> 28) & 0xF) > 0
    return x, y, p, timestamps

from drone_track_multi import (
    find_propellers,
    render_frame,
    ROI_SIZE,
    GRID_SIZE,
    MIN_DENSITY,
    SCORE_THRESHOLD,
)

# --- Constants ---
FFT_WINDOW_MS = 200  # Time window for FFT analysis in milliseconds
RPM_MOVING_AVERAGE_SIZE = 5 # Number of RPM values to average for stability

def calculate_rpm(events_timestamps, duration_s):
    """
    Calculates RPM from event timestamps using FFT.
    Assumes a 2-bladed propeller.
    """
    # We need a minimum number of events to get a reliable FFT
    if len(events_timestamps) < 50:
        return 0

    # Create a histogram of events over time
    # We can use a coarser binning to speed up FFT
    num_bins = int(duration_s * 2000)  # 0.5ms bins for better resolution
    if num_bins < 10: # Need enough bins for FFT
        return 0
    
    # Use the actual time range of events for more accurate binning
    min_ts, max_ts = events_timestamps.min(), events_timestamps.max()
    actual_duration = (max_ts - min_ts) / 1e6
    if actual_duration <= 0:
        return 0

    time_bins = np.linspace(min_ts, max_ts, num_bins)
    event_counts, _ = np.histogram(events_timestamps, bins=time_bins)

    # Perform FFT
    N = len(event_counts)
    T = actual_duration / N
    yf = fft(event_counts)
    xf = fftfreq(N, T)[:N//2]

    # Find the peak frequency with more robust filtering
    min_freq_hz = 20  # Min expected propeller frequency (corresponds to ~600 RPM)
    max_freq_hz = 400 # Max expected propeller frequency in Hz
    
    min_freq_index = np.searchsorted(xf, min_freq_hz, side='right')
    max_freq_index = np.searchsorted(xf, max_freq_hz, side='left')
    
    # Ensure we have a valid frequency range to search
    if min_freq_index >= max_freq_index:
        return 0

    # Find the peak in the desired frequency range
    peak_idx = np.argmax(np.abs(yf[min_freq_index:max_freq_index])) + min_freq_index
    
    # Check if the peak is significant
    peak_magnitude = np.abs(yf[peak_idx])
    mean_magnitude = np.mean(np.abs(yf[min_freq_index:max_freq_index]))
    if peak_magnitude < mean_magnitude * 2: # Threshold for peak significance
        return 0

    peak_freq = xf[peak_idx]
    
    # Convert frequency to RPM. For a 2-bladed propeller, the event frequency is 2x the rotation frequency.
    rpm = (peak_freq / 2) * 60
    
    # Sanity check for RPM value
    if rpm < 500 or rpm > 12000:
        return 0
        
    return rpm


def draw_hud(frame, propellers, rpm):
    """Draws the HUD, marking propellers and displaying a single RPM."""
    width = frame.shape[1]
    for prop_center in propellers:
        cv2.circle(frame, prop_center, 15, (0, 0, 255), 2)
    
    rpm_text = f"RPM: {rpm:.0f}"
    cv2.putText(frame, rpm_text, (width - 220, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)


def main():
    parser = argparse.ArgumentParser(description="Drone Propeller RPM Calculator.")
    parser.add_argument("dat", help="Path to .dat file")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed")
    parser.add_argument("--force-speed", action="store_true", help="Force playback speed")
    args = parser.parse_args()

    # We use a larger window to have enough data for FFT
    src = DatFileSource(args.dat, width=1280, height=720, window_length_us=FFT_WINDOW_MS * 1000)
    pacer = Pacer(speed=args.speed, force_speed=args.force_speed)
    all_timestamps = np.array(src.timestamps, copy=False)
    
    cv2.namedWindow("Propeller RPM", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Propeller RPM", src.width, src.height)

    print("Starting RPM calculation... Press 'q' or ESC to quit.")
    
    # --- Stability: Use a moving average for RPM ---
    rpm_history = []
    stable_rpm = 0

    for batch in pacer.pace(src.ranges()):
        x_full, y_full, p_full, t_full = get_window_events(src.event_words, all_timestamps, src.order, batch.start, batch.stop)

        if len(x_full) == 0:
            frame = render_frame(x_full, y_full, p_full, src.width, src.height)
            draw_hud(frame, [], stable_rpm)
            cv2.imshow("Propeller RPM", frame)
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                break
            continue

        # Find propellers in the whole frame
        propellers = find_propellers(x_full, y_full, p_full, src.width, src.height)
        
        current_rpm = 0
        if propellers:
            # --- Optimization: Calculate RPM only for the first detected propeller ---
            prop_center = propellers[0]
            half_roi = ROI_SIZE // 2
            x1, y1 = prop_center[0] - half_roi, prop_center[1] - half_roi
            x2, y2 = prop_center[0] + half_roi, prop_center[1] + half_roi
            
            # Extract events within the ROI
            mask = (x_full >= x1) & (x_full < x2) & (y_full >= y1) & (y_full < y2)
            
            if np.any(mask):
                prop_timestamps = t_full[mask]
                duration_s = (batch.end_ts_us - batch.start_ts_us) / 1e6
                current_rpm = calculate_rpm(prop_timestamps, duration_s)

        # --- Update moving average for stable RPM ---
        if current_rpm > 0:
            rpm_history.append(current_rpm)
            if len(rpm_history) > RPM_MOVING_AVERAGE_SIZE:
                rpm_history.pop(0)
            
            stable_rpm = np.mean(rpm_history)

        # Render the frame and HUD
        frame = render_frame(x_full, y_full, p_full, src.width, src.height)
        draw_hud(frame, propellers, stable_rpm)
        
        cv2.imshow("Propeller RPM", frame)

        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
            break
            
    cv2.destroyAllWindows()
    print("Demo finished.")

if __name__ == "__main__":
    main()
