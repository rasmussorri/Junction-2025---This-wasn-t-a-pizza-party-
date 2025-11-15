import argparse
from enum import Enum
import cv2
import numpy as np

try:
    from evio.core.pacer import Pacer
    from evio.source.dat_file import DatFileSource
except ImportError:
    print("ERROR: evio library not found. Please install it as per the README.")
    exit(1)

# --- Algorithm & Tracking Constants ---
GRID_SIZE = 20
MIN_DENSITY = 5
SCORE_THRESHOLD = 20.0
ROI_SIZE = 150
TRACK_LOSS_THRESHOLD = 10

class TrackingStatus(Enum):
    """Defines the current state of the tracker."""
    SEARCHING = 1
    TRACKING = 2

def get_window_events(event_words: np.ndarray, time_order: np.ndarray, win_start: int, win_stop: int):
    """Extracts and decodes a slice of event data."""
    event_indexes = time_order[win_start:win_stop]
    words = event_words[event_indexes].astype(np.uint32, copy=False)
    x = (words & 0x3FFF).astype(np.int32, copy=False)
    y = ((words >> 14) & 0x3FFF).astype(np.int32, copy=False)
    p = ((words >> 28) & 0xF) > 0
    return x, y, p

def find_drone_hotspot(x, y, p, width, height):
    """Analyzes events to find the most likely drone location (hotspot)."""
    if len(x) == 0:
        return None

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

    if np.max(H_score) < SCORE_THRESHOLD:
        return None

    hotspot_idx = np.unravel_index(np.argmax(H_score), H_score.shape)
    x_min, x_max = xedges[hotspot_idx[0]], xedges[hotspot_idx[0] + 1]
    y_min, y_max = yedges[hotspot_idx[1]], yedges[hotspot_idx[1] + 1]

    mask = (x >= x_min) & (x < x_max) & (y >= y_min) & (y < y_max)
    if not np.any(mask):
        return None

    return int(np.mean(x[mask])), int(np.mean(y[mask]))

def render_frame(x, y, p, width, height):
    """Renders events into a visual frame."""
    frame = np.zeros((height, width, 3), np.uint8)
    frame[y[p], x[p]] = (255, 255, 255)
    frame[y[~p], x[~p]] = (100, 100, 100)
    return frame

def draw_hud(frame, status, centroid, roi):
    """Draws the HUD on the frame."""
    if status == TrackingStatus.TRACKING and centroid:
        color = (0, 255, 0)
        text = "STATUS: TRACKING"
        if roi:
            cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (255, 255, 0), 1)
        x, y = centroid
        cv2.line(frame, (x - 20, y), (x + 20, y), color, 2)
        cv2.line(frame, (x, y - 20), (x, y + 20), color, 2)
    else:
        color = (0, 0, 255)
        text = "STATUS: SEARCHING"
    
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def main():
    parser = argparse.ArgumentParser(description="A more elegant drone tracker.")
    parser.add_argument("dat", help="Path to .dat file")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed")
    parser.add_argument("--window", type=float, default=20.0, help="Window duration in ms")
    parser.add_argument("--force-speed", action="store_true", help="Force playback speed")
    args = parser.parse_args()

    src = DatFileSource(args.dat, width=1280, height=720, window_length_us=args.window * 1000)
    pacer = Pacer(speed=args.speed, force_speed=args.force_speed)
    
    cv2.namedWindow("Drone Tracker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Drone Tracker", src.width, src.height)

    status = TrackingStatus.SEARCHING
    last_centroid = None
    frames_since_last_seen = 0
    
    print("Starting tracking loop... Press 'q' or ESC to quit.")
    for batch in pacer.pace(src.ranges()):
        x_full, y_full, p_full = get_window_events(src.event_words, src.order, batch.start, batch.stop)

        search_roi = None
        x_roi, y_roi, p_roi = x_full, y_full, p_full

        if status == TrackingStatus.TRACKING and last_centroid:
            half_roi = ROI_SIZE // 2
            x1, y1 = max(0, last_centroid[0] - half_roi), max(0, last_centroid[1] - half_roi)
            x2, y2 = min(src.width, last_centroid[0] + half_roi), min(src.height, last_centroid[1] + half_roi)
            search_roi = (x1, y1, x2, y2)
            
            mask = (x_full >= x1) & (x_full < x2) & (y_full >= y1) & (y_full < y2)
            x_roi, y_roi, p_roi = x_full[mask], y_full[mask], p_full[mask]

        centroid = find_drone_hotspot(x_roi, y_roi, p_roi, src.width, src.height)

        if centroid:
            status = TrackingStatus.TRACKING
            last_centroid = centroid
            frames_since_last_seen = 0
        else:
            frames_since_last_seen += 1
            if frames_since_last_seen > TRACK_LOSS_THRESHOLD:
                status = TrackingStatus.SEARCHING
                last_centroid = None

        frame = render_frame(x_full, y_full, p_full, src.width, src.height)
        draw_hud(frame, status, last_centroid, search_roi)
        cv2.imshow("Drone Tracker", frame)

        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
            break
            
    cv2.destroyAllWindows()
    print("Demo finished.")

if __name__ == "__main__":
    main()
