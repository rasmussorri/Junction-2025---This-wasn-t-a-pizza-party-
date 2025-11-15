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
MIN_DENSITY = 10
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

def draw_hud(frame, status, propellers, overall_centroid, roi):
    """Draws the HUD, marking all propellers and the overall center."""
    if status == TrackingStatus.TRACKING and propellers:
        color = (0, 255, 0)
        text = f"STATUS: TRACKING ({len(propellers)} props)"
        if roi:
            cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (255, 255, 0), 1)
        
        # Draw a small circle on each propeller
        for prop_center in propellers:
            cv2.circle(frame, prop_center, 10, (0, 0, 255), 2)

        # Draw a large crosshair on the overall center of the drone
        if overall_centroid:
            x, y = overall_centroid
            cv2.line(frame, (x - 20, y), (x + 20, y), color, 2)
            cv2.line(frame, (x, y - 20), (x, y + 20), color, 2)
    else:
        color = (0, 0, 255)
        text = "STATUS: SEARCHING"
    
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def main():
    parser = argparse.ArgumentParser(description="Multi-propeller drone tracker.")
    parser.add_argument("dat", help="Path to .dat file")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed")
    parser.add_argument("--window", type=float, default=20.0, help="Window duration in ms")
    parser.add_argument("--force-speed", action="store_true", help="Force playback speed")
    args = parser.parse_args()

    src = DatFileSource(args.dat, width=1280, height=720, window_length_us=args.window * 1000)
    pacer = Pacer(speed=args.speed, force_speed=args.force_speed)
    
    cv2.namedWindow("Drone Tracker - Multi", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Drone Tracker - Multi", src.width, src.height)

    status = TrackingStatus.SEARCHING
    last_overall_centroid = None
    frames_since_last_seen = 0
    
    print("Starting tracking loop... Press 'q' or ESC to quit.")
    for batch in pacer.pace(src.ranges()):
        x_full, y_full, p_full = get_window_events(src.event_words, src.order, batch.start, batch.stop)

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
        else:
            frames_since_last_seen += 1
            if frames_since_last_seen > TRACK_LOSS_THRESHOLD:
                status = TrackingStatus.SEARCHING
                last_overall_centroid = None

        frame = render_frame(x_full, y_full, p_full, src.width, src.height)
        draw_hud(frame, status, propellers, last_overall_centroid, search_roi)
        cv2.imshow("Drone Tracker - Multi", frame)

        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
            break
            
    cv2.destroyAllWindows()
    print("Demo finished.")

if __name__ == "__main__":
    main()
