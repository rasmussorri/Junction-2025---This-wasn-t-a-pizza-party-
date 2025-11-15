import argparse
import time
from enum import Enum

import cv2
import numpy as np

# We need these from the 'evio' library
try:
    from evio.core.pacer import Pacer
    from evio.source.dat_file import BatchRange, DatFileSource
except ImportError:
    print("ERROR: evio library not found.")
    print("Please clone the repo and run 'pip install -r requirements.txt' as per the README.")
    exit(1)

class TrackingStatus(Enum):
    """Defines the current state of the tracker."""
    SEARCHING = 1  # Looking for a new target across the full frame
    TRACKING = 2   # Locked onto a target, searching in a small ROI

def get_window_events(
    event_words: np.ndarray,
    time_order: np.ndarray,
    win_start: int,
    win_stop: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts and decodes event data for a specific time window.
    Returns raw coordinates and polarities.
    """
    event_indexes = time_order[win_start:win_stop]
    words = event_words[event_indexes].astype(np.uint32, copy=False)
    
    x_coords = (words & 0x3FFF).astype(np.int32, copy=False)
    y_coords = ((words >> 14) & 0x3FFF).astype(np.int32, copy=False)
    polarities_on = ((words >> 28) & 0xF) > 0

    return x_coords, y_coords, polarities_on, event_indexes

def find_drone_hotspot(
    x_coords: np.ndarray, 
    y_coords: np.ndarray, 
    polarities_on: np.ndarray, 
    width: int,
    height: int,
    grid_size: int, 
    min_density: int,
    score_threshold: float,
) -> tuple[tuple[int, int] | None, np.ndarray | None]:
    """
    Finds the drone by identifying the single best "hotspot" cell and
    building a cluster around it. This is the core detection algorithm.
    """
    if len(x_coords) == 0:
        return None, None

    # 1. Define the spatial grid for analysis
    bins_x = np.arange(0, width + grid_size, grid_size)
    bins_y = np.arange(0, height + grid_size, grid_size)

    # 2. Create density histograms for ON and OFF events
    H_on, xedges, yedges = np.histogram2d(
        x_coords[polarities_on], y_coords[polarities_on], bins=[bins_x, bins_y]
    )
    H_off, _, _ = np.histogram2d(
        x_coords[~polarities_on], y_coords[~polarities_on], bins=[bins_x, bins_y]
    )

    # 3. Calculate the "Propeller Signature" Score
    H_density = H_on + H_off
    with np.errstate(divide='ignore', invalid='ignore'):
        # The balance score (ratio of ON/OFF events) is the key to finding the flicker
        H_balance = (np.minimum(H_on, H_off) + 1) / (np.maximum(H_on, H_off) + 1)
        H_balance[np.isnan(H_balance)] = 0

    # The final score prioritizes flicker quality over raw event count
    H_score = np.sqrt(H_density) * H_balance
    
    # 4. Filter out low-density cells to reduce noise
    H_score[H_density < min_density] = 0
    
    # 5. Find the single best cell (the "hotspot")
    max_score = np.max(H_score)
    if max_score < score_threshold:
        return None, None  # No cell is good enough
    
    hotspot_idx = np.unravel_index(np.argmax(H_score), H_score.shape)
    
    # 6. Create a cluster of events from the hotspot cell
    x_min, x_max = xedges[hotspot_idx[0]], xedges[hotspot_idx[0] + 1]
    y_min, y_max = yedges[hotspot_idx[1]], yedges[hotspot_idx[1] + 1]
    
    cluster_mask = (x_coords >= x_min) & (x_coords < x_max) & \
                   (y_coords >= y_min) & (y_coords < y_max)
    
    if np.sum(cluster_mask) == 0:
        return None, None

    # 7. Calculate the precise centroid of the hotspot cluster
    centroid = (
        int(np.mean(x_coords[cluster_mask])),
        int(np.mean(y_coords[cluster_mask]))
    )

    return centroid, cluster_mask

def render_frame(
    x_coords: np.ndarray, y_coords: np.ndarray, polarities_on: np.ndarray,
    width: int, height: int
) -> np.ndarray:
    """Renders the events into a visual frame."""
    frame = np.full((height, width, 3), (0, 0, 0), np.uint8)
    frame[y_coords[polarities_on], x_coords[polarities_on]] = (255, 255, 255)
    frame[y_coords[~polarities_on], x_coords[~polarities_on]] = (100, 100, 100)
    return frame

def draw_hud(frame: np.ndarray, status: TrackingStatus, centroid: tuple | None, roi: tuple | None) -> None:
    """Draws the tracking status and visualization overlays."""
    if status == TrackingStatus.TRACKING and centroid:
        # Green indicates a successful track lock
        color = (0, 255, 0)
        cv2.putText(frame, f"STATUS: TRACKING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw the ROI search box
        if roi:
            x1, y1, x2, y2 = roi
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1) # Cyan box

        # Draw crosshair at centroid
        x, y = centroid
        cv2.line(frame, (x - 20, y), (x + 20, y), color, 2)
        cv2.line(frame, (x, y - 20), (x, y + 20), color, 2)
        
    else:
        # Red indicates searching for a target
        color = (0, 0, 255)
        cv2.putText(frame, "STATUS: SEARCHING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def main() -> None:
    parser = argparse.ArgumentParser(description="Optimized drone tracker using hotspot and track-lock logic.")
    parser.add_argument("dat", help="Path to .dat file")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed (1.0 is real time)")
    parser.add_argument("--window", type=float, default=20.0, help="Window duration in ms (default: 20.0)")
    parser.add_argument("--force-speed", action="store_true", help="Force playback speed by dropping windows")
    
    # Algorithm tuning parameters
    parser.add_argument("--grid_size", type=int, default=20, help="Size of analysis grid cells (default: 20)")
    parser.add_argument("--min_density", type=int, default=10, help="Minimum events in a cell to be a candidate (default: 10)")
    parser.add_argument("--score_threshold", type=float, default=15.0, help="Minimum propeller score for detection (default: 15.0)")
    
    # Track-lock tuning parameters
    parser.add_argument("--roi_size", type=int, default=150, help="Size of the search box (ROI) when tracking (default: 150)")
    parser.add_argument("--track_loss_threshold", type=int, default=5, help="Frames to wait before reverting to SEARCHING mode (default: 5)")
    
    args = parser.parse_args()

    # 1. Load data source
    print(f"Loading {args.dat}...")
    src = DatFileSource(args.dat, width=1280, height=720, window_length_us=args.window * 1000)
    print(f"File loaded. Resolution: {src.width}x{src.height}")

    # 2. Setup visualization and playback
    pacer = Pacer(speed=args.speed, force_speed=args.force_speed)
    cv2.namedWindow("Drone Hotspot Tracker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Drone Hotspot Tracker", src.width, src.height)

    # 3. Initialize tracking state
    status = TrackingStatus.SEARCHING
    last_centroid = None
    frames_since_last_seen = 0
    search_roi = None

    print("Starting tracking loop... Press 'q' or ESC to quit.")
    
    # 4. Main Loop
    for batch_range in pacer.pace(src.ranges()):
        
        x_coords, y_coords, polarities, event_indices = get_window_events(
            src.event_words, src.order, batch_range.start, batch_range.stop
        )

        # --- Track-Lock Logic ---
        if status == TrackingStatus.TRACKING and last_centroid:
            # Define the search box (ROI) around the last known position
            half_roi = args.roi_size // 2
            x1 = max(0, last_centroid[0] - half_roi)
            y1 = max(0, last_centroid[1] - half_roi)
            x2 = min(src.width, last_centroid[0] + half_roi)
            y2 = min(src.height, last_centroid[1] + half_roi)
            search_roi = (x1, y1, x2, y2)

            # Create a mask to filter events within the ROI
            roi_mask = (x_coords >= x1) & (x_coords < x2) & (y_coords >= y1) & (y_coords < y2)
            
            # Use only the events inside the ROI for detection
            x_roi, y_roi, pol_roi = x_coords[roi_mask], y_coords[roi_mask], polarities[roi_mask]
        else:
            # In SEARCHING mode, use all events
            x_roi, y_roi, pol_roi = x_coords, y_coords, polarities
            search_roi = None

        # Run the detection algorithm on the selected events (either full-frame or ROI)
        centroid, _ = find_drone_hotspot(
            x_roi, y_roi, pol_roi,
            width=src.width, height=src.height,
            grid_size=args.grid_size,
            min_density=args.min_density,
            score_threshold=args.score_threshold
        )

        # --- Update Tracking State ---
        if centroid:
            # Found it! Switch to TRACKING mode and update position.
            status = TrackingStatus.TRACKING
            last_centroid = centroid
            frames_since_last_seen = 0
        else:
            # Didn't find it in this frame.
            frames_since_last_seen += 1
            if frames_since_last_seen > args.track_loss_threshold:
                # If lost for too long, revert to full-frame SEARCHING.
                status = TrackingStatus.SEARCHING
                last_centroid = None
                search_roi = None

        # 5. Render and display the output
        frame = render_frame(x_coords, y_coords, polarities, src.width, src.height)
        draw_hud(frame, status, last_centroid, search_roi)
        cv2.imshow("Drone Hotspot Tracker", frame)

        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
            break
    
    cv2.destroyAllWindows()
    print("Demo finished.")

if __name__ == "__main__":
    main()
