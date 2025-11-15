import argparse
import time
from enum import Enum  # Added from Code 2

import cv2
import numpy as np
from scipy import ndimage  # Required for Code 1's cluster logic

# We need these from the 'evio' library
try:
    from evio.core.pacer import Pacer
    from evio.source.dat_file import BatchRange, DatFileSource
except ImportError:
    print("ERROR: evio library not found.")
    print("Please clone the repo and run 'uv sync' as per the hackathon plan.")
    exit(1)


# --- State Machine Enum (from Code 2) ---
class TrackingStatus(Enum):
    """Defines the current state of the tracker."""
    SEARCHING = 1  # Looking for a new target across the full frame
    TRACKING = 2   # Locked onto a target, searching in a small ROI


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
    width: int,
    height: int,
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
    width: int, # Added width/height to make function independent of hardcoded values
    height: int,
    grid_size: int = 30, 
    min_density: int = 20,
    score_threshold: float = 3000.0,
    prev_centroid: tuple[int, int] | None = None, # This is our "state" input
    search_region_size: int = 300 # This is our "ROI size"
) -> tuple[tuple[int, int] | None, np.ndarray | None, int]:
    """
    Finds the drone by identifying ALL spatial grid cells with
    a strong "propeller signature" using adaptive multi-scale detection.

    If 'prev_centroid' is provided, it will first perform a fast, focused
    search within the 'search_region_size' (TRACKING mode).
    If 'prev_centroid' is None, it will perform a full-frame search (SEARCHING mode).
    """
    
    # 0. Handle empty frames
    if len(x_coords) == 0:
        return None, None, 0

    # 1. HYBRID LOGIC: FOCUSED SEARCH (TRACKING) vs. FULL SEARCH (SEARCHING)
    # This logic is now driven entirely by whether 'prev_centroid' is provided.
    search_mask = np.ones(len(x_coords), dtype=bool)
    if prev_centroid is not None:
        # --- TRACKING MODE ---
        # We have a previous position. Create a focused search mask.
        px, py = prev_centroid
        distances = np.sqrt((x_coords - px)**2 + (y_coords - py)**2)
        search_mask = distances <= search_region_size
        
        # If we have enough events in search region, use focused search
        if np.sum(search_mask) >= 50:
            x_coords_search = x_coords[search_mask]
            y_coords_search = y_coords[search_mask]
            polarities_search = polarities_on[search_mask]
        else:
            # Fall back to full frame if ROI is empty (e.g., drone moved fast)
            x_coords_search = x_coords
            y_coords_search = y_coords
            polarities_search = polarities_on
            search_mask = np.ones(len(x_coords), dtype=bool)
    else:
        # --- SEARCHING MODE ---
        # No previous centroid. Search the entire frame.
        x_coords_search = x_coords
        y_coords_search = y_coords
        polarities_search = polarities_on

    # 2. Adaptive thresholding based on event density (from Code 1)
    total_events = len(x_coords_search)
    frame_area = width * height
    if frame_area > 0:
        avg_density = total_events / (frame_area / (grid_size ** 2))
    else:
        avg_density = 0
    adaptive_min_density = max(3, int(min_density * min(1.0, (avg_density / 5.0) ** 0.7)))
    
    # 3. Multi-scale detection (from Code 1)
    best_centroid = None
    best_mask = None
    best_score = 0
    best_num_propellers = 0
    
    for scale_factor in [0.7, 1.0, 1.5]:
        current_grid_size = int(grid_size * scale_factor)
        
        # 3a. Define the spatial grid boundaries
        bins_x = np.arange(0, width + current_grid_size, current_grid_size)
        bins_y = np.arange(0, height + current_grid_size, current_grid_size)

        # 3b. Create density histograms for ON and OFF events
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

        # 3c. Calculate "Propeller Signature" Score
        H_density = H_on + H_off
        with np.errstate(divide='ignore', invalid='ignore'):
            H_balance = (np.minimum(H_on, H_off) + 1) / (np.maximum(H_on, H_off) + 1)
            H_balance[np.isnan(H_balance)] = 0
        H_score = H_density * (H_balance ** 1.5)
        
        # 3d. Find ALL cells above adaptive threshold
        adaptive_score_threshold = max(
            score_threshold * 0.1,
            np.percentile(H_score[H_score > 0], 90) if np.any(H_score > 0) else 0
        )
        valid_density = H_density >= float(adaptive_min_density)
        valid_balance = H_balance >= 0.3
        valid_cells = valid_density & valid_balance & (H_score >= float(adaptive_score_threshold))
        
        # 3e. Cluster cells using scipy (from Code 1)
        if np.any(valid_cells):
            try:
                labeled_array, num_features = ndimage.label(valid_cells)  # type: ignore
                
                if num_features > 0:
                    cluster_sizes = ndimage.sum(valid_cells, labeled_array, range(1, num_features + 1))
                    largest_cluster_label = np.argmax(cluster_sizes) + 1
                    valid_cells = labeled_array == largest_cluster_label
            except Exception:
                pass
        
        num_propellers = int(np.sum(valid_cells))
        if num_propellers == 0:
            continue
        
        # 3f. Extract events from ALL valid cells
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
        
        # 3g. Score this detection
        detection_score = np.sum(H_score[valid_cells]) * num_propellers
        
        # Bonus for proximity (only applies if we are in TRACKING mode)
        if prev_centroid is not None:
            distance = np.sqrt((centroid[0] - prev_centroid[0])**2 + 
                             (centroid[1] - prev_centroid[1])**2)
            proximity_bonus = max(0, 1.0 - distance / search_region_size) * detection_score * 0.5
            detection_score += proximity_bonus
        
        # 3h. Keep best detection across scales
        if detection_score > best_score:
            best_score = detection_score
            best_centroid = centroid
            best_num_propellers = num_propellers
            
            # Map local mask back to full coordinate arrays
            # This is the magic: it works whether search_mask was full or partial
            best_mask = np.zeros(len(x_coords), dtype=bool)
            best_mask[search_mask] = combined_cluster_mask_local

    return best_centroid, best_mask, best_num_propellers


def calculate_speed(
    current_centroid, prev_centroid, 
    current_time_us, prev_time_us
) -> tuple[float, tuple[float, float]]:
    """Calculate speed between two centroids (from Code 1)."""
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


def draw_tracking_overlay(
    frame: np.ndarray, 
    status: TrackingStatus, # Added status
    centroid: tuple[int, int] | None, 
    speed: float,
    avg_propellers: int = 0,
    cluster_mask: np.ndarray | None = None, 
    x_coords: np.ndarray | None = None, 
    y_coords: np.ndarray | None = None,
    search_roi: tuple[int, int, int, int] | None = None # Added ROI for drawing
) -> None:
    """Draw tracking visualization on the frame (from Code 1, enhanced)."""
    
    if status == TrackingStatus.SEARCHING:
        # Draw "SEARCHING" message
        cv2.putText(
            frame, "STATUS: SEARCHING", (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA
        )
        cv2.putText(
            frame, "Looking for drone...", (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA
        )
        return
    
    # --- TRACKING Mode --- (or if centroid was just lost)
    if centroid is None:
        # We were tracking, but just lost it
        cv2.putText(
            frame, "STATUS: TRACKING (LOST)", (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA
        )
        return

    # We are actively tracking a centroid
    x, y = centroid
    color = (0, 255, 0) # Green for active track
    
    # Draw a bounding box around the cluster (from Code 1)
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
    
    # Draw crosshair at centroid
    crosshair_size = 20
    cv2.line(frame, (x - crosshair_size, y), (x + crosshair_size, y), color, 2)
    cv2.line(frame, (x, y - crosshair_size), (x, y + crosshair_size), color, 2)

    # Draw the search ROI box (from Code 2)
    if search_roi:
        cv2.rectangle(frame, (search_roi[0], search_roi[1]), (search_roi[2], search_roi[3]), (255, 255, 0), 1) # Cyan
    
    # Display tracking info
    cv2.putText(
        frame, f"STATUS: TRACKING", (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
    )
    cv2.putText(
        frame, f"POS: ({x}, {y})", (10, 110),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
    )
    cv2.putText(
        frame, f"SPEED: {speed:.1f} px/s", (10, 140),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
    )
    if avg_propellers > 0:
        cv2.putText(
            frame, f"PROPELLERS: {avg_propellers}", (10, 170),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
        )


def draw_hud(
    frame: np.ndarray,
    pacer: Pacer,
    batch_range: BatchRange,
    *,
    color: tuple[int, int, int] = (255, 255, 0),  # Cyan
) -> None:
    """Overlay timing info: wall time, recording time, and playback speed (from Code 1)."""
    if pacer._t_start is None or pacer._e_start is None:
        return

    wall_time_s = time.perf_counter() - pacer._t_start
    rec_time_s = max(0.0, (batch_range.end_ts_us - pacer._e_start) / 1e6)

    first_row_str = f"(target) speed={pacer.speed:.2f}x"
    if pacer.force_speed:
        first_row_str = (
            f"speed={pacer.speed:.2f}x"
            f"  avg(drops/ms)={pacer.average_drop_rate:.2f}"
        )

    second_row_str = f"wall={wall_time_s:7.3f}s   rec={rec_time_s:7.3f}s"

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
        description="Hybrid drone tracker (Robust Detection + Speedy Tracking)"
    )
    parser.add_argument(
        "dat",
        nargs='?',
        default="C:\\Users\\Henri\\Downloads\\Junction\\Data\\fred-1-20251114T194658Z-1-001\\fred-1\\Event\\events.dat",
        #"C:\\Users\\Henri\\Downloads\\Junction\\Data\\drone_moving-20251114T191633Z-1-002\\drone_moving\\drone_moving.dat",
        #"C:\\Users\\Henri\\Downloads\\Junction\\Data\\fred-1-20251114T194658Z-1-001\\fred-1\\Event\\events.dat",
        help="Path to .dat file (e.g., path/to/drone_moving.dat)"
    )
    # --- Playback Args ---
    parser.add_argument(
        "--window", type=float, default=10.0, 
        help="Window duration in ms (default: 20.0)"
    )
    parser.add_argument(
        "--speed", type=float, default=10.0, 
        help="Playback speed (1.0 is real time)"
    )
    parser.add_argument(
        "--force-speed", action="store_true", default=True,
        help="Force the playback speed by dropping windows",
    )
    # --- Code 1 Detection Args ---
    parser.add_argument(
        "--grid_size", type=int, default=15,
        help="Size (in pixels) of the analysis grid cells."
    )
    parser.add_argument(
        "--min_density", type=int, default=5,
        help="Minimum events in a cell to be considered a candidate."
    )
    parser.add_argument(
        "--score_threshold", type=float, default=2000.0,
        help="Minimum 'propeller score' (density * balance) to trigger a detection."
    )
    # --- Code 2 Tracking Args ---
    parser.add_argument(
        "--roi_size", type=int, default=300,
        help="Size of the search box (ROI) when tracking. (default: 300)"
    )
    parser.add_argument(
        "--track_loss_threshold", type=int, default=5,
        help="Frames to wait before reverting to SEARCHING mode (default: 5)"
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
    cv2.namedWindow("Hybrid Drone Tracker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hybrid Drone Tracker", src.width, src.height)

    # 3. Initialize HYBRID tracking state
    status = TrackingStatus.SEARCHING  # Start by searching
    last_centroid = None               # Last known position
    frames_since_last_seen = 0         # Counter for track loss
    prev_time_us = None                # For speed calculation
    
    # Propeller count tracking (from Code 1)
    propeller_history = []
    propeller_timestamps = []
    rolling_window_us = 10000000  # 10 seconds in microseconds

    print("Starting hybrid tracking loop... Press 'q' or ESC to quit.")
    print(f"Using settings: grid_size={args.grid_size}, score_threshold={args.score_threshold}")
    print(f"Tracking settings: roi_size={args.roi_size}, track_loss={args.track_loss_threshold} frames")
    
    # 4. Main Loop
    for batch_range in pacer.pace(src.ranges()):
        
        # 4a. Get all events in this window
        x_coords, y_coords, polarities = get_window(
            src.event_words,
            src.order,
            batch_range.start,
            batch_range.stop,
        )
        
        # 4b. Determine search parameters based on state
        # This is the core of the hybrid logic!
        if status == TrackingStatus.SEARCHING:
            # Full-frame search. find_drone_cluster will use its full logic.
            centroid_to_use_for_search = None
            current_roi_size = src.width # Not really used, but conceptually
        else: # TRACKING
            # Focused search. Tell find_drone_cluster to search around last_centroid.
            centroid_to_use_for_search = last_centroid
            current_roi_size = args.roi_size
        
        # 4c. Run the robust detection function
        # It will be fast if in TRACKING mode (prev_centroid != None)
        # It will be robust if in SEARCHING mode (prev_centroid == None)
        centroid, cluster_mask, num_propellers = find_drone_cluster(
            x_coords, y_coords, polarities, 
            width=src.width, height=src.height,
            grid_size=args.grid_size, 
            min_density=args.min_density,
            score_threshold=args.score_threshold,
            prev_centroid=centroid_to_use_for_search,
            search_region_size=current_roi_size
        )
        
        # 4d. Calculate speed
        current_time_us = batch_range.end_ts_us
        speed, velocity = calculate_speed(
            centroid, last_centroid, 
            current_time_us, prev_time_us
        )
        
        # 4e. Update propeller count history (from Code 1)
        if num_propellers > 0:
            propeller_history.append(num_propellers)
            propeller_timestamps.append(current_time_us)
            while propeller_timestamps and (current_time_us - propeller_timestamps[0]) > rolling_window_us:
                propeller_history.pop(0)
                propeller_timestamps.pop(0)
        
        avg_propellers = int(np.ceil(np.mean(propeller_history))) if propeller_history else 0
        
        # 4f. Update state machine (from Code 2)
        if centroid:
            # We found it! Lock on.
            status = TrackingStatus.TRACKING
            last_centroid = centroid
            prev_time_us = current_time_us
            frames_since_last_seen = 0
        else:
            # We did not find it in this frame.
            frames_since_last_seen += 1
            if frames_since_last_seen > args.track_loss_threshold:
                # We've lost the track for too long. Revert to full search.
                status = TrackingStatus.SEARCHING
                last_centroid = None # This will trigger the full search next frame
        
        # 4g. Render the frame
        frame = get_frame((x_coords, y_coords, polarities), src.width, src.height)
        draw_hud(frame, pacer, batch_range)
        
        # Define the visual ROI box for drawing
        search_roi_box = None
        if status == TrackingStatus.TRACKING and last_centroid:
            half_roi = args.roi_size // 2
            x1 = max(0, last_centroid[0] - half_roi)
            y1 = max(0, last_centroid[1] - half_roi)
            x2 = min(src.width, last_centroid[0] + half_roi)
            y2 = min(src.height, last_centroid[1] + half_roi)
            search_roi_box = (x1, y1, x2, y2)
        
        # Use the enhanced overlay function
        draw_tracking_overlay(
            frame, status, last_centroid, speed, avg_propellers, 
            cluster_mask, x_coords, y_coords, search_roi_box
        )
        
        cv2.imshow("Hybrid Drone Tracker", frame)

        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
            break
    
    cv2.destroyAllWindows()
    print("Demo finished.")


if __name__ == "__main__":
    main()