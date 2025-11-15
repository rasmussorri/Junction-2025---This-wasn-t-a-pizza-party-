import argparse
import time

import cv2
import numpy as np

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
    """Extract event coordinates and polarities for the given window."""
    event_indexes = time_order[win_start:win_stop]
    words = event_words[event_indexes].astype(np.uint32, copy=False)
    x_coords = (words & 0x3FFF).astype(np.int32, copy=False)
    y_coords = ((words >> 14) & 0x3FFF).astype(np.int32, copy=False)
    polarities_on = ((words >> 28) & 0xF) > 0
    return x_coords, y_coords, polarities_on


def get_frame(
    window: tuple[np.ndarray, np.ndarray, np.ndarray],
    width: int = 1280,
    height: int = 720,
    *,
    base_color: tuple[int, int, int] = (0, 0, 0),
    on_color: tuple[int, int, int] = (255, 255, 255),
    off_color: tuple[int, int, int] = (100, 100, 100),
) -> np.ndarray:
    """Render events into a frame for visualization."""
    x_coords, y_coords, polarities_on = window
    frame = np.full((height, width, 3), base_color, np.uint8)
    if len(x_coords) > 0:
        frame[y_coords[polarities_on], x_coords[polarities_on]] = on_color
        frame[y_coords[~polarities_on], x_coords[~polarities_on]] = off_color
    return frame


def find_drone_cluster(
    x_coords: np.ndarray, 
    y_coords: np.ndarray, 
    polarities_on: np.ndarray, 
    grid_size: int = 30, 
    min_density: int = 20,
    score_threshold: float = 300.0  # <-- NEW default
) -> tuple[tuple[int, int] | None, np.ndarray | None]:
    """
    Finds the drone by identifying the spatial grid cell with the
    best "propeller signature."
    """
    
    # 0. Handle empty frames
    if len(x_coords) == 0:
        return None, None

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

    # 3. --- NEW SCORING LOGIC ---
    
    # Total density (H_on + H_off)
    H_density = H_on + H_off
    
    # Polarity Balance (min(on, off) / max(on, off))
    # Add 1 to avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        H_balance = (np.minimum(H_on, H_off) + 1) / (np.maximum(H_on, H_off) + 1)
        H_balance[np.isnan(H_balance)] = 0 # Handle 0/0 case

    # --- THIS IS THE KEY CHANGE ---
    # Score = (Balance ^ 4) * log(Density)
    # 1. (H_balance ** 4) exponentially punishes imbalance.
    # 2. np.log1p(H_density) compresses density, making it a
    #    tie-breaker, not the dominant factor. (np.log1p(x) is log(x+1))
    H_score = (H_balance ** 4) * np.log1p(H_density)
    # --- END OF NEW SCORING LOGIC ---
    
    # 4. Find the winning cell
    
    # Ignore cells with very low density (background noise)
    H_score[H_density < min_density] = 0
    
    # Get the best score on the grid
    max_score = np.max(H_score)
    
    # --- THRESHOLD CHECK ---
    if max_score < score_threshold:
        return None, None
            
    # Get the 2D index (row, col) of the highest scoring cell
    idx = np.unravel_index(np.argmax(H_score), H_score.shape)
    
    # 5. Extract the drone cluster from the winning cell
    x_min, x_max = xedges[idx[0]], xedges[idx[0] + 1]
    y_min, y_max = yedges[idx[1]], yedges[idx[1] + 1]

    cluster_mask = (x_coords >= x_min) & (x_coords < x_max) & \
                   (y_coords >= y_min) & (y_coords < y_max)
                   
    if np.sum(cluster_mask) == 0:
        return None, None

    # 6. Calculate the precise centroid of this cluster
    centroid = (
        int(np.mean(x_coords[cluster_mask])),
        int(np.mean(y_coords[cluster_mask]))
    )

    return centroid, cluster_mask


def calculate_speed(
    current_centroid, prev_centroid, 
    current_time_us, prev_time_us
) -> tuple[float, tuple[float, float]]:
    """Calculate speed between two centroids."""
    if prev_centroid is None or prev_time_us is None:
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
    centroid: tuple[int, int] | None, 
    speed: float, 
    cluster_mask: np.ndarray | None = None, 
    x_coords: np.ndarray | None = None, 
    y_coords: np.ndarray | None = None
) -> None:
    """Draw tracking visualization on the frame."""
    
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
    # --- Tunable Parameters ---
    parser.add_argument(
        "--grid_size", type=int, default=30,
        help="Size (in pixels) of the analysis grid cells. (default: 30)"
    )
    parser.add_argument(
        "--min_density", type=int, default=20,
        help="Minimum events in a cell to be considered a candidate. (default: 20)"
    )
    # --- UPDATED THRESHOLD ---
    parser.add_argument(
        "--score_threshold", type=float, default=3.0,
        help="Minimum 'propeller score' (log(density) * balance^4). (default: 3.0)"
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
        
        centroid, cluster_mask = find_drone_cluster(
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
        
        if centroid is not None:
            prev_centroid = centroid
            prev_time_us = current_time_us
        else:
            prev_centroid = None
            prev_time_us = None
        
        frame = get_frame((x_coords, y_coords, polarities))
        draw_hud(frame, pacer, batch_range)
        draw_tracking_overlay(frame, centroid, speed, cluster_mask, x_coords, y_coords)
        
        cv2.imshow("Drone Tracker (Propeller Signature)", frame)

        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
            break
    
    cv2.destroyAllWindows()
    print("Demo finished.")


if __name__ == "__main__":
    main()