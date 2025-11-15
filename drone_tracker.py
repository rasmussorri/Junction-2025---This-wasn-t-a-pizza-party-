import argparse
import time

import cv2
import numpy as np

from evio.core.pacer import Pacer
from evio.source.dat_file import BatchRange, DatFileSource


def get_window(
    event_words: np.ndarray,
    time_order: np.ndarray,
    win_start: int,
    win_stop: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract event coordinates and polarities for the given window."""
    # get indexes corresponding to events within the window
    event_indexes = time_order[win_start:win_stop]
    words = event_words[event_indexes].astype(np.uint32, copy=False)
    x_coords = (words & 0x3FFF).astype(np.int32, copy=False)
    y_coords = ((words >> 14) & 0x3FFF).astype(np.int32, copy=False)
    pixel_polarity = ((words >> 28) & 0xF) > 0

    return x_coords, y_coords, pixel_polarity


def get_frame(
    window: tuple[np.ndarray, np.ndarray, np.ndarray],
    width: int = 1280,
    height: int = 720,
    *,
    base_color: tuple[int, int, int] = (0, 0, 0),  # black background
    on_color: tuple[int, int, int] = (255, 255, 255),  # white
    off_color: tuple[int, int, int] = (100, 100, 100),  # gray
) -> np.ndarray:
    """Render events into a frame."""
    x_coords, y_coords, polarities_on = window
    frame = np.full((height, width, 3), base_color, np.uint8)
    
    if len(x_coords) > 0:
        frame[y_coords[polarities_on], x_coords[polarities_on]] = on_color
        frame[y_coords[~polarities_on], x_coords[~polarities_on]] = off_color

    return frame


def find_drone_cluster(x_coords, y_coords, polarities_on, eps=30, min_samples=1000, **kwargs):
    """
    Find the dense cluster representing the drone by looking for a
    high-frequency spatio-temporal signature (propellers).
    
    This is proxied by finding the region with the highest density
    of *both* ON and OFF events.
    
    Args:
        x_coords: Array of x coordinates
        y_coords: Array of y coordinates
        polarities_on: Boolean array (True for ON, False for OFF)
        eps: Grid cell size for density estimation
        
    Returns:
        centroid: (x, y) tuple of the drone's position, or None if not found
        cluster_points: Boolean mask of points belonging to the main cluster
    """
    if len(x_coords) == 0:
        return None, None

    # Define the grid
    grid_size = int(eps)
    bins_x = range(0, 1280 + grid_size, grid_size)
    bins_y = range(0, 720 + grid_size, grid_size)
    
    # 1. Create a histogram for ON events
    x_on = x_coords[polarities_on]
    y_on = y_coords[polarities_on]
    
    H_on, _, _ = np.histogram2d(
        x_on, y_on, 
        bins=[bins_x, bins_y]
    )
    
    # 2. Create a histogram for OFF events
    x_off = x_coords[~polarities_on]
    y_off = y_coords[~polarities_on]

    H_off, xedges, yedges = np.histogram2d(
        x_off, y_off,
        bins=[bins_x, bins_y]
    )

    # 3. Find the "Propeller Signature"
    # A drone cluster will be dense in *both* ON and OFF events.
    # A tree branch will be dense in one or the other, but not both.
    # We multiply them to find the cells with the best joint-density.
    H_propeller_score = H_on * H_off
    
    # 4. Find the best candidate cell
    flat_idx = np.argmax(H_propeller_score)
    
    # If no cell has both ON and OFF events, H is all zero
    if H_propeller_score.flat[flat_idx] == 0:
        return None, None
        
    # Convert flat index back to 2D
    idx = np.unravel_index(flat_idx, H_propeller_score.shape)
    
    # Get the center of this winning cell
    center_x = (xedges[idx[0]] + xedges[idx[0] + 1]) / 2
    center_y = (yedges[idx[1]] + yedges[idx[1] + 1]) / 2

    # 5. Get all points around this winning cell to form the cluster
    cluster_radius = eps * 1.5  # Look in a radius around the cell center
    distances_sq = (x_coords - center_x)**2 + (y_coords - center_y)**2
    cluster_mask = distances_sq < (cluster_radius**2)
    
    # Ensure minimum number of points
    if np.sum(cluster_mask) < min_samples:
        return None, None
        
    # Calculate the precise centroid of all points in this cluster
    best_centroid = (
        int(np.mean(x_coords[cluster_mask])), 
        int(np.mean(y_coords[cluster_mask]))
    )
    
    return best_centroid, cluster_mask


def calculate_speed(current_centroid, prev_centroid, current_time_us, prev_time_us):
    """
    Calculate speed between two centroids.
    
    Args:
        current_centroid: (x, y) tuple of current position
        prev_centroid: (x, y) tuple of previous position
        current_time_us: Current timestamp in microseconds
        prev_time_us: Previous timestamp in microseconds
        
    Returns:
        speed: Speed in pixels/second
        velocity: (vx, vy) velocity vector in pixels/second
    """
    if prev_centroid is None or prev_time_us is None:
        return 0, (0, 0)
    
    # Calculate displacement
    dx = current_centroid[0] - prev_centroid[0]
    dy = current_centroid[1] - prev_centroid[1]
    
    # Calculate time difference in seconds
    dt_seconds = (current_time_us - prev_time_us) / 1e6
    
    if dt_seconds == 0:
        return 0, (0, 0)
    
    # Calculate velocity (pixels/second)
    vx = dx / dt_seconds
    vy = dy / dt_seconds
    
    # Calculate scalar speed
    speed = np.sqrt(vx**2 + vy**2)
    
    return speed, (vx, vy)


def draw_tracking_overlay(frame, centroid, speed, cluster_mask=None, x_coords=None, y_coords=None):
    """
    Draw tracking visualization on the frame.
    
    Args:
        frame: The frame to draw on
        centroid: (x, y) tuple of the drone's position
        speed: Speed in pixels/second
        cluster_mask: Boolean mask of cluster points (optional)
        x_coords: X coordinates of all events (optional)
        y_coords: Y coordinates of all events (optional)
    """
    if centroid is None:
        # Draw "LOST" message
        cv2.putText(
            frame,
            "DRONE: LOST",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),  # Red
            2,
            cv2.LINE_AA,
        )
        return
    
    x, y = centroid
    
    # Draw a circle at the centroid
    cv2.circle(frame, (x, y), 10, (0, 255, 0), 2)  # Green circle
    
    # Draw a bounding box around the cluster
    if cluster_mask is not None and x_coords is not None and y_coords is not None:
        cluster_x = x_coords[cluster_mask]
        cluster_y = y_coords[cluster_mask]
        
        if len(cluster_x) > 0:
            min_x, max_x = int(np.min(cluster_x)), int(np.max(cluster_x))
            min_y, max_y = int(np.min(cluster_y)), int(np.max(cluster_y))
            
            # Add some padding
            padding = 20
            min_x = max(0, min_x - padding)
            min_y = max(0, min_y - padding)
            max_x = min(frame.shape[1], max_x + padding)
            max_y = min(frame.shape[0], max_y + padding)
            
            # Draw rectangle
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
    
    # Draw crosshair at centroid
    crosshair_size = 20
    cv2.line(frame, (x - crosshair_size, y), (x + crosshair_size, y), (0, 255, 0), 2)
    cv2.line(frame, (x, y - crosshair_size), (x, y + crosshair_size), (0, 255, 0), 2)
    
    # Display position and speed
    cv2.putText(
        frame,
        f"DRONE: ({x}, {y})",
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    
    cv2.putText(
        frame,
        f"SPEED: {speed:.1f} px/s",
        (10, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def draw_hud(
    frame: np.ndarray,
    pacer: Pacer,
    batch_range: BatchRange,
    *,
    color: tuple[int, int, int] = (0, 0, 0),  # black by default
) -> None:
    """Overlay timing info: wall time, recording time, and playback speed."""
    if pacer._t_start is None or pacer._e_start is None:
        return

    wall_time_s = time.perf_counter() - pacer._t_start
    rec_time_s = max(0.0, (batch_range.end_ts_us - pacer._e_start) / 1e6)

    if pacer.force_speed:
        first_row_str = (
            f"speed={pacer.speed:.2f}x"
            f"  drops/ms={pacer.instantaneous_drop_rate:.2f}"
            f"  avg(drops/ms)={pacer.average_drop_rate:.2f}"
        )
    else:
        first_row_str = (
            f"(target) speed={pacer.speed:.2f}x  force_speed = False, no drops"
        )

    second_row_str = f"wall={wall_time_s:7.3f}s  rec={rec_time_s:7.3f}s"

    # first row
    cv2.putText(
        frame,
        first_row_str,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )

    # second row
    cv2.putText(
        frame,
        second_row_str,
        (8, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Track drone in event camera data using DBSCAN clustering")
    parser.add_argument(
        "dat",
        nargs='?',
        default="C:\\Users\\Henri\\Downloads\\Junction\\Data\\drone_moving-20251114T191633Z-1-002\\drone_moving\\drone_moving.dat",
        help="Path to .dat file"
    )
    parser.add_argument(
        "--window", type=float, default=5, help="Window duration in ms"
    )
    parser.add_argument(
        "--speed", type=float, default=2, help="Playback speed (1 is real time)"
    )
    parser.add_argument(
        "--force-speed",
        action="store_true",
        help="Force the playback speed by dropping windows",
    )
    parser.add_argument(
        "--eps", type=float, default=30,
        help="DBSCAN eps parameter (max distance between points in a cluster)"
    )
    parser.add_argument(
        "--min-samples", type=int, default=10,
        help="DBSCAN min_samples parameter (min points to form a cluster)"
    )
    args = parser.parse_args()

    print(f"Loading data file: {args.dat}")
    
    try:
        src = DatFileSource(
            args.dat, width=1280, height=720, window_length_us=args.window * 1000
        )
        print(f"Data loaded successfully!")
        print(f"Resolution: {src.width}x{src.height}")
        print(f"Number of event batches: {len(src)}")
    except Exception as e:
        print(f"ERROR loading data file: {e}")
        return

    # Enforce playback speed via dropping:
    pacer = Pacer(speed=args.speed, force_speed=args.force_speed)

    # Tracking state
    prev_centroid = None
    prev_time_us = None

    print("Opening visualization window...")
    cv2.namedWindow("Drone Tracker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Drone Tracker", 1280, 720)
    cv2.setWindowProperty("Drone Tracker", cv2.WND_PROP_TOPMOST, 1)
    print("Starting tracking loop... Press 'q' or ESC to quit.")
    
    frame_count = 0
    for batch_range in pacer.pace(src.ranges()):
        frame_count += 1
        
        # Get event window
        x_coords, y_coords, polarities = get_window(
            src.event_words,
            src.order,
            batch_range.start,
            batch_range.stop,
        )
        
        # Debug first frame
        if frame_count == 1:
            print(f"First frame - Number of events: {len(x_coords)}")
            print(f"X range: {x_coords.min() if len(x_coords) > 0 else 'N/A'} to {x_coords.max() if len(x_coords) > 0 else 'N/A'}")
            print(f"Y range: {y_coords.min() if len(y_coords) > 0 else 'N/A'} to {y_coords.max() if len(y_coords) > 0 else 'N/A'}")
        
        # Find drone cluster using DBSCAN
        centroid, cluster_mask = find_drone_cluster(
            x_coords, y_coords, polarities, 
            eps=args.eps, 
            min_samples=args.min_samples
        )
        
        # Debug clustering on first frame
        if frame_count == 1 and cluster_mask is not None:
            cluster_size = np.sum(cluster_mask)
            print(f"Cluster found: {cluster_size} events ({cluster_size/len(x_coords)*100:.1f}% of total)")
            print(f"Centroid: {centroid}")
            print(f"DBSCAN params: eps={args.eps}, min_samples={args.min_samples}")
        
        # Calculate speed
        current_time_us = batch_range.end_ts_us
        speed, velocity = calculate_speed(centroid, prev_centroid, current_time_us, prev_time_us)
        
        # Render frame
        frame = get_frame((x_coords, y_coords, polarities))
        
        # Draw HUD
        draw_hud(frame, pacer, batch_range)
        
        # Draw tracking overlay
        draw_tracking_overlay(frame, centroid, speed, cluster_mask, x_coords, y_coords)
        
        # Update state for next iteration
        if centroid is not None:
            prev_centroid = centroid
            prev_time_us = current_time_us
        
        # Display
        cv2.imshow("Drone Tracker", frame)
        
        # Print status every 30 frames
        if frame_count == 1:
            print(f"Window should be visible now! Frame {frame_count} displayed.")
            print(f"Events in frame: {len(x_coords)}, Centroid: {centroid}")
        elif frame_count % 30 == 0:
            print(f"Frame {frame_count} - Centroid: {centroid}, Speed: {speed:.1f} px/s")

        # Increased wait time to ensure window updates properly
        key = cv2.waitKey(100) & 0xFF
        if key in (27, ord("q")):  # ESC or 'q'
            break
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
