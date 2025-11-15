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

# --- 1. Understanding the Data and Core Concepts ---
#
# How is the .dat file imported and what is our data structure?
#
# 1. Loading the Raw Data (`DatFileSource`):
#    - When we create a `DatFileSource` object, the `evio` library reads the
#      entire binary `.dat` file into your computer's RAM.
#    - This raw data is stored primarily in a huge NumPy array called `src.event_words`.
#
# 2. The `event_words` NumPy Array:
#    - This is a massive, one-dimensional array where each element is a single
#      32-bit unsigned integer (a `uint32`).
#    - This integer is a highly efficient, packed representation of a single "event".
#      It encodes the X, Y coordinates, and the polarity all in one number to save space.
#
# 3. Decoding the `uint32` Event Word (The Transformation):
#    - A raw integer like `3090300416` is meaningless to us directly. We need to
#      "unpack" or "decode" it to get our usable data.
#    - This is done using fast bitwise operations inside the `get_window_events` function.
#      Here's a simplified breakdown of how a 32-bit integer is decoded:
#
#      [P] [UNUSED] [YYYYYYYYYYYYYY] [XXXXXXXXXXXXXX]
#      |      |           |                |
#      |      |           |                +-- Bits 0-13:  X coordinate
#      |      |           +------------------- Bits 14-27: Y coordinate
#      |      +------------------------------- Bits 28-30: Unused/Reserved
#      +--------------------------------------- Bit 31:    Polarity (1 for ON, 0 for OFF)
#
#    - `x = word & 0x3FFF`: This performs a bitwise AND to isolate the lower 14 bits, giving the X coordinate.
#    - `y = (word >> 14) & 0x3FFF`: This shifts the bits 14 places to the right, then isolates the next 14 bits for Y.
#    - `p = (word >> 28) & 0xF`: This shifts 28 places to get the polarity bits.
#
# This entire process is designed for extreme performance. Loading everything into RAM once
# and using bitwise operations for decoding is much faster than repeatedly reading from disk.
# ---

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
    Extracts and decodes a slice of event data from the main raw arrays.

    This function takes a start and end index for a specific time window,
    retrieves the corresponding raw 'event words', and decodes them into
    usable NumPy arrays for X, Y coordinates and polarity.

    Args:
        event_words: The full, raw array of packed 32-bit event integers.
        time_order: An array of indices that sorts the event_words chronologically.
        win_start: The starting index of the time window.
        win_stop: The ending index of the time window.

    Returns:
        A tuple containing:
        - x_coords (np.ndarray): The X positions of events in the window.
        - y_coords (np.ndarray): The Y positions of events in the window.
        - polarities_on (np.ndarray): A boolean array (True for ON, False for OFF).
        - event_indexes (np.ndarray): The original indices of the events in this window.
    """
    # Get the indices for the events that occurred in this specific time slice
    event_indexes = time_order[win_start:win_stop]
    # Retrieve the raw 32-bit integer words for these events
    words = event_words[event_indexes].astype(np.uint32, copy=False)
    
    # --- The Bitwise Transformation ---
    # Use fast bitwise operations to "unpack" the data from the integers.
    # This is the core of the high-speed decoding process.
    
    # Isolate the lowest 14 bits for the X coordinate
    x_coords = (words & 0x3FFF).astype(np.int32, copy=False)
    # Shift right by 14 bits, then isolate the next 14 bits for the Y coordinate
    y_coords = ((words >> 14) & 0x3FFF).astype(np.int32, copy=False)
    # Shift right by 28 bits to get the polarity bit
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
    Analyzes a set of events to find the most likely location of a drone.

    This is the core detection algorithm. It works by dividing the frame into a
    grid and calculating a "propeller score" for each grid cell. The score is
    high if a cell has both a high density of events and a good balance
    between ON (brightening) and OFF (darkening) events, which is the unique
    signature of a fast-spinning propeller.

    It then finds the single best cell (the "hotspot") and returns its location.

    Args:
        x_coords: X positions of events to analyze.
        y_coords: Y positions of events to analyze.
        polarities_on: Polarity of the events.
        width: The width of the sensor frame.
        height: The height of the sensor frame.
        grid_size: The side length (in pixels) of each analysis cell.
        min_density: The minimum number of events a cell must have to be considered.
        score_threshold: The minimum propeller score a cell must have to be valid.

    Returns:
        A tuple containing:
        - centroid (tuple[int, int] | None): The (x, y) coordinate of the detected drone, or None.
        - cluster_mask (np.ndarray | None): A boolean mask identifying the events belonging to the hotspot.
    """
    # If there are no events in this window, we can't find anything.
    if len(x_coords) == 0:
        return None, None

    # 1. Define the spatial grid for analysis.
    # We will divide the full frame into smaller cells to analyze them individually.
    bins_x = np.arange(0, width + grid_size, grid_size)
    bins_y = np.arange(0, height + grid_size, grid_size)

    # 2. Create 2D histograms of event density.
    # This is a very fast way to count how many ON and OFF events fall into each grid cell.
    H_on, xedges, yedges = np.histogram2d(
        x_coords[polarities_on], y_coords[polarities_on], bins=[bins_x, bins_y]
    )
    H_off, _, _ = np.histogram2d(
        x_coords[~polarities_on], y_coords[~polarities_on], bins=[bins_x, bins_y]
    )

    # 3. Calculate the "Propeller Signature" Score for every cell.
    # This score is the heart of the algorithm.
    H_density = H_on + H_off
    with np.errstate(divide='ignore', invalid='ignore'):
        # Balance Score: A value from 0 to 1. A score of 1.0 means a perfect
        # 50/50 split of ON and OFF events, which is the "flicker" we want.
        # We add 1 to the numerator and denominator to avoid division by zero.
        H_balance = (np.minimum(H_on, H_off) + 1) / (np.maximum(H_on, H_off) + 1)
        H_balance[np.isnan(H_balance)] = 0

    # Final Score: Combines density and balance. Using sqrt(density) makes the
    # score less dominated by raw event count and more sensitive to the *quality*
    # of the flicker (the balance).
    H_score = np.sqrt(H_density) * H_balance
    
    # 4. Filter out cells that are unlikely to be a drone.
    # Any cell with fewer events than `min_density` is probably just noise.
    H_score[H_density < min_density] = 0
    
    # 5. Find the single best cell (the "hotspot").
    # This is the cell with the highest propeller score in the entire grid.
    max_score = np.max(H_score)
    # If even the best score isn't good enough, then we haven't found anything.
    if max_score < score_threshold:
        return None, None  # No cell is good enough
    
    # Get the (row, column) index of the best cell.
    hotspot_idx = np.unravel_index(np.argmax(H_score), H_score.shape)
    
    # 6. Create a "cluster" of events from just the hotspot cell.
    # We get the pixel boundaries of our best cell...
    x_min, x_max = xedges[hotspot_idx[0]], xedges[hotspot_idx[0] + 1]
    y_min, y_max = yedges[hotspot_idx[1]], yedges[hotspot_idx[1] + 1]
    
    # ...and create a boolean mask to select only the events inside it.
    cluster_mask = (x_coords >= x_min) & (x_coords < x_max) & \
                   (y_coords >= y_min) & (y_coords < y_max)
    
    # Safety check: if for some reason no events are in the mask, exit.
    if np.sum(cluster_mask) == 0:
        return None, None

    # 7. Calculate the precise centroid (center point) of the hotspot cluster.
    # We take the average position of all events within our hotspot cell.
    centroid = (
        int(np.mean(x_coords[cluster_mask])),
        int(np.mean(y_coords[cluster_mask]))
    )

    return centroid, cluster_mask

def render_frame(
    x_coords: np.ndarray, y_coords: np.ndarray, polarities_on: np.ndarray,
    width: int, height: int
) -> np.ndarray:
    """
    Renders the events from a time window into a visual frame (an image).
    
    Args:
        x_coords: X positions of all events in the window.
        y_coords: Y positions of all events in the window.
        polarities_on: Boolean mask for event polarity.
        width: The width of the output frame.
        height: The height of the output frame.

    Returns:
        A NumPy array representing a BGR image.
    """
    # Start with a black frame
    frame = np.full((height, width, 3), (0, 0, 0), np.uint8)
    # Draw ON events as white pixels
    frame[y_coords[polarities_on], x_coords[polarities_on]] = (255, 255, 255)
    # Draw OFF events as gray pixels
    frame[y_coords[~polarities_on], x_coords[~polarities_on]] = (100, 100, 100)
    return frame

def draw_hud(frame: np.ndarray, status: TrackingStatus, centroid: tuple | None, roi: tuple | None) -> None:
    """
    Draws the Heads-Up Display (HUD) on the frame.

    This shows the current tracking status, the drone's position, and the
    search box (ROI) if the tracker is locked on.

    Args:
        frame: The image frame to draw on.
        status: The current TrackingStatus (SEARCHING or TRACKING).
        centroid: The current detected (x, y) position of the drone.
        roi: The (x1, y1, x2, y2) coordinates of the search box.
    """
    if status == TrackingStatus.TRACKING and centroid:
        # --- TRACKING MODE ---
        # Green indicates a successful track lock.
        color = (0, 255, 0)
        cv2.putText(frame, f"STATUS: TRACKING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw the ROI search box in cyan to show where the algorithm is looking.
        if roi:
            x1, y1, x2, y2 = roi
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)

        # Draw a crosshair at the drone's detected centroid.
        x, y = centroid
        cv2.line(frame, (x - 20, y), (x + 20, y), color, 2)
        cv2.line(frame, (x, y - 20), (x, y + 20), color, 2)
        
    else:
        # --- SEARCHING MODE ---
        # Red indicates the algorithm is searching the full frame for a new target.
        color = (0, 0, 255)
        cv2.putText(frame, "STATUS: SEARCHING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def main() -> None:
    parser = argparse.ArgumentParser(description="Optimized drone tracker using hotspot and track-lock logic.")
    parser.add_argument("dat", help="Path to .dat file")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed (1.0 is real time)")
    parser.add_argument("--window", type=float, default=20.0, help="Window duration in ms (default: 20.0)")
    parser.add_argument("--force-speed", action="store_true", help="Force playback speed by dropping windows")
    
    # --- Algorithm Tuning Parameters ---
    # These allow you to adjust the sensitivity of the detection algorithm.
    parser.add_argument("--grid_size", type=int, default=20, help="Size of analysis grid cells (default: 20)")
    parser.add_argument("--min_density", type=int, default=10, help="Minimum events in a cell to be a candidate (default: 10)")
    parser.add_argument("--score_threshold", type=float, default=15.0, help="Minimum propeller score for detection (default: 15.0)")
    
    # --- Track-Lock Tuning Parameters ---
    # These control the behavior of the "track-lock" state machine.
    parser.add_argument("--roi_size", type=int, default=150, help="Size of the search box (ROI) when tracking (default: 150)")
    parser.add_argument("--track_loss_threshold", type=int, default=5, help="Frames to wait before reverting to SEARCHING mode (default: 5)")
    
    args = parser.parse_args()

    # 1. Load data source using the evio library.
    # This loads the entire file into RAM for high-speed access.
    print(f"Loading {args.dat}...")
    src = DatFileSource(args.dat, width=1280, height=720, window_length_us=args.window * 1000)
    print(f"File loaded. Resolution: {src.width}x{src.height}")

    # 2. Setup visualization window and the Pacer for real-time playback.
    pacer = Pacer(speed=args.speed, force_speed=args.force_speed)
    cv2.namedWindow("Drone Hotspot Tracker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Drone Hotspot Tracker", src.width, src.height)

    # 3. Initialize the state machine for the track-lock loop.
    status = TrackingStatus.SEARCHING  # Start by searching the full frame.
    last_centroid = None               # Last known position of the drone.
    frames_since_last_seen = 0         # Counter for track loss logic.
    search_roi = None                  # The (x1, y1, x2, y2) search box.

    print("Starting tracking loop... Press 'q' or ESC to quit.")
    
    # 4. Main Application Loop
    # The pacer yields data for each time window at the correct playback speed.
    for batch_range in pacer.pace(src.ranges()):
        
        # Get all events for the current time window.
        x_coords, y_coords, polarities, event_indices = get_window_events(
            src.event_words, src.order, batch_range.start, batch_range.stop
        )

        # --- Track-Lock State Machine ---
        if status == TrackingStatus.TRACKING and last_centroid:
            # --- A. We are locked on. Only search in a small box. ---
            
            # Define the search box (Region of Interest or ROI) around the last known position.
            half_roi = args.roi_size // 2
            x1 = max(0, last_centroid[0] - half_roi)
            y1 = max(0, last_centroid[1] - half_roi)
            x2 = min(src.width, last_centroid[0] + half_roi)
            y2 = min(src.height, last_centroid[1] + half_roi)
            search_roi = (x1, y1, x2, y2)

            # Create a boolean mask to select only the events inside the ROI.
            # This is a very fast NumPy operation.
            roi_mask = (x_coords >= x1) & (x_coords < x2) & (y_coords >= y1) & (y_coords < y2)
            
            # Overwrite our event arrays with the filtered, smaller arrays.
            x_roi, y_roi, pol_roi = x_coords[roi_mask], y_coords[roi_mask], polarities[roi_mask]
        else:
            # --- B. We are searching. Use the full frame. ---
            x_roi, y_roi, pol_roi = x_coords, y_coords, polarities
            search_roi = None

        # Run the detection algorithm on the selected events (either full-frame or ROI).
        centroid, _ = find_drone_hotspot(
            x_roi, y_roi, pol_roi,
            width=src.width, height=src.height,
            grid_size=args.grid_size,
            min_density=args.min_density,
            score_threshold=args.score_threshold
        )

        # --- Update Tracking State for the *Next* Frame ---
        if centroid:
            # We found it!
            # Set status to TRACKING for the next loop iteration.
            status = TrackingStatus.TRACKING
            # Update the drone's last known position.
            last_centroid = centroid
            # Reset the track loss counter.
            frames_since_last_seen = 0
        else:
            # We did not find it in this frame.
            frames_since_last_seen += 1
            # If we've lost the drone for too many consecutive frames...
            if frames_since_last_seen > args.track_loss_threshold:
                # ...revert to full-frame SEARCHING mode to find it again.
                status = TrackingStatus.SEARCHING
                last_centroid = None
                search_roi = None

        # 5. Render the final frame and display it.
        # We use the original, full set of events for rendering so we see the whole picture.
        frame = render_frame(x_coords, y_coords, polarities, src.width, src.height)
        # Draw the HUD elements on top.
        draw_hud(frame, status, last_centroid, search_roi)
        cv2.imshow("Drone Hotspot Tracker", frame)

        # Exit if the user presses 'q' or the ESC key.
        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
            break
    
    cv2.destroyAllWindows()
    print("Demo finished.")

if __name__ == "__main__":
    main()
