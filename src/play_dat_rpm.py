"""Enhanced play_dat.py with RPM tracking and dynamic movement following.

This script extends the evio playback functionality with:
- RPM tracking for rotating objects
- Dynamic ROI that follows movement
- Real-time RPM display
"""

import argparse  # noqa: INP001
import sys
from pathlib import Path
import time

import cv2
import numpy as np

# Add evio to path - handle both running from Junction 2025 and evio directories
script_dir = Path(__file__).parent
junction_dir = script_dir.parent
evio_dir = junction_dir / "evio"

# Add evio src to path
sys.path.insert(0, str(evio_dir / "src"))

# Add src directory to path for rpm_tracker import
sys.path.insert(0, str(script_dir))

from evio.core.pacer import Pacer
from evio.core.recording import open_dat
from evio.source.dat_file import BatchRange, DatFileSource

# Import our custom RPM tracker
from rpm_tracker import RPMTracker

# Global variables for slider callbacks
slider_state = {
    "window_ms": 10.0,
    "speed": 1.0,
    "force_speed": False,
    "window_changed": False,
    "dat_path": None,
    "width": 1280,
    "height": 720,
}


def on_window_change(val: int) -> None:
    """Callback for window slider (val is in 0.1ms units)."""
    slider_state["window_ms"] = max(0.1, float(val) / 10.0)
    slider_state["window_changed"] = True


def on_speed_change(val: int) -> None:
    """Callback for speed slider (val is in 0.01x units)."""
    slider_state["speed"] = max(0.01, float(val) / 100.0)


def detect_movement_center(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    width: int,
    height: int,
    grid_size: int = 32,
) -> tuple[int, int] | None:
    """
    Detect the center of movement by analyzing event density in a grid.
    
    Uses a grid-based approach to find the center of mass of event activity,
    which naturally follows moving objects like rotating fan blades.
    
    Args:
        x_coords: X coordinates of events
        y_coords: Y coordinates of events
        width: Image width
        height: Image height
        grid_size: Size of grid cells for density analysis
        
    Returns:
        Tuple of (center_x, center_y) or None if no events
    """
    if len(x_coords) == 0:
        return None
    
    # Filter valid coordinates
    valid = (x_coords >= 0) & (x_coords < width) & (y_coords >= 0) & (y_coords < height)
    if not np.any(valid):
        return None
    
    x_valid = x_coords[valid]
    y_valid = y_coords[valid]
    
    # Create a grid and count events in each cell using numpy
    grid_w = width // grid_size
    grid_h = height // grid_size
    
    # Map coordinates to grid cells
    grid_x = np.clip(x_valid // grid_size, 0, grid_w - 1)
    grid_y = np.clip(y_valid // grid_size, 0, grid_h - 1)
    
    # Count events in each grid cell
    density_grid = np.zeros((grid_h, grid_w), dtype=np.float32)
    np.add.at(density_grid, (grid_y, grid_x), 1.0)
    
    # Find center of mass
    total_density = density_grid.sum()
    if total_density > 0:
        # Calculate weighted center using meshgrid
        y_indices, x_indices = np.mgrid[0:grid_h, 0:grid_w]
        center_x_grid = (x_indices * density_grid).sum() / total_density
        center_y_grid = (y_indices * density_grid).sum() / total_density
        
        # Convert back to pixel coordinates
        center_x = int(center_x_grid * grid_size + grid_size // 2)
        center_y = int(center_y_grid * grid_size + grid_size // 2)
        
        # Clamp to image bounds
        center_x = max(0, min(width - 1, center_x))
        center_y = max(0, min(height - 1, center_y))
        
        return (center_x, center_y)
    
    return None


def get_timestamps_from_source(src: DatFileSource) -> np.ndarray:
    """
    Get timestamps from DatFileSource.
    
    Uses the timestamps property if available, otherwise reconstructs from recording.
    """
    if hasattr(src, "timestamps"):
        return src.timestamps
    # Fallback: reconstruct from recording
    rec = open_dat(slider_state["dat_path"], width=src.width, height=src.height)
    t_sorted = rec.timestamps[src.order]
    return t_sorted


def get_window(
    event_words: np.ndarray,
    time_order: np.ndarray,
    timestamps: np.ndarray,
    win_start: int,
    win_stop: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get events and timestamps for a time window."""
    event_indexes = time_order[win_start:win_stop]
    words = event_words[event_indexes].astype(np.uint32, copy=False)
    x_coords = (words & 0x3FFF).astype(np.int32, copy=False)
    y_coords = ((words >> 14) & 0x3FFF).astype(np.int32, copy=False)
    pixel_polarity = ((words >> 28) & 0xF) > 0
    event_timestamps = timestamps[event_indexes]

    return x_coords, y_coords, pixel_polarity, event_timestamps


def get_frame(
    window: tuple[np.ndarray, np.ndarray, np.ndarray],
    width: int = 1280,
    height: int = 720,
    *,
    base_color: tuple[int, int, int] = (127, 127, 127),  # gray
    on_color: tuple[int, int, int] = (255, 255, 255),  # white
    off_color: tuple[int, int, int] = (0, 0, 0),  # black
) -> np.ndarray:
    x_coords, y_coords, polarities_on = window
    frame = np.full((height, width, 3), base_color, np.uint8)
    frame[y_coords[polarities_on], x_coords[polarities_on]] = on_color
    frame[y_coords[~polarities_on], x_coords[~polarities_on]] = off_color

    return frame


def draw_hud(
    frame: np.ndarray,
    pacer: Pacer,
    batch_range: BatchRange,
    rpm_tracker: RPMTracker | None = None,
    *,
    color: tuple[int, int, int] = (0, 0, 0),  # black by default
) -> None:
    """Overlay timing info: wall time, recording time, playback speed, and RPM."""
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

    # RPM row (if tracker is active)
    if rpm_tracker is not None:
        rpm, confidence = rpm_tracker.get_rpm()
        if rpm > 0:
            rpm_str = f"RPM: {rpm:.1f}  (confidence: {confidence:.2f})"
            rpm_color = (
                (0, 255, 0) if confidence > 0.5 else (0, 165, 255)
            )  # Green if confident, orange if low confidence
            cv2.putText(
                frame,
                rpm_str,
                (8, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                rpm_color,
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                frame,
                "RPM: calculating...",
                (8, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (128, 128, 128),
                1,
                cv2.LINE_AA,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play .dat files with RPM tracking and dynamic movement following"
    )
    parser.add_argument("dat", help="Path to .dat file")
    parser.add_argument(
        "--window", type=float, default=10, help="Windows duration in ms"
    )
    parser.add_argument(
        "--speed", type=float, default=1, help="Playback speed (1 is real time)"
    )
    parser.add_argument(
        "--force-speed",
        action="store_true",
        help="Force the playback speed by dropping windows",
    )
    parser.add_argument(
        "--roi-x",
        type=int,
        default=None,
        help="X coordinate of ROI center (default: image center)",
    )
    parser.add_argument(
        "--roi-y",
        type=int,
        default=None,
        help="Y coordinate of ROI center (default: image center)",
    )
    parser.add_argument(
        "--roi-radius",
        type=int,
        default=200,
        help="Radius of circular ROI in pixels (default: 200)",
    )
    parser.add_argument(
        "--no-rpm",
        action="store_true",
        help="Disable RPM tracking",
    )
    parser.add_argument(
        "--static-roi",
        action="store_true",
        help="Use static ROI instead of dynamic movement tracking",
    )
    args = parser.parse_args()

    # Initialize slider state with command line arguments
    slider_state["window_ms"] = args.window
    slider_state["speed"] = args.speed
    slider_state["force_speed"] = args.force_speed
    slider_state["dat_path"] = args.dat

    # Create initial source and pacer
    src = DatFileSource(
        args.dat,
        width=slider_state["width"],
        height=slider_state["height"],
        window_length_us=slider_state["window_ms"] * 1000,
    )
    pacer = Pacer(speed=slider_state["speed"], force_speed=slider_state["force_speed"])

    # Get timestamps (workaround since we don't modify evio)
    timestamps = get_timestamps_from_source(src)

    # Initialize RPM tracker if enabled
    rpm_tracker = None
    if not args.no_rpm:
        roi_x = args.roi_x if args.roi_x is not None else slider_state["width"] // 2
        roi_y = args.roi_y if args.roi_y is not None else slider_state["height"] // 2
        rpm_tracker = RPMTracker(
            center_x=roi_x,
            center_y=roi_y,
            radius=args.roi_radius,
            width=slider_state["width"],
            height=slider_state["height"],
        )

    cv2.namedWindow("Evio Player", cv2.WINDOW_NORMAL)

    # Create trackbars
    # Window slider: 0.1ms to 500ms (in 0.1ms increments, so 1-5000)
    cv2.createTrackbar(
        "Window (ms)",
        "Evio Player",
        int(slider_state["window_ms"] * 10),
        5000,
        on_window_change,
    )

    # Speed slider: 0.01x to 5.0x (in 0.01x increments, so 1-500)
    cv2.createTrackbar(
        "Speed (x)",
        "Evio Player",
        int(slider_state["speed"] * 100),
        500,
        on_speed_change,
    )

    while True:
        # Check if window changed and recreate source if needed
        if slider_state["window_changed"]:
            slider_state["window_changed"] = False
            src = DatFileSource(
                slider_state["dat_path"],
                width=slider_state["width"],
                height=slider_state["height"],
                window_length_us=slider_state["window_ms"] * 1000,
            )
            timestamps = get_timestamps_from_source(src)
            # Reset pacer when restarting
            pacer = Pacer(
                speed=slider_state["speed"], force_speed=slider_state["force_speed"]
            )
            # Reset RPM tracker if active
            if rpm_tracker is not None:
                rpm_tracker.reset()

        # Update speed and force_speed dynamically
        pacer.speed = max(1e-9, slider_state["speed"])
        pacer.force_speed = slider_state["force_speed"]

        for batch_range in pacer.pace(src.ranges()):
            # Check for window change during playback
            if slider_state["window_changed"]:
                break

            # Update speed dynamically (in case it changed)
            pacer.speed = max(1e-9, slider_state["speed"])
            pacer.force_speed = slider_state["force_speed"]

            window = get_window(
                src.event_words,
                src.order,
                timestamps,
                batch_range.start,
                batch_range.stop,
            )
            x_coords, y_coords, pixel_polarity, event_timestamps = window
            
            frame = get_frame(
                (x_coords, y_coords, pixel_polarity),
                width=slider_state["width"],
                height=slider_state["height"],
            )
            
            # Update RPM tracker if active
            if rpm_tracker is not None:
                # Dynamic tracking: update ROI center based on movement
                if not args.static_roi and len(x_coords) > 10:
                    movement_center = detect_movement_center(
                        x_coords,
                        y_coords,
                        slider_state["width"],
                        slider_state["height"],
                    )
                    if movement_center is not None:
                        rpm_tracker.update_center(movement_center[0], movement_center[1])
                
                rpm_tracker.update(
                    x_coords,
                    y_coords,
                    event_timestamps,
                    batch_range.start_ts_us,
                    batch_range.end_ts_us,
                )
                # Draw ROI circle
                cv2.circle(
                    frame,
                    (rpm_tracker.center_x, rpm_tracker.center_y),
                    rpm_tracker.radius,
                    (0, 255, 255),
                    2,
                )
                # Draw center point
                cv2.circle(
                    frame,
                    (rpm_tracker.center_x, rpm_tracker.center_y),
                    3,
                    (0, 255, 255),
                    -1,
                )
            
            draw_hud(frame, pacer, batch_range, rpm_tracker)

            # Display current slider values and controls on frame
            controls_text = (
                f"Window: {slider_state['window_ms']:.1f}ms  "
                f"Speed: {slider_state['speed']:.2f}x  "
                f"Force: {'ON' if slider_state['force_speed'] else 'OFF'} (press 'f')"
            )
            cv2.putText(
                frame,
                controls_text,
                (8, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("Evio Player", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                cv2.destroyAllWindows()
                return
            elif key == ord("f") or key == ord("F"):
                # Toggle force speed
                slider_state["force_speed"] = not slider_state["force_speed"]
                pacer.force_speed = slider_state["force_speed"]

        # End of file reached, restart from beginning with current settings
        pacer = Pacer(
            speed=slider_state["speed"], force_speed=slider_state["force_speed"]
        )

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

