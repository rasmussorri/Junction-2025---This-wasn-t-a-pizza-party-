import cv2
import numpy as np
from evio.source.dat_file import DatFileSource

# Load data
print("Loading data...")
src = DatFileSource(
    r"C:\Users\Henri\Downloads\Junction\Data\drone_moving-20251114T191633Z-1-002\drone_moving\drone_moving.dat",
    width=1280, 
    height=720,
    window_length_us=10000
)

print(f"Loaded! Number of batches: {len(src)}")

# Get first batch
ranges = list(src.ranges())
first_range = ranges[0]

print(f"First batch: start={first_range.start}, stop={first_range.stop}")

# Extract events
event_indexes = src.order[first_range.start:first_range.stop]
words = src.event_words[event_indexes].astype(np.uint32, copy=False)
x_coords = (words & 0x3FFF).astype(np.int32, copy=False)
y_coords = ((words >> 14) & 0x3FFF).astype(np.int32, copy=False)
polarities = ((words >> 28) & 0xF) > 0

print(f"Number of events: {len(x_coords)}")
print(f"X range: {x_coords.min()} to {x_coords.max()}")
print(f"Y range: {y_coords.min()} to {y_coords.max()}")

# Create frame
frame = np.zeros((720, 1280, 3), dtype=np.uint8)
frame[y_coords[polarities], x_coords[polarities]] = (255, 255, 255)  # White
frame[y_coords[~polarities], x_coords[~polarities]] = (100, 100, 100)  # Gray

print("Displaying frame... Press any key to close.")
cv2.imshow("Test Frame", frame)
cv2.waitKey(0)  # Wait indefinitely for key press
cv2.destroyAllWindows()
print("Done!")
