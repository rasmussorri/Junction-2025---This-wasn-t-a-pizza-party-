<<<<<<< HEAD
# evio

Minimal Python library for standardized handling of event camera data.

**evio** provides a single abstraction for event streams. Each source yields standardized event packets containing `x_coords, y_coords, timestamps, polarities` arrays. This makes algorithms and filters source-agnostic.

---

## Features
- Unified async interface for event streams
- Read `.dat` recordings with optional real-time pacing
- Extensible to live cameras via adapter classes (requires Metavision SDK)

---

## Repository Structure

```
.
├─ pyproject.toml
├─ README.md
├─ LICENSE
├─ .gitignore
├─ scripts/
│  └─ play_dat.py    
└─ src/
   └─ evio/
      ├─ __init__.py
      ├── core/
      │   ├── __init__.py
      │   ├── index_scheduler.py
      │   ├── mmap.py
      │   ├── pacer.py
      │   └── recording.py
      └─── source/
          ├── __init__.py
          └── dat_file.py
       
```

---

## Quick start using UV
If not already installed, install UV (instructions [here](https://docs.astral.sh/uv/getting-started/installation/)) \
Clone the repo and in the repo root run

```bash
# create venv and install dependencies.
uv sync

# play a .dat file in real time
uv run scripts/play_dat.py path/to/dat/file.dat
```

Adjust window duration in ms using `--window` argument and playback speed factor with `--speed` argument. When event data is constructed to frames we take all events between t and t + window and display them in the frame. With very short windows the rendering of the frames can take longer than the actual window duration and the player falls behind (depends on the playback speed), you can see this by comparing the wall clock to the recording clock in the GUI. In such cases you can force the playback speed with a `--force-speed` argument. This drops enough frames to make the recording play according to the set speed.

---


## `.dat` File Encoding

`evio` reads Prophesee Metavision-style DAT files, which store events as fixed-width binary records following a short ASCII header.

### Header
The file starts with text lines beginning with `%`, for example:

```
% Width 1280
% Height 720
% Format EVT3
```

After the header, two bytes appear:

- **event_type** — currently only stored in metadata (not interpreted by `evio`)
- **event_size** — must be `8`, meaning each event occupies 8 bytes

### Event Record Format (8 bytes)
The binary payload is interpreted as an array of structured records with dtype:

```python
_DTYPE_CD8 = np.dtype([("t32", "<u4"), ("w32", "<u4")])
```

Each event record is 8 bytes (64 bits):

- `t32` (upper 32 bits) is a little-endian `uint32` timestamp in microseconds.
- `w32` (lower 32 bits) packs polarity and coordinates as:

| Bits  | Meaning                                   |
|-------|-------------------------------------------|
| 31–28 | polarity (4 bits; > 0 → ON, 0 → OFF)      |
| 27–14 | y coordinate (14 bits)                    |
| 13–0  | x coordinate (14 bits)                    |

This matches the decoder:

```python
packed_w32 = raw_events["w32"].astype(np.uint32, copy=False)

decoded_x = (packed_w32 & 0x3FFF).astype(np.uint16, copy=False)
decoded_y = ((packed_w32 >> 14) & 0x3FFF).astype(np.uint16, copy=False)
raw_polarity = ((packed_w32 >> 28) & 0xF).astype(np.uint8, copy=False)
decoded_polarity = (raw_polarity > 0).astype(np.int8, copy=False)
```

### Decoded Arrays in `evio`
`evio` exposes the following decoded NumPy arrays:

- `x_coords` — uint16 (from bits 0–13)
- `y_coords` — uint16 (from bits 14–27)
- `timestamps` — int64 (from `t32` promoted from uint32)
- `polarities` — int8 (0 for OFF, 1 for ON)

### Memory-Mapped Reading
`evio` uses a `numpy.memmap` view of the event region with `_DTYPE_CD8` and performs zero-copy decoding of the packed fields. This allows:

- fast slicing of large recordings
- stable real-time playback
- minimal memory use even with millions of events




## License
MIT
=======
# Junction 2025

**Team:** This wasn't a pizza party?

## About

Project description coming soon...

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rasmussorri/Junction-2025---This-wasn-t-a-pizza-party-.git
   cd Junction-2025---This-wasn-t-a-pizza-party-
   ```

2. **Set up the evio library** (in the parent directory):
   ```bash
   cd ..
   git clone https://github.com/ahtihelminen/evio.git
   cd Junction-2025---This-wasn-t-a-pizza-party-
   ```

3. **Create and activate a virtual environment:**
   
   **Windows:**
   ```powershell
   python -m venv ../.venv
   ../.venv/Scripts/activate
   ```
   
   **Mac/Linux:**
   ```bash
   python -m venv ../.venv
   source ../.venv/bin/activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Data Setup

The event camera data files are **not included** in this repository (they're too large for git).

1. Download the data separately or generate your own
2. Place data files in `../Data/` (parent directory of this repo)
3. Expected structure:
   ```
   Data/
     fan-*/
       fan/
         fan_const_rpm.dat
         fan_varying_rpm.dat
     fred-0/
       Event/
         events.dat
   ```

### Usage

**Analyze constant fan RPM:**
```bash
python fan_rpm.py ../Data/fan-*/fan/fan_const_rpm.dat --blades 3
```

**Analyze varying fan RPM with spectrogram:**
```bash
python fan_rpm.py ../Data/fan-*/fan/fan_varying_rpm.dat --blades 3 --stft
```

## Team

This wasn't a pizza party?

## License

[Add your license here]
>>>>>>> d77ded5b3cdded87e4d766086ab49af0cafec847

