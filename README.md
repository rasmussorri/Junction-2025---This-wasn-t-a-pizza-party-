# Junction 2025

**Team:** This wasn't a pizza party?

## About

Event camera data analysis project for Junction 2025. This project uses the `evio` library to process event camera data, with a focus on analyzing RPM patterns from fan and drone event recordings.

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

2. **Create and activate a virtual environment:**
   
   **Windows:**
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```
   
   **Mac/Linux:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   Note: The `evio` library is automatically installed from GitHub as part of the dependencies.

### Data Setup

The event camera data files are **not included** in this repository (they're too large for git).

1. Download the data separately or generate your own
2. Place data files in `../Data/` (parent directory of this repo) or adjust paths in scripts
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

**Play event camera data:**
```bash
python scripts/play_dat.py path/to/file.dat --window 10 --speed 1.0
```

**Analyze constant fan RPM:**
```bash
python fan_rpm.py ../Data/fan-*/fan/fan_const_rpm.dat --blades 3
```

**Analyze varying fan RPM with spectrogram:**
```bash
python fan_rpm.py ../Data/fan-*/fan/fan_varying_rpm.dat --blades 3 --stft
```

**Play and track RPM:**
```bash
python src/play_dat_rpm.py path/to/file.dat --blades 3
```

## Project Structure

```
.
├─ pyproject.toml          # Project configuration
├─ requirements.txt        # Python dependencies
├─ README.md              # This file
├─ LICENSE                # MIT License
├─ fan_rpm.py             # Fan RPM analysis script
├─ drone-draft.py         # Drone detection draft code
├─ scripts/
│  └─ play_dat.py         # Event camera data player
└─ src/
   ├─ play_dat_rpm.py     # RPM tracking player
   └─ rpm_tracker.py      # RPM tracking utilities
```

## Dependencies

- **evio**: Event camera data processing library (installed from GitHub)
- **numpy**: Numerical computing
- **opencv-python**: Computer vision and image processing
- **matplotlib**: Plotting and visualization
- **scipy**: Scientific computing (for signal processing)

## Team

This wasn't a pizza party?

## License

MIT License - see LICENSE file for details

