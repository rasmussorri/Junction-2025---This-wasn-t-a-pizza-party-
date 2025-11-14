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

