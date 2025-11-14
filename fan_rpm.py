import argparse
import numpy as np
from evio.source.dat_file import DatFileSource


def build_time_series(src: DatFileSource, window_size_ms: float = 1.0):
    """
    Build a time series by counting events in fixed-time windows.
    
    Args:
        src: DatFileSource object
        window_size_ms: Window size in milliseconds
        
    Returns:
        timestamps: List of window timestamps (in ms)
        event_counts: List of event counts per window
    """
    event_counts = []
    timestamps = []
    
    window_size_us = window_size_ms * 1000
    
    # Extract timestamps from the event words
    # The event words are packed as: [x:14bits][y:14bits][polarity:4bits]
    # We need to get timestamps from the recording
    # Use the precomputed ranges from DatFileSource
    ranges = list(src.ranges())
    
    if len(ranges) == 0:
        return timestamps, event_counts
    
    # Get the start and end times from the ranges
    start_time = ranges[0].start_ts_us
    end_time = ranges[-1].end_ts_us
    
    current_window_start = start_time
    
    # Count events in each time window
    while current_window_start < end_time:
        current_window_end = current_window_start + window_size_us
        
        # Count events from all ranges that overlap with this window
        count = 0
        for batch_range in ranges:
            # Check if this range overlaps with our window
            if batch_range.end_ts_us >= current_window_start and batch_range.start_ts_us < current_window_end:
                # This is a rough count - we're counting all events in overlapping ranges
                # For better accuracy, we'd need to decode timestamps, but this is sufficient for FFT
                count += (batch_range.stop - batch_range.start)
        
        event_counts.append(count)
        timestamps.append((current_window_start - start_time) / 1000.0)  # Convert to ms
        
        current_window_start = current_window_end
    
    return timestamps, event_counts


def calculate_rpm(event_counts, sample_rate_hz, num_blades):
    """
    Calculate RPM using FFT.
    
    Args:
        event_counts: List of event counts per time window
        sample_rate_hz: Sampling rate in Hz (1/window_size)
        num_blades: Number of fan blades
        
    Returns:
        rpm: Calculated RPM
        dominant_freq: Dominant frequency in Hz
    """
    # Perform FFT
    fft_result = np.fft.fft(event_counts)
    fft_magnitude = np.abs(fft_result)
    fft_freq = np.fft.fftfreq(len(event_counts), 1.0 / sample_rate_hz)
    
    # Only consider positive frequencies, ignore DC component (0 Hz)
    positive_freq_mask = fft_freq > 0
    positive_freqs = fft_freq[positive_freq_mask]
    positive_magnitudes = fft_magnitude[positive_freq_mask]
    
    # Find the dominant frequency
    dominant_idx = np.argmax(positive_magnitudes)
    dominant_freq = positive_freqs[dominant_idx]
    
    # Calculate RPM
    # The detected frequency is (RPM / 60) * num_blades
    rpm = (dominant_freq * 60) / num_blades
    
    return rpm, dominant_freq


def analyze_const_rpm(dat_file_path, window_size_ms=1.0, num_blades=3):
    """
    Analyze constant RPM fan data.
    
    Args:
        dat_file_path: Path to the .dat file
        window_size_ms: Window size in milliseconds
        num_blades: Number of fan blades
    """
    print(f"Loading {dat_file_path}...")
    src = DatFileSource(dat_file_path, width=1280, height=720)
    
    print(f"Building time series (window size: {window_size_ms}ms)...")
    timestamps, event_counts = build_time_series(src, window_size_ms)
    
    print(f"Total windows: {len(event_counts)}")
    print(f"Duration: {timestamps[-1] if timestamps else 0:.2f}ms")
    
    # Calculate sample rate
    sample_rate_hz = 1000.0 / window_size_ms  # Convert ms to Hz
    
    print(f"\nCalculating RPM (assuming {num_blades} blades)...")
    rpm, dominant_freq = calculate_rpm(event_counts, sample_rate_hz, num_blades)
    
    print(f"\nResults:")
    print(f"  Dominant frequency: {dominant_freq:.2f} Hz")
    print(f"  Detected RPM: {rpm:.2f}")
    
    return timestamps, event_counts, rpm


def analyze_varying_rpm_stft(dat_file_path, window_size_ms=1.0, num_blades=3):
    """
    Analyze varying RPM fan data using Short-Time Fourier Transform (STFT).
    
    Args:
        dat_file_path: Path to the .dat file
        window_size_ms: Window size in milliseconds for event counting
        num_blades: Number of fan blades
    """
    try:
        from scipy import signal
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: scipy and matplotlib are required for STFT analysis")
        print("Install with: pip install scipy matplotlib")
        return
    
    print(f"Loading {dat_file_path}...")
    src = DatFileSource(dat_file_path, width=1280, height=720)
    
    print(f"Building time series (window size: {window_size_ms}ms)...")
    timestamps, event_counts = build_time_series(src, window_size_ms)
    
    print(f"Total windows: {len(event_counts)}")
    print(f"Duration: {timestamps[-1] if timestamps else 0:.2f}ms ({timestamps[-1]/1000:.2f}s)")
    
    # Calculate sample rate
    sample_rate_hz = 1000.0 / window_size_ms
    
    print(f"\nPerforming STFT analysis...")
    # Perform STFT
    f, t, Zxx = signal.stft(event_counts, fs=sample_rate_hz, nperseg=256)
    
    # Convert frequencies to RPM
    rpm_axis = (f * 60) / num_blades
    
    # Plot spectrogram
    plt.figure(figsize=(12, 8))
    plt.pcolormesh(t, rpm_axis, np.abs(Zxx), shading='gouraud', cmap='viridis')
    plt.ylabel('RPM')
    plt.xlabel('Time (s)')
    plt.title(f'Fan RPM Over Time (STFT Spectrogram, {num_blades} blades assumed)')
    plt.colorbar(label='Magnitude')
    plt.ylim([0, 2000])  # Focus on reasonable RPM range
    
    # Add reference lines for expected RPM range
    plt.axhline(y=1100, color='r', linestyle='--', alpha=0.5, label='Expected min RPM')
    plt.axhline(y=1300, color='r', linestyle='--', alpha=0.5, label='Expected max RPM')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('fan_rpm_spectrogram.png', dpi=150)
    print(f"Spectrogram saved to: fan_rpm_spectrogram.png")
    plt.show()
    
    # Extract dominant RPM over time
    dominant_rpm_over_time = []
    for time_idx in range(Zxx.shape[1]):
        spectrum = np.abs(Zxx[:, time_idx])
        # Focus on reasonable RPM range (500-2000 RPM)
        valid_rpm_mask = (rpm_axis >= 500) & (rpm_axis <= 2000)
        if np.any(valid_rpm_mask):
            valid_spectrum = spectrum[valid_rpm_mask]
            valid_rpms = rpm_axis[valid_rpm_mask]
            if len(valid_spectrum) > 0:
                dominant_idx = np.argmax(valid_spectrum)
                dominant_rpm_over_time.append(valid_rpms[dominant_idx])
    
    if dominant_rpm_over_time:
        print(f"\nRPM Statistics:")
        print(f"  Min RPM: {np.min(dominant_rpm_over_time):.2f}")
        print(f"  Max RPM: {np.max(dominant_rpm_over_time):.2f}")
        print(f"  Mean RPM: {np.mean(dominant_rpm_over_time):.2f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze fan RPM from event camera data")
    parser.add_argument(
        "dat", 
        nargs='?',
        default="C:\\Users\\Henri\\Downloads\\Junction\\Data\\fan-20251114T191847Z-1-001\\fan\\fan_const_rpm.dat",
        #"C:\\Users\\Henri\\Downloads\\Junction\\Data\\fan-20251114T191847Z-1-001\\fan\\fan_varying_rpm.dat",
        help="Path to .dat file"
    )
    parser.add_argument(
        "--window", type=float, default=1.0, 
        help="Window size in milliseconds for event counting (default: 1.0)"
    )
    parser.add_argument(
        "--blades", type=int, default=3,
        help="Number of fan blades (default: 3)"
    )
    parser.add_argument(
        "--stft", action="store_true",
        help="Use STFT analysis for varying RPM (creates spectrogram)"
    )
    args = parser.parse_args()
    
    if args.stft:
        analyze_varying_rpm_stft(args.dat, args.window, args.blades)
    else:
        analyze_const_rpm(args.dat, args.window, args.blades)


if __name__ == "__main__":
    main()
