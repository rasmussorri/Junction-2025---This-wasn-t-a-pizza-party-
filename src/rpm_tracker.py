"""RPM tracking for rotating objects using event camera data."""

import numpy as np
from collections import deque


class RPMTracker:
    """
    Tracks RPM of rotating objects (e.g., fan blades) using event camera data.
    
    Uses event density analysis in a circular ROI to detect blade passes
    and estimates rotation period using peak detection and FFT analysis.
    """

    def __init__(
        self,
        center_x: int,
        center_y: int,
        radius: int,
        width: int = 1280,
        height: int = 720,
        min_events_per_window: int = 10,
        history_size: int = 100,
    ):
        """
        Initialize RPM tracker.

        Args:
            center_x: X coordinate of ROI center (pixels)
            center_y: Y coordinate of ROI center (pixels)
            radius: Radius of circular ROI (pixels)
            width: Image width (default: 1280)
            height: Image height (default: 720)
            min_events_per_window: Minimum events needed to consider a window
            history_size: Number of windows to keep in history for analysis
        """
        self.center_x = int(center_x)
        self.center_y = int(center_y)
        self.radius = int(radius)
        self.width = width
        self.height = height
        self.min_events_per_window = min_events_per_window
        self.history_size = history_size

        # History buffers
        self.event_counts = deque(maxlen=history_size)
        self.timestamps = deque(maxlen=history_size)
        self.window_durations = deque(maxlen=history_size)

        # RPM estimation
        self.current_rpm = 0.0
        self.confidence = 0.0
        self.last_update_time = None

        # Dynamic tracking
        self.smoothing_factor = 0.3  # How much to smooth center updates (0-1, lower = more smoothing)

    def _is_in_roi(self, x_coords: np.ndarray, y_coords: np.ndarray) -> np.ndarray:
        """Check which events are within the circular ROI."""
        dx = x_coords.astype(np.float32) - self.center_x
        dy = y_coords.astype(np.float32) - self.center_y
        distances_sq = dx * dx + dy * dy
        return distances_sq <= (self.radius * self.radius)

    def update(
        self,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        timestamps: np.ndarray,
        window_start_ts_us: int,
        window_end_ts_us: int,
    ) -> None:
        """
        Update tracker with events from a time window.

        Args:
            x_coords: X coordinates of events
            y_coords: Y coordinates of events
            timestamps: Timestamps of events (microseconds)
            window_start_ts_us: Start timestamp of window (microseconds)
            window_end_ts_us: End timestamp of window (microseconds)
        """
        if len(x_coords) == 0:
            self.event_counts.append(0)
            self.timestamps.append(window_end_ts_us)
            self.window_durations.append(window_end_ts_us - window_start_ts_us)
            return

        # Filter events in ROI
        in_roi = self._is_in_roi(x_coords, y_coords)
        event_count = np.sum(in_roi)

        # Store history
        self.event_counts.append(event_count)
        self.timestamps.append(window_end_ts_us)
        self.window_durations.append(window_end_ts_us - window_start_ts_us)

        # Update RPM estimate if we have enough data
        if len(self.event_counts) >= 10:
            self._estimate_rpm()

    def _estimate_rpm(self) -> None:
        """Estimate RPM from event density history."""
        if len(self.event_counts) < 10:
            return

        counts = np.array(self.event_counts)
        timestamps_us = np.array(self.timestamps)
        durations_us = np.array(self.window_durations)

        # Method 1: Peak detection for periodic patterns
        rpm_peak = self._estimate_rpm_from_peaks(counts, timestamps_us)

        # Method 2: FFT-based frequency detection
        rpm_fft = self._estimate_rpm_from_fft(counts, timestamps_us, durations_us)

        # Combine estimates (prefer peak detection if confident, else use FFT)
        if rpm_peak > 0 and self.confidence > 0.5:
            self.current_rpm = rpm_peak
        elif rpm_fft > 0:
            self.current_rpm = rpm_fft
            self.confidence = 0.3  # Lower confidence for FFT-only
        else:
            self.current_rpm = 0.0
            self.confidence = 0.0

    def _estimate_rpm_from_peaks(
        self, counts: np.ndarray, timestamps_us: np.ndarray
    ) -> float:
        """Estimate RPM by detecting peaks in event density."""
        if len(counts) < 5:
            return 0.0

        # Smooth the signal to reduce noise
        if len(counts) >= 5:
            # Simple moving average
            kernel_size = min(5, len(counts) // 2)
            if kernel_size > 1:
                smoothed = np.convolve(
                    counts, np.ones(kernel_size) / kernel_size, mode="same"
                )
            else:
                smoothed = counts
        else:
            smoothed = counts

        # Find peaks (local maxima)
        # A peak is where value is higher than neighbors
        peaks = []
        for i in range(1, len(smoothed) - 1):
            if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]:
                # Only consider significant peaks (above mean)
                if smoothed[i] > np.mean(smoothed) * 1.2:
                    peaks.append(i)

        if len(peaks) < 2:
            return 0.0

        # Calculate time differences between consecutive peaks
        peak_times = timestamps_us[peaks]
        if len(peak_times) < 2:
            return 0.0

        # Get time differences between peaks
        peak_intervals = np.diff(peak_times) / 1e6  # Convert to seconds

        # Filter out unrealistic intervals (too short or too long)
        # Assume RPM between 100 and 10000
        min_period = 60.0 / 10000.0  # seconds
        max_period = 60.0 / 100.0  # seconds
        valid_intervals = peak_intervals[
            (peak_intervals >= min_period) & (peak_intervals <= max_period)
        ]

        if len(valid_intervals) == 0:
            return 0.0

        # Use median to be robust to outliers
        median_period = np.median(valid_intervals)

        if median_period > 0:
            rpm = 60.0 / median_period
            # Update confidence based on consistency
            std_period = np.std(valid_intervals)
            self.confidence = max(0.0, 1.0 - (std_period / median_period))
            return rpm

        return 0.0

    def _estimate_rpm_from_fft(
        self,
        counts: np.ndarray,
        timestamps_us: np.ndarray,
        durations_us: np.ndarray,
    ) -> float:
        """Estimate RPM using FFT to find dominant frequency."""
        if len(counts) < 20:
            return 0.0

        # Normalize counts
        if np.max(counts) > 0:
            normalized = (counts - np.mean(counts)) / (np.std(counts) + 1e-9)
        else:
            return 0.0

        # Apply window function to reduce spectral leakage
        window = np.hanning(len(normalized))
        windowed = normalized * window

        # Compute FFT
        fft = np.fft.rfft(windowed)
        freqs = np.fft.rfftfreq(len(windowed))

        # Get average time step
        if len(timestamps_us) > 1:
            avg_duration = np.mean(durations_us) / 1e6  # Convert to seconds
        else:
            return 0.0

        # Find dominant frequency (excluding DC component)
        power = np.abs(fft[1:]) ** 2
        freqs_non_dc = freqs[1:]

        if len(power) == 0:
            return 0.0

        # Find peak frequency
        peak_idx = np.argmax(power)
        peak_freq_normalized = freqs_non_dc[peak_idx]

        # Convert normalized frequency to actual frequency (Hz)
        if avg_duration > 0:
            actual_freq_hz = peak_freq_normalized / avg_duration
        else:
            return 0.0

        # Convert to RPM (assuming one blade pass per cycle)
        # For a fan with N blades, actual RPM = freq_hz * 60 / N
        # We'll assume 1 blade pass = 1 cycle for now
        rpm = actual_freq_hz * 60.0

        # Validate RPM range
        if 100 <= rpm <= 10000:
            return rpm

        return 0.0

    def get_rpm(self) -> tuple[float, float]:
        """
        Get current RPM estimate and confidence.

        Returns:
            Tuple of (RPM, confidence) where confidence is 0.0 to 1.0
        """
        return self.current_rpm, self.confidence

    def update_center(self, new_x: int, new_y: int) -> None:
        """
        Update ROI center with smoothing to prevent jitter.
        
        Args:
            new_x: New X coordinate
            new_y: New Y coordinate
        """
        # Smooth the center update to prevent jumping
        self.center_x = int(
            self.center_x * (1 - self.smoothing_factor) + new_x * self.smoothing_factor
        )
        self.center_y = int(
            self.center_y * (1 - self.smoothing_factor) + new_y * self.smoothing_factor
        )
        
        # Clamp to image bounds
        self.center_x = max(self.radius, min(self.width - self.radius, self.center_x))
        self.center_y = max(self.radius, min(self.height - self.radius, self.center_y))

    def reset(self) -> None:
        """Reset tracker history."""
        self.event_counts.clear()
        self.timestamps.clear()
        self.window_durations.clear()
        self.current_rpm = 0.0
        self.confidence = 0.0

