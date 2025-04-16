import numpy as np
import matplotlib.pyplot as plt
from time import gmtime, strftime
from scipy.signal import butter, lfilter
from typing import Optional, Tuple, List, Dict, Any
import os
import argparse 
from .utils.io_utils import load_ecg_data
from .utils.output_utils import create_qrs_detection_output
from ._findpeaks import findpeaks
from .utils.plotting import plot_detection_results


DEFAULT_PLOT_DIR = "./qrs_output/" 


class QRSDetectorOffline(object):
    """
    This class implements an offline QRS complex detection algorithm for ECG signals,


    """

    # --- Configuration Defaults ---
    DEFAULT_SIGNAL_FREQ: Optional[int] = None
    DEFAULT_FILTER_LOWCUT: float = 0.0
    DEFAULT_FILTER_HIGHCUT: float = 15.0
    DEFAULT_FILTER_ORDER: int = 1
    # Default window sizes in ms (will be converted to samples based on actual signal_frequency)
    DEFAULT_INTEGRATION_WINDOW_MS: int = 60
    DEFAULT_FINDPEAKS_SPACING_MS: int = 200
    DEFAULT_REFRACTORY_PERIOD_MS: int = 200

    DEFAULT_FINDPEAKS_LIMIT: float = 0.35
    DEFAULT_QRS_PEAK_FILTERING_FACTOR: float = 0.125
    DEFAULT_NOISE_PEAK_FILTERING_FACTOR: float = 0.125
    DEFAULT_QRS_NOISE_DIFF_WEIGHT: float = 0.25

    def __init__(self,
                 ecg_data_path: str,
                 signal_frequency: Optional[int] = DEFAULT_SIGNAL_FREQ,
                 filter_lowcut: float = DEFAULT_FILTER_LOWCUT,
                 filter_highcut: float = DEFAULT_FILTER_HIGHCUT,
                 filter_order: int = DEFAULT_FILTER_ORDER,
                 integration_window_ms: int = DEFAULT_INTEGRATION_WINDOW_MS,
                 findpeaks_spacing_ms: int = DEFAULT_FINDPEAKS_SPACING_MS,
                 findpeaks_limit: float = DEFAULT_FINDPEAKS_LIMIT,
                 refractory_period_ms: int = DEFAULT_REFRACTORY_PERIOD_MS,
                 qrs_peak_filtering_factor: float = DEFAULT_QRS_PEAK_FILTERING_FACTOR,
                 noise_peak_filtering_factor: float = DEFAULT_NOISE_PEAK_FILTERING_FACTOR,
                 qrs_noise_diff_weight: float = DEFAULT_QRS_NOISE_DIFF_WEIGHT,
                 plot_data: bool = False,
                 show_plot: bool = False,
                 plot_dir: str = DEFAULT_PLOT_DIR):
        """Initializes the QRSDetectorOffline instance.

        Args:
            ecg_data_path: Path to the ECG dataset file (CSV or WAV).
            signal_frequency: ECG signal frequency (samples/sec). If None (default),
                              it must be detectable from the input file (e.g., WAV header).
                              If provided for a WAV file, it will be overridden by the
                              frequency from the WAV header (with a warning).
            filter_lowcut: Low cut-off frequency for the bandpass filter (Hz).
            filter_highcut: High cut-off frequency for the bandpass filter (Hz).
            filter_order: Order of the Butterworth bandpass filter.
            integration_window_ms: Width of the moving window integrator in milliseconds.
            findpeaks_spacing_ms: Minimum spacing between detected peaks in milliseconds.
            findpeaks_limit: Amplitude threshold factor for initial peak detection.
            refractory_period_ms: Refractory period after a QRS detection in milliseconds.
            qrs_peak_filtering_factor: Filtering factor for updating QRS peak value.
            noise_peak_filtering_factor: Filtering factor for updating noise peak value.
            qrs_noise_diff_weight: Weighting factor for threshold calculation.
            plot_data: If True, generate and save a plot of the detection stages.
            show_plot: If True and plot_data is True, display the generated plot.
            plot_dir: Directory to save generated plots.
        """
        self.ecg_data_path = ecg_data_path
        self._initial_signal_frequency = signal_frequency 
        self.signal_frequency = signal_frequency # This might be updated during data load

        self.plot_data = plot_data
        self.show_plot = show_plot
        self.plot_dir = plot_dir

        os.makedirs(self.plot_dir, exist_ok=True)

        self.filter_lowcut = filter_lowcut
        self.filter_highcut = filter_highcut
        self.filter_order = filter_order

        # Store parameters in ms, conversion to samples happens after frequency is known
        self.integration_window_ms = integration_window_ms
        self.findpeaks_spacing_ms = findpeaks_spacing_ms
        self.refractory_period_ms = refractory_period_ms

        # These will be calculated later in _update_sample_parameters
        self.integration_window: int = 0
        self.findpeaks_spacing: int = 0
        self.refractory_period: int = 0

        self.findpeaks_limit = findpeaks_limit
        self.qrs_peak_filtering_factor = qrs_peak_filtering_factor
        self.noise_peak_filtering_factor = noise_peak_filtering_factor
        self.qrs_noise_diff_weight = qrs_noise_diff_weight

        self._initialize_state()

        self.run()


    def _ms_to_samples(self, ms_duration: int) -> int:
        """Converts a duration in milliseconds to samples based on current signal frequency."""
        if self.signal_frequency is None or self.signal_frequency <= 0:
            raise ValueError("Signal frequency must be set and positive to convert ms to samples.")
        return int((ms_duration / 1000.0) * self.signal_frequency)

    def _update_sample_parameters(self) -> None:
        """Calculates sample-based parameters from millisecond parameters."""
        try:
            self.integration_window = self._ms_to_samples(self.integration_window_ms)
            self.findpeaks_spacing = self._ms_to_samples(self.findpeaks_spacing_ms)
            self.refractory_period = self._ms_to_samples(self.refractory_period_ms)
        except ValueError as e:
            print(f"Error calculating sample parameters: {e}")
            # Handle error appropriately, maybe raise it further or set flags
            raise

    def _initialize_state(self) -> None:
        """Initializes the internal state variables for processing."""
        self.ecg_data_raw = None
        self.filtered_ecg_measurements = None
        self.differentiated_ecg_measurements = None
        self.squared_ecg_measurements = None
        self.integrated_ecg_measurements = None
        self.detected_peaks_indices = None
        self.detected_peaks_values = None

        self.qrs_peak_value = 0.0
        self.noise_peak_value = 0.0
        self.threshold_value = 0.0

        self.qrs_peaks_indices = np.array([], dtype=int)
        self.noise_peaks_indices = np.array([], dtype=int)

        self.ecg_data_detected = None
        self.plot_path = None


    def run(self) -> None:
        """Executes the full QRS detection pipeline."""
        self._initialize_state() # Reset state if run multiple times

        self.ecg_data_raw, self.signal_frequency = load_ecg_data(
            self.ecg_data_path, self._initial_signal_frequency
        )

        if self.signal_frequency is None:
             print("Error: Signal frequency not determined. Aborting.")
             return

        if self.ecg_data_raw is None or self.ecg_data_raw.size == 0:
            # Error already printed by load_ecg_data
            return

        # Now that frequency is known, calculate sample-based parameters
        try:
            self._update_sample_parameters()
        except ValueError:
             return # Error already printed in helper method

        self.detect_peaks()
        if self.detected_peaks_indices is None:
             return

        self.detect_qrs()

        if self.plot_data:
            os.makedirs(self.plot_dir, exist_ok=True)
            self.plot_path = os.path.join(
                self.plot_dir,
                f"QRS_offline_detector_plot_{strftime('%Y_%m_%d_%H_%M_%S', gmtime())}.png"
            )
            plot_detection_results(
                ecg_data_path=self.ecg_data_path,
                plot_path=self.plot_path,
                signal_frequency=self.signal_frequency,
                ecg_data_raw=self.ecg_data_raw,
                filtered_ecg_measurements=self.filtered_ecg_measurements,
                differentiated_ecg_measurements=self.differentiated_ecg_measurements,
                squared_ecg_measurements=self.squared_ecg_measurements,
                integrated_ecg_measurements=self.integrated_ecg_measurements,
                detected_peaks_indices=self.detected_peaks_indices,
                qrs_peaks_indices=self.qrs_peaks_indices,
                ecg_data_detected=self.ecg_data_detected,
                show_plot=self.show_plot
            )

    def detect_peaks(self) -> None:
        """
        Processes the loaded ECG data to find potential peak locations.

        This involves:
        1. Bandpass filtering.
        2. Differentiation.
        3. Squaring.
        4. Moving window integration.
        5. Initial peak detection (fiducial marks).
        """
        if self.ecg_data_raw is None:
            print("Error: Cannot detect peaks, ECG data not loaded.")
            return
        if self.signal_frequency is None or self.signal_frequency <= 0:
             print("Error: Cannot detect peaks, signal frequency not set.")
             return
        if self.integration_window <= 0 or self.findpeaks_spacing <=0: # Check if parameters were calculated
            print("Error: Sample-based parameters not calculated correctly. Aborting peak detection.")
            return


        ecg_measurements = self.ecg_data_raw[:, 1]

        # 1. Bandpass Filter
        self.filtered_ecg_measurements = self.bandpass_filter(
            ecg_measurements,
            lowcut=self.filter_lowcut,
            highcut=self.filter_highcut,
            signal_freq=self.signal_frequency,
            filter_order=self.filter_order
        )
        # Simple edge effect handling (consider more sophisticated methods if needed)
        if len(self.filtered_ecg_measurements) > self.filter_order:
            pad_width = min(5, len(self.filtered_ecg_measurements) // 2)
            self.filtered_ecg_measurements[:pad_width] = self.filtered_ecg_measurements[pad_width]

        # 2. Derivative
        self.differentiated_ecg_measurements = np.ediff1d(self.filtered_ecg_measurements)
        # Pad to maintain length consistency with filtered signal
        self.differentiated_ecg_measurements = np.pad(self.differentiated_ecg_measurements, (1, 0), 'edge')

        # 3. Squaring
        self.squared_ecg_measurements = self.differentiated_ecg_measurements ** 2

        # 4. Moving Window Integration
        # Using 'same' mode ensures output length matches input length (squared signal)
        # Normalizing kernel by window size
        integration_kernel = np.ones(self.integration_window) / self.integration_window
        self.integrated_ecg_measurements = np.convolve(self.squared_ecg_measurements, integration_kernel, mode='same')

        # 5. Fiducial Mark (Initial Peak Detection)
        # Limit is relative to the max of the integrated signal
        signal_max = np.max(self.integrated_ecg_measurements)
        # Handle case where signal max is zero or negative
        dynamic_limit = self.findpeaks_limit * signal_max if signal_max > 0 else None

        # Use imported findpeaks function
        self.detected_peaks_indices = findpeaks(
            data=self.integrated_ecg_measurements,
            limit=dynamic_limit,
            spacing=self.findpeaks_spacing
        )

        if self.detected_peaks_indices.size > 0:
            # Ensure indices are within bounds before accessing values
            valid_peak_indices = self.detected_peaks_indices[self.detected_peaks_indices < len(self.integrated_ecg_measurements)]
            self.detected_peaks_values = self.integrated_ecg_measurements[valid_peak_indices]
            # Update indices to only include valid ones
            self.detected_peaks_indices = valid_peak_indices
        else:
             self.detected_peaks_values = np.array([])
             # Minimize printing - handled downstream if no QRS found
             # print("Warning: No initial peaks detected in integrated signal.")


    # --- QRS Detection Logic ---

    def detect_qrs(self) -> None:
        """
        Classifies detected peaks as QRS complexes or noise based on adaptive thresholds.
        """
        if self.detected_peaks_indices is None or self.detected_peaks_values is None:
             print("Error: Cannot detect QRS, initial peak detection failed or produced no results.")
             return
        if len(self.detected_peaks_indices) != len(self.detected_peaks_values):
             print(f"Error: Mismatch between detected peak indices ({len(self.detected_peaks_indices)}) and values ({len(self.detected_peaks_values)}). Aborting QRS detection.")
             # This might indicate an issue in detect_peaks logic
             return
        if self.refractory_period <= 0:
            print("Error: Refractory period must be positive. Aborting QRS detection.")
            return


        # Reset peak lists
        self.qrs_peaks_indices = np.array([], dtype=int)
        self.noise_peaks_indices = np.array([], dtype=int)

        # Initialize thresholds
        # Consider initializing based on the first few seconds of data for robustness
        spki = 0.0 # Running estimate of signal peak (QRS peak)
        npki = 0.0 # Running estimate of noise peak
        # Initial threshold: Often set based on initial signal/noise estimates
        # Using the midpoint initially, will adapt quickly.
        initial_buffer_len = min(len(self.detected_peaks_values), max(10, self.findpeaks_spacing)) # Use a few peaks or spacing
        if initial_buffer_len > 0:
            initial_peaks = self.detected_peaks_values[:initial_buffer_len]
            # Heuristic: guess signal peak is high percentile, noise is lower
            spki = np.percentile(initial_peaks, 75) if len(initial_peaks) > 1 else initial_peaks[0]
            # Heuristic: guess noise peak is lower percentile or fraction of initial peak
            npki = np.percentile(initial_peaks, 25) if len(initial_peaks) > 1 else initial_peaks[0] * 0.5
        thri = npki + self.qrs_noise_diff_weight * (spki - npki) # Initial Threshold I
        # Ensure threshold isn't negative
        thri = max(thri, 0)
        # Store initial values used for thresholding
        self.qrs_peak_value = spki
        self.noise_peak_value = npki
        self.threshold_value = thri


        for i, peak_index in enumerate(self.detected_peaks_indices):
            peak_value = self.detected_peaks_values[i]

            # Check refractory period relative to the last *detected QRS peak*
            # Important: Use the index in the *original signal space* if possible,
            # although the peak_index here refers to the integrated signal. Assuming refractory
            # period in samples applies similarly to the integrated signal peaks for now.
            if self.qrs_peaks_indices.size > 0:
                last_qrs_index = self.qrs_peaks_indices[-1]
                if peak_index - last_qrs_index < self.refractory_period:
                    # Skip peak if it falls within the refractory period of the last QRS
                    continue

            # Classification based on threshold (Threshold I)
            if peak_value > self.threshold_value:
                # QRS peak detected
                self.qrs_peaks_indices = np.append(self.qrs_peaks_indices, peak_index)
                # Update signal peak estimate (SPKI)
                self.qrs_peak_value = self.qrs_peak_filtering_factor * peak_value + \
                                      (1 - self.qrs_peak_filtering_factor) * self.qrs_peak_value
            else:
                # Noise peak detected (or potentially a low-amplitude QRS)
                self.noise_peaks_indices = np.append(self.noise_peaks_indices, peak_index)
                # Update noise peak estimate (NPKI)
                self.noise_peak_value = self.noise_peak_filtering_factor * peak_value + \
                                        (1 - self.noise_peak_filtering_factor) * self.noise_peak_value

            # Update threshold (Threshold I) based on new QRS/noise peak estimates
            self.threshold_value = self.noise_peak_value + \
                                   self.qrs_noise_diff_weight * (self.qrs_peak_value - self.noise_peak_value)
            # Heuristic lower bound for threshold
            self.threshold_value = max(self.threshold_value, self.noise_peak_value * 0.5)


        # --- Search back for missed QRS peaks (Optional but part of Pan-Tompkins) ---
        # If the RR-interval becomes too large, lower the threshold (Threshold II) and search back.
        # This adds complexity and is omitted for now.

        # --- Final Output Array --- #
        # Create output array marking QRS locations on the *original* time scale
        if self.ecg_data_raw is not None:
            measurement_qrs_detection_flag = np.zeros((self.ecg_data_raw.shape[0], 1))

            # Map integrated signal indices back to raw signal indices.
            # This is non-trivial due to filter delays and integration window.
            # Simple approach (used in original): Use the indices directly. This introduces latency.
            # More accurate: Estimate group delay of filter + integration window center/end.
            # For now, sticking to the simpler approach assuming indices roughly align, but latency exists.
            qrs_indices_in_raw = self.qrs_peaks_indices # This is an approximation!

            # Ensure indices are valid before marking
            valid_indices = qrs_indices_in_raw[(qrs_indices_in_raw >= 0) & (qrs_indices_in_raw < len(measurement_qrs_detection_flag))]
            measurement_qrs_detection_flag[valid_indices] = 1
            self.ecg_data_detected = np.append(self.ecg_data_raw, measurement_qrs_detection_flag, axis=1)
        else:
            self.ecg_data_detected = None


    @staticmethod
    def bandpass_filter(data: np.ndarray,
                        lowcut: float,
                        highcut: float,
                        signal_freq: float,
                        filter_order: int) -> np.ndarray:
        """
        Applies a Butterworth bandpass filter to the data.

        Args:
            data: Input signal array.
            lowcut: Low cut-off frequency (Hz). Must be > 0 for bandpass.
            highcut: High cut-off frequency (Hz). Must be < nyquist freq.
            signal_freq: Sampling frequency of the signal (Hz).
            filter_order: Order of the filter.

        Returns:
            Filtered signal array.
        """
        nyquist_freq = 0.5 * signal_freq
        low = lowcut / nyquist_freq
        high = highcut / nyquist_freq


        if low < 0:
            low = 0
        if high >= 1.0:
            # Clamp highcut frequency slightly below Nyquist if it's too high
            high = 0.999 # Clamp slightly below Nyquist
        if low >= high:
            raise ValueError(f"Lowcut frequency ({lowcut} Hz) must be lower than highcut frequency ({highcut} Hz).")

        # Choose filter type based on cutoff frequencies
        if low <= 0 and high < 1.0: # Low-pass filter case
            b, a = butter(filter_order, high, btype='lowpass', output='ba')
        elif low > 0 and high >= 1.0: # High-pass filter case
            b, a = butter(filter_order, low, btype='highpass', output='ba')
        elif low > 0 and high < 1.0: # Band-pass filter
            b, a = butter(filter_order, [low, high], btype='bandpass', output='ba')
        else: # Should not happen with validation, but just in case
            raise ValueError("Invalid frequency range for filter.")

        # Apply the filter
        try:
            y = lfilter(b, a, data)
            if not np.all(np.isfinite(y)):
                print("Warning: Filter output contains non-finite values.")
                y = np.nan_to_num(y)
            return y
        except ValueError as e:
            print(f"Error applying lfilter: {e}.")
            raise # Reraise the exception for now
