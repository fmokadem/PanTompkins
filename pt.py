import numpy as np
import matplotlib.pyplot as plt
from time import gmtime, strftime
from scipy.signal import butter, lfilter
from typing import Optional, Tuple, List, Dict, Any
import os
import wave


# Default plot directory if not specified
DEFAULT_PLOT_DIR = "./"


class QRSDetectorOffline(object):
    """
    Python Offline ECG QRS Detector based on the Pan-Tomkins algorithm.

    This class implements an offline QRS complex detection algorithm for ECG signals,
    based on the Pan-Tompkins method. It processes a pre-recorded ECG signal (CSV or WAV),
    identifies QRS complexes, and provides results including peak indices and optional
    logging and plotting.

    Attributes:
        ecg_data_path (str): Path to the ECG dataset file (CSV or WAV).
        signal_frequency (int): The sampling frequency of the ECG signal (samples/sec).
                                If reading a WAV file, this is determined from the file,
                                potentially overriding the value provided in the constructor.
        filter_lowcut (float): Low cut-off frequency for the bandpass filter (Hz).
        filter_highcut (float): High cut-off frequency for the bandpass filter (Hz).
        filter_order (int): Order of the Butterworth bandpass filter.
        integration_window_ms (int): Width of the moving window integrator (milliseconds).
        findpeaks_spacing_ms (int): Minimum spacing between detected peaks (milliseconds).
        refractory_period_ms (int): Refractory period after a QRS detection (milliseconds).
        integration_window (int): Width of the moving window integrator (samples).
        findpeaks_spacing (int): Minimum spacing between detected peaks (samples).
        refractory_period (int): Refractory period after a QRS detection (samples).
        findpeaks_limit (float): Amplitude threshold factor for initial peak detection.
        qrs_peak_filtering_factor (float): Filtering factor for updating the QRS peak value.
        noise_peak_filtering_factor (float): Filtering factor for updating the noise peak value.
        qrs_noise_diff_weight (float): Weighting factor for threshold calculation.
        verbose (bool): If True, print detection results.
        log_data (bool): If True, log detection results to a CSV file.
        plot_data (bool): If True, generate and save a plot of the detection stages.
        show_plot (bool): If True and plot_data is True, display the generated plot.
        plot_dir (str): Directory to save generated plots.
        log_dir (str): Directory to save log files (currently uses plot_dir).
        ecg_data_raw (np.ndarray | None): Raw ECG data loaded (shape: [n_samples, 2] with time/index and value).
        filtered_ecg_measurements (np.ndarray | None): ECG data after bandpass filtering.
        differentiated_ecg_measurements (np.ndarray | None): Filtered data after differentiation.
        squared_ecg_measurements (np.ndarray | None): Differentiated data after squaring.
        integrated_ecg_measurements (np.ndarray | None): Squared data after moving window integration.
        detected_peaks_indices (np.ndarray | None): Indices of peaks detected in the integrated signal.
        detected_peaks_values (np.ndarray | None): Values of peaks detected in the integrated signal.
        qrs_peak_value (float): Running estimate of the QRS peak amplitude.
        noise_peak_value (float): Running estimate of the noise peak amplitude.
        threshold_value (float): Adaptive threshold for QRS detection.
        qrs_peaks_indices (np.ndarray): Indices identified as QRS peaks.
        noise_peaks_indices (np.ndarray): Indices identified as noise peaks.
        ecg_data_detected (np.ndarray | None): ECG data with an added column marking detected QRS locations.
        log_path (str | None): Full path for the log file.
        plot_path (str | None): Full path for the plot file.

    Based on: Pan J, Tompkins W.J., A real-time QRS detection algorithm,
    IEEE Transactions on Biomedical Engineering, Vol. BME-32, No. 3, March 1985, pp. 230-236.

    Note: This implementation is for educational/research purposes and is not a
    certified medical device.
    """

    # --- Configuration Defaults ---
    DEFAULT_SIGNAL_FREQ: Optional[int] = None # Detect from WAV if possible, else require input
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
                 verbose: bool = True,
                 log_data: bool = False,
                 plot_data: bool = False,
                 show_plot: bool = False,
                 plot_dir: str = DEFAULT_PLOT_DIR,
                 log_dir: Optional[str] = None):
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
            verbose: If True, print detection results.
            log_data: If True, log detection results to a CSV file.
            plot_data: If True, generate and save a plot of the detection stages.
            show_plot: If True and plot_data is True, display the generated plot.
            plot_dir: Directory to save generated plots.
            log_dir: Directory to save log files. If None, defaults to plot_dir.
        """
        self.ecg_data_path = ecg_data_path
        self._initial_signal_frequency = signal_frequency # Store user-provided freq
        self.signal_frequency = signal_frequency # This might be updated during data load

        self.verbose = verbose
        self.log_data = log_data
        self.plot_data = plot_data
        self.show_plot = show_plot
        self.plot_dir = plot_dir
        self.log_dir = log_dir if log_dir is not None else plot_dir

        # Create directories if they don't exist
        os.makedirs(self.plot_dir, exist_ok=True)
        if self.log_dir != self.plot_dir:
            os.makedirs(self.log_dir, exist_ok=True)

        # --- Algorithm Parameters ---
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

        # --- Internal State Variables ---
        self._initialize_state()

        # --- Run Processing ---
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
        self.log_path = None
        self.plot_path = None


    def run(self) -> None:
        """Executes the full QRS detection pipeline."""
        self._initialize_state() # Reset state if run multiple times
        self.load_ecg_data()

        if self.signal_frequency is None:
             print("Error: Signal frequency not determined (required for processing). Aborting.")
             return

        if self.ecg_data_raw is None or self.ecg_data_raw.size == 0:
            print(f"Error: No data loaded from {self.ecg_data_path}. Aborting.")
            return

        # Now that frequency is known, calculate sample-based parameters
        try:
            self._update_sample_parameters()
        except ValueError:
             return # Error already printed in helper method

        self.detect_peaks()
        if self.detected_peaks_indices is None:
             print(f"Error: Peak detection failed. Aborting.")
             return

        self.detect_qrs()

        if self.verbose:
            self.print_detection_data()

        if self.log_data:
            # Ensure log directory exists (might be redundant if created in __init__, but safe)
            os.makedirs(self.log_dir, exist_ok=True)
            self.log_path = os.path.join(
                self.log_dir,
                f"QRS_offline_detector_log_{strftime('%Y_%m_%d_%H_%M_%S', gmtime())}.csv"
            )
            self.log_detection_data()

        if self.plot_data:
             # Ensure plot directory exists
            os.makedirs(self.plot_dir, exist_ok=True)
            self.plot_path = os.path.join(
                self.plot_dir,
                f"QRS_offline_detector_plot_{strftime('%Y_%m_%d_%H_%M_%S', gmtime())}.png"
            )
            self.plot_detection_data(show_plot=self.show_plot)


    # --- Data Loading ---

    def load_ecg_data(self) -> None:
        """Loads ECG data from the specified file (CSV or WAV)."""
        file_extension = os.path.splitext(self.ecg_data_path)[1].lower()

        try:
            if file_extension == '.csv':
                self._load_csv()
            elif file_extension == '.wav':
                self._load_wav()
            else:
                raise ValueError(f"Unsupported file type: {file_extension}. Please use .csv or .wav")

        except FileNotFoundError:
            print(f"Error: ECG data file not found at {self.ecg_data_path}")
            self.ecg_data_raw = None
            self.signal_frequency = self._initial_signal_frequency # Revert if load fails
        except Exception as e:
            print(f"Error loading or processing ECG data from {self.ecg_data_path}: {e}")
            self.ecg_data_raw = None
            self.signal_frequency = self._initial_signal_frequency # Revert if load fails

    def _load_csv(self) -> None:
        """Loads data from a CSV file."""
        print(f"Loading CSV data from: {self.ecg_data_path}")
        if self.signal_frequency is None:
             raise ValueError("Signal frequency must be provided when loading CSV data.")

        # Assuming CSV has a header row and columns: timestamp, ecg_measurement
        loaded_data = np.loadtxt(self.ecg_data_path, skiprows=1, delimiter=',')
        if loaded_data.ndim == 1:
             # Handle case with only one column (assume it's values, generate time)
             print("Warning: CSV has only one column. Assuming it's ECG values, generating time indices.")
             values = loaded_data
             time_index = np.arange(len(values)) # Or calculate time based on freq?
             self.ecg_data_raw = np.vstack((time_index, values)).T
        elif loaded_data.ndim == 2 and loaded_data.shape[1] >= 2:
             # Assume first column is time/index, second is value
             self.ecg_data_raw = loaded_data[:, :2]
        else:
            raise ValueError("CSV data format error. Expected >= 2 columns or 1 value column.")
        print(f"Successfully loaded {self.ecg_data_raw.shape[0]} samples from CSV.")

    def _load_wav(self) -> None:
        """Loads data from a WAV file."""
        print(f"Loading WAV data from: {self.ecg_data_path}")
        rate, sampwidth, wav_array = self._read_wav_static(self.ecg_data_path)

        if self._initial_signal_frequency is not None and self._initial_signal_frequency != rate:
            print(f"Warning: Provided signal frequency ({self._initial_signal_frequency} Hz) "
                  f"differs from WAV file frequency ({rate} Hz). Using frequency from WAV file.")
        self.signal_frequency = rate # Always use frequency from WAV file

        # Handle multi-channel WAV - using only the first channel for now
        if wav_array.ndim == 2:
            if wav_array.shape[1] > 1:
                print(f"Warning: WAV file has {wav_array.shape[1]} channels. Using only the first channel.")
            ecg_values = wav_array[:, 0]
        elif wav_array.ndim == 1:
            ecg_values = wav_array
        else:
            raise ValueError("Unexpected WAV data array shape.")

        # Create time axis based on sample rate
        time_axis = np.arange(len(ecg_values)) / self.signal_frequency

        # Store raw data as [time, value]
        self.ecg_data_raw = np.vstack((time_axis, ecg_values)).T
        print(f"Successfully loaded {self.ecg_data_raw.shape[0]} samples from WAV (Frequency: {self.signal_frequency} Hz).")


    # --- ECG Signal Processing Steps ---

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

        self.detected_peaks_indices = self.findpeaks(
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
             print("Warning: No initial peaks detected in integrated signal.")


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

                # ----- Potential T-wave discrimination ----- >
                # Check slope? Check width? Check relative amplitude? Not implemented here.
                # <-------------------------------------------

                # Pan-Tompkins also has a second threshold (Threshold II) which is lower,
                # used for detecting missed beats after a search-back. Not implemented here.

            else:
                # Noise peak detected (or potentially a low-amplitude QRS)
                self.noise_peaks_indices = np.append(self.noise_peaks_indices, peak_index)
                # Update noise peak estimate (NPKI)
                self.noise_peak_value = self.noise_peak_filtering_factor * peak_value + \
                                        (1 - self.noise_peak_filtering_factor) * self.noise_peak_value

            # Update threshold (Threshold I) based on new QRS/noise peak estimates
            self.threshold_value = self.noise_peak_value + \
                                   self.qrs_noise_diff_weight * (self.qrs_peak_value - self.noise_peak_value)
            # Ensure threshold doesn't fall below a certain fraction of noise / become negative
            self.threshold_value = max(self.threshold_value, self.noise_peak_value * 0.5) # Heuristic lower bound


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


    # --- Results Reporting ---

    def print_detection_data(self) -> None:
        """Prints the detected QRS and noise peak indices to the console."""
        if self.signal_frequency is None:
             print("Cannot print RR intervals: Signal frequency not set.")
             rr_intervals = []
        elif len(self.qrs_peaks_indices) > 1:
            # Calculate RR based on the detected peak indices (latency included)
            rr_intervals_samples = np.diff(self.qrs_peaks_indices)
            rr_intervals_ms = rr_intervals_samples / self.signal_frequency * 1000
        else:
            rr_intervals_ms = []

        print("\n--- QRS Detection Results ---")
        print(f"Input File: {self.ecg_data_path}")
        print(f"Signal Frequency (Hz): {self.signal_frequency}")
        print(f"Detected QRS peak indices ({len(self.qrs_peaks_indices)}): [Refer to integrated signal indexing]")
        print(self.qrs_peaks_indices)
        print(f"Detected Noise peak indices ({len(self.noise_peaks_indices)}): [Refer to integrated signal indexing]")
        print(self.noise_peaks_indices)

        if len(rr_intervals_ms) > 0:
            print(f"RR Intervals (ms): N={len(rr_intervals_ms)}, Min={np.min(rr_intervals_ms):.1f}, Max={np.max(rr_intervals_ms):.1f}, Mean={np.mean(rr_intervals_ms):.1f}, Std={np.std(rr_intervals_ms):.1f}")
            # Calculate Heart Rate Variability (HRV) - simple SDNN
            hr = 60000.0 / rr_intervals_ms # Instantaneous HR in bpm
            print(f"Heart Rate (bpm): Min={np.min(hr):.1f}, Max={np.max(hr):.1f}, Mean={np.mean(hr):.1f}")


    def log_detection_data(self) -> None:
        """Logs the ECG data with QRS detection flags to a CSV file."""
        if self.ecg_data_detected is None or self.log_path is None:
            print("Warning: Cannot log data. Detection results or log path not available.")
            return

        print(f"Logging detection results to: {self.log_path}")
        try:
            with open(self.log_path, "w", newline='') as f: # Use text mode, handle EOL with newline=''
                # Header depends on whether input was CSV (timestamp) or WAV (generated time)
                # Using a generic header for now
                header = "time,ecg_measurement,qrs_detected\n"
                f.write(header)
                # Use appropriate format specifiers
                np.savetxt(f, self.ecg_data_detected, delimiter=",", fmt=['%.6f','%.6f','%d'])
        except Exception as e:
            print(f"Error logging data to {self.log_path}: {e}")


    def plot_detection_data(self, show_plot: bool = False) -> None:
        """Generates and saves a plot showing different stages of the detection process."""
        if self.plot_path is None:
             print("Warning: Cannot plot data. Plot path not specified.")
             return
        # Check if all necessary data arrays are available
        required_data = [
            self.ecg_data_raw,
            self.filtered_ecg_measurements,
            self.differentiated_ecg_measurements,
            self.squared_ecg_measurements,
            self.integrated_ecg_measurements,
            self.ecg_data_detected
        ]
        if any(data is None for data in required_data):
             print("Warning: Cannot plot data. Intermediate processing results missing.")
             # Optionally print which data is missing
             return
        if self.ecg_data_raw.shape[1] < 2:
            print("Warning: Cannot plot data. Raw ECG data has unexpected shape.")
            return


        print(f"Plotting detection results to: {self.plot_path}")

        plt.close('all') # Close previous plots
        fig, axes = plt.subplots(6, 1, sharex=True, figsize=(15, 18)) # 6 rows, 1 column
        fig.suptitle(f'Pan-Tompkins QRS Detection Steps\nFile: {os.path.basename(self.ecg_data_path)}', fontsize=16)

        # Use the time axis from the raw data
        time_axis = self.ecg_data_raw[:, 0]

        # Plotting helper function
        def _plot_signal(ax: plt.Axes, time: np.ndarray, data: np.ndarray, title: str, color: str = "salmon") -> None:
            if len(time) != len(data):
                 print(f"Warning plotting '{title}': Time axis length ({len(time)}) mismatch with data length ({len(data)}). Adjusting time axis.")
                 min_len = min(len(time), len(data))
                 time = time[:min_len]
                 data = data[:min_len]
            if len(time) == 0:
                 print(f"Warning plotting '{title}': No data to plot.")
                 return
            ax.plot(time, data, color=color, linewidth=0.8)
            ax.set_title(title, fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_ylabel("Amplitude")
            ax.ticklabel_format(axis='y', style='sci', scilimits=(-3,3))

        def _plot_peaks(ax: plt.Axes, time: np.ndarray, data: np.ndarray, indices: np.ndarray, color: str = 'black', marker: str = 'o', size: int = 50, label: str = '') -> None:
             # Ensure indices are within the bounds of the *plotted* time and data
             indices_in_bounds = indices[(indices >= 0) & (indices < len(time)) & (indices < len(data))]
             if indices_in_bounds.size > 0:
                 ax.scatter(time[indices_in_bounds], data[indices_in_bounds], c=color, s=size, marker=marker, zorder=5, label=label)
             elif indices.size > 0:
                  print(f"Warning plotting peaks '{label}': Indices ({indices.min()}-{indices.max()}) are out of bounds for plotted data length ({len(time)}).")


        # Plot each stage
        _plot_signal(axes[0], time_axis, self.ecg_data_raw[:, 1], '1. Raw ECG Signal')

        # The following signals might have slightly different lengths due to processing (e.g., diff)
        # We plot them against the time axis corresponding to their length
        _plot_signal(axes[1], time_axis[:len(self.filtered_ecg_measurements)], self.filtered_ecg_measurements, '2. Bandpass Filtered ECG')
        _plot_signal(axes[2], time_axis[:len(self.differentiated_ecg_measurements)], self.differentiated_ecg_measurements, '3. Differentiated Signal')
        _plot_signal(axes[3], time_axis[:len(self.squared_ecg_measurements)], self.squared_ecg_measurements, '4. Squared Signal')
        _plot_signal(axes[4], time_axis[:len(self.integrated_ecg_measurements)], self.integrated_ecg_measurements, '5. Integrated Signal')

        # Plot initial peaks found on the integrated signal
        # These indices refer to the integrated signal array
        _plot_peaks(axes[4], time_axis[:len(self.integrated_ecg_measurements)], self.integrated_ecg_measurements, self.detected_peaks_indices, color='blue', marker='+', size=60, label='Initial Peaks')

        # Plot final QRS peaks on the *raw* signal
        # The indices self.qrs_peaks_indices refer to the integrated signal.
        # We use the approximation that these indices map directly to the raw signal time axis (includes latency).
        _plot_signal(axes[5], time_axis, self.ecg_data_detected[:, 1], '6. Raw ECG with Detected QRS Peaks')
        _plot_peaks(axes[5], time_axis, self.ecg_data_detected[:, 1], self.qrs_peaks_indices, color='red', marker='x', size=70, label='Detected QRS')

        # Add legends
        axes[4].legend(loc='upper right')
        axes[5].legend(loc='upper right')

        axes[-1].set_xlabel(f"Time (seconds) - Sample Rate: {self.signal_frequency} Hz")
        plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to prevent title overlap

        try:
            fig.savefig(self.plot_path)
        except Exception as e:
            print(f"Error saving plot to {self.plot_path}: {e}")


        if show_plot:
            plt.show()

        plt.close(fig) # Close the figure explicitly after saving/showing


    # --- Static Helper Methods ---

    @staticmethod
    def _wav2array_static(nchannels: int, sampwidth: int, data: bytes) -> np.ndarray:
        """Converts raw WAV byte data to a NumPy array."""
        num_samples, remainder = divmod(len(data), sampwidth * nchannels)
        if remainder > 0:
            raise ValueError('The length of data is not a multiple of sampwidth * num_channels.')
        if sampwidth > 4:
            raise ValueError("sampwidth must not be greater than 4.")

        if sampwidth == 3:
            # Special handling for 24-bit (3-byte) samples
            a = np.empty((num_samples, nchannels, 4), dtype=np.uint8)
            raw_bytes = np.frombuffer(data, dtype=np.uint8)
            a[:, :, :sampwidth] = raw_bytes.reshape(-1, nchannels, sampwidth)
            a[:, :, sampwidth:] = (a[:, :, sampwidth - 1:sampwidth] >> 7) * 255 # Sign extension
            result = a.view('<i4').reshape(a.shape[:-1]) # View as 4-byte signed integer
        else:
            # Determine dtype based on sample width (1=uint8, 2/4=int)
            dt_char = 'u' if sampwidth == 1 else 'i'
            dtype_str = f'<{dt_char}{sampwidth}' # Little-endian
            a = np.frombuffer(data, dtype=dtype_str)
            result = a.reshape(-1, nchannels)

        # Ensure the result is float for consistency in processing
        return result.astype(np.float64)

    @staticmethod
    def _read_wav_static(filepath: str) -> Tuple[int, int, np.ndarray]:
        """
        Reads a WAV file.

        Args:
            filepath: Path to the WAV file.

        Returns:
            Tuple containing: (frame rate, sample width in bytes, data as NumPy array).

        Raises:
            ValueError: If the WAV file is compressed or unsupported format.
            FileNotFoundError: If the file does not exist.
        """
        try:
            with wave.open(filepath, 'rb') as wav:
                rate = wav.getframerate()
                nchannels = wav.getnchannels()
                sampwidth = wav.getsampwidth()
                nframes = wav.getnframes()
                comp_type = wav.getcomptype()
                comp_name = wav.getcompname()

                if comp_type != 'NONE':
                    raise ValueError(f"Cannot read compressed WAV file (Compression: {comp_type} - {comp_name})")

                data_bytes = wav.readframes(nframes)

            array = QRSDetectorOffline._wav2array_static(nchannels, sampwidth, data_bytes)
            return rate, sampwidth, array
        except wave.Error as e:
             raise ValueError(f"Error reading WAV file headers or data: {e}")


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

        # Input validation for frequencies
        if low < 0:
            print("Warning: lowcut frequency is negative, setting to 0 for lowpass filter.")
            low = 0
        if high >= 1.0:
            print(f"Warning: highcut frequency ({highcut} Hz) is >= Nyquist frequency ({nyquist_freq} Hz). Clamping highcut.")
            high = 0.999 # Clamp slightly below Nyquist
        if low >= high:
            raise ValueError(f"Lowcut frequency ({lowcut} Hz) must be lower than highcut frequency ({highcut} Hz).")

        # Choose filter type based on cutoff frequencies
        if low <= 0 and high < 1.0: # Low-pass filter case
            # print("Using lowpass filter.")
            b, a = butter(filter_order, high, btype='lowpass', output='ba')
        elif low > 0 and high >= 1.0: # High-pass filter case
            # print("Using highpass filter.")
            b, a = butter(filter_order, low, btype='highpass', output='ba')
        elif low > 0 and high < 1.0: # Band-pass filter
            # print("Using bandpass filter.")
            b, a = butter(filter_order, [low, high], btype='bandpass', output='ba')
        else: # Should not happen with validation, but just in case
            raise ValueError("Invalid frequency range for filter.")

        # Apply the filter
        try:
            y = lfilter(b, a, data)
            # Check for NaNs or Infs in output (can happen with unstable filters)
            if not np.all(np.isfinite(y)):
                print("Warning: Filter output contains non-finite values. Check filter parameters and input data.")
                # Optionally replace NaNs/Infs or handle differently
                y = np.nan_to_num(y)
            return y
        except ValueError as e:
            print(f"Error applying lfilter: {e}. Check filter coefficients (b, a) and input data.")
            # Return original data or raise error?
            raise # Reraise the exception for now


    @staticmethod
    def findpeaks(data: np.ndarray, spacing: int = 1, limit: Optional[float] = None) -> np.ndarray:
        """
        Detects peaks in data that are separated by at least `spacing` indices.

        Args:
            data: Input data array (1D).
            spacing: Minimum number of indices between peaks. Must be >= 1.
            limit: Optional minimum height threshold for peaks.

        Returns:
            Array of indices corresponding to the detected peaks.

        Note: This is a simplified peak detection based on comparing neighbors.
              Consider using `scipy.signal.find_peaks` for more robustness and options.
        """
        if not isinstance(data, np.ndarray) or data.ndim != 1:
            raise ValueError("`data` must be a 1D NumPy array.")
        if spacing < 1:
            raise ValueError("`spacing` must be at least 1.")

        # Pad the data at the ends to handle boundary peaks correctly
        len_data = data.size
        x = np.zeros(len_data + 2 * spacing)
        # Use the first/last value for padding, subtracting a small epsilon
        # to ensure boundary points aren't incorrectly detected as peaks.
        pad_value_start = data[0] - 1.e-6 if len_data > 0 else 0
        pad_value_end = data[-1] - 1.e-6 if len_data > 0 else 0
        x[:spacing] = pad_value_start
        x[spacing:spacing + len_data] = data
        x[spacing + len_data:] = pad_value_end

        # Find indices where the value is greater than all neighbors within `spacing` distance
        peak_candidate = np.ones(len_data, dtype=bool)
        for s in range(1, spacing + 1):
            # Check left neighbors
            peak_candidate &= (data > x[spacing - s : spacing - s + len_data])
            # Check right neighbors
            peak_candidate &= (data > x[spacing + s : spacing + s + len_data])

        # Get indices of peak candidates
        ind = np.argwhere(peak_candidate).flatten()

        # Filter by height limit if provided
        if limit is not None:
            if len(ind) > 0:
                ind = ind[data[ind] >= limit] # Use >= for inclusive limit
            else:
                 ind = np.array([], dtype=int)

        return ind


if __name__ == '__main__':

    print("Running QRS Detector Offline - Example Usage")

    # --- Select Input File --- #
    # Option 1: Use a CSV file (requires providing signal frequency)
    # data_path_csv = 'ecg_data/ecg_data_1.csv' # Replace with your CSV file path
    # freq_csv = 250 # Must provide frequency for CSV
    # input_file = data_path_csv
    # input_freq = freq_csv

    # Option 2: Use a WAV file (frequency is read from the file)
    data_path_wav = 'sig100.wav' # Replace with your WAV file path (ensure it exists)
    input_file = data_path_wav
    input_freq = None # Set to None to let the detector read from WAV header
                      # Or provide a value (e.g., 360) - it will be checked against the header

    if not os.path.exists(input_file):
        print(f"Error: Input ECG file not found at '{input_file}'")
        print("Please ensure the file exists or modify the path in the script.")
        # Example of creating dummy data (optional)
        # if input_file.endswith('.csv'):
        #     print("Creating dummy CSV data...")
        #     fs = input_freq if input_freq else 250
        #     duration = 10
        #     samples = fs * duration
        #     time_dummy = np.arange(samples) / fs
        #     ecg_dummy = 0.5 * np.sin(2 * np.pi * 1 * time_dummy) + 0.2 * np.sin(2 * np.pi * 10 * time_dummy) + np.random.normal(0, 0.1, samples)
        #     dummy_data = np.vstack((time_dummy, ecg_dummy)).T
        #     os.makedirs(os.path.dirname(input_file), exist_ok=True)
        #     np.savetxt(input_file, dummy_data, delimiter=',', header='timestamp,value', comments='')
        #     print(f"Created dummy CSV file at '{input_file}'")
        # else:
        #     print("Cannot create dummy WAV data easily. Please provide a real WAV file.")
    else:
        # --- Instantiate and Run Detector --- #
        output_dir = "qrs_output" # Sub-directory for logs and plots

        print(f"\nInitializing QRSDetectorOffline with file: {input_file}")
        qrs_detector = QRSDetectorOffline(
            ecg_data_path=input_file,
            signal_frequency=input_freq, # Pass None for WAV to auto-detect, or required freq for CSV
            verbose=True,
            log_data=True,
            plot_data=True,
            show_plot=False, # Set to True to display the plot immediately
            plot_dir=output_dir,
            log_dir=output_dir
        )

        # Example of accessing results after run() finishes in __init__
        # if qrs_detector.qrs_peaks_indices is not None:
        #    print(f"\nNumber of QRS peaks detected: {len(qrs_detector.qrs_peaks_indices)}")
        # if qrs_detector.ecg_data_detected is not None:
        #    print(f"Shape of data with detections: {qrs_detector.ecg_data_detected.shape}")

        print("\nDetection process finished.")
