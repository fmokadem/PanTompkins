# PanTompkins ECG QRS Detector

This repository contains a Python implementation of the Pan-Tompkins algorithm for detecting QRS complexes in ECG signals.
The code can process ECG data from both CSV and WAV files.

## Features

- Implements the core steps of the Pan-Tompkins algorithm:
    - Bandpass Filtering
    - Differentiation
    - Squaring
    - Moving Window Integration
    - Adaptive Thresholding for QRS detection
- Supports loading data from:
    - CSV files (requires specifying signal frequency, format: `timestamp,value`)
    - Uncompressed WAV files (signal frequency read from header)
- Generates optional outputs:
    - Log file (.csv) containing timestamps, raw ECG values, and QRS detection flags (0 or 1).
    - Plot (.png) visualizing the different stages of the algorithm and the final QRS detections.
- Configurable parameters for filtering, window sizes, and detection thresholds.

## Usage

The main script is `pt.py`.

1.  **Prepare your ECG data:**
    *   **CSV:** Ensure your file has at least two columns, typically `timestamp,ecg_value`. A header row is expected and skipped.
    *   **WAV:** Use standard, uncompressed WAV files.

2.  **Modify `pt.py` (if needed):**
    *   Locate the `if __name__ == '__main__':` block at the end of the script.
    *   Update the `input_file` variable to the path of your ECG data file (`.csv` or `.wav`).
    *   If using a CSV file, set the `input_freq` variable to the correct signal sampling frequency (in Hz).
    *   If using a WAV file, you can set `input_freq = None` to automatically detect the frequency from the WAV header.
    *   Adjust `output_dir`, `log_data`, `plot_data`, `show_plot` as desired.
    *   You can also adjust algorithm parameters (filter cutoffs, window sizes, etc.) when creating the `QRSDetectorOffline` instance if the defaults are not suitable.

3.  **Run the script:**
    ```bash
    python pt.py
    ```

4.  **Check the output:**
    *   Detection results (peak indices, RR intervals, heart rate) will be printed to the console.
    *   If `log_data=True`, a CSV log file will be created in the specified `output_dir`.
    *   If `plot_data=True`, a PNG plot file will be created in the specified `output_dir`.

## Example Plot

(The generated plot shows the signal at various processing stages and the final detected QRS peaks marked on the raw ECG signal)

```
[ Placeholder: Add an example qrs_output/*.png image here if available ]

Example: ![Example QRS Detection Plot](qrs_output/QRS_offline_detector_plot_YYYY_MM_DD_HH_MM_SS.png)
```

## Dependencies

- Python 3.x
- NumPy (`pip install numpy`)
- Matplotlib (`pip install matplotlib`)
- SciPy (`pip install scipy`)

## Notes

- This implementation is intended for educational and research purposes.
- The mapping of detected peak indices (from the integrated signal) back to the raw signal time includes some latency due to filtering and integration. More precise mapping would require calculating group delays.
- The implementation includes basic adaptive thresholding but omits the more complex search-back procedure with a second threshold, which is part of the full Pan-Tompkins algorithm for detecting potentially missed beats.
