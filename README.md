# Pan-Tompkins QRS Detector (Offline)

This repository contains a Python implementation of the Pan-Tompkins algorithm for offline QRS complex detection in ECG signals.

## Structure

*   `run_detector.py`: The main script to run the detector from the command line.
*   `PanTompkins/pt.py`: Contains the `QRSDetectorOffline` class implementing the core algorithm logic (filtering, differentiation, squaring, integration, adaptive thresholding).
*   `PanTompkins/utils/io_utils.py`: Utility functions for loading ECG data from `.csv` and `.wav` files.
*   `PanTompkins/utils/output_utils.py`: Utility function for formatting the output data.
*   `PanTompkins/utils/plotting.py`: Utility function for generating plots of the detection steps.
*   `PanTompkins/_findpeaks.py`: A simplified peak detection function.
*   `qrs_output/`: Default directory where output plots are saved (created automatically).
*   `PanTompkins/sig100.wav`: Example WAV file for testing (located inside the package directory).

## Algorithm Steps

1.  **Load Data**: Reads ECG data from a specified `.csv` or `.wav` file.
2.  **Bandpass Filter**: Filters the signal to isolate frequencies relevant to the QRS complex (default 0-15 Hz).
3.  **Differentiate**: Calculates the signal's derivative to highlight sharp slopes.
4.  **Square**: Squares the differentiated signal to enhance high-frequency components and make values positive.
5.  **Moving Window Integration**: Averages the squared signal over a time window to create a feature signal where QRS complexes appear as peaks.
6.  **Peak Detection**: Identifies initial potential peaks in the integrated signal.
7.  **Adaptive Thresholding**: Classifies peaks as QRS or noise using adaptive thresholds based on running estimates of signal and noise peak amplitudes.
8.  **Output**: (Optional) Generates a plot showing the signal at various processing stages and marking the detected QRS complexes.

The script is run from the command line using `run_detector.py`.

```bash
python run_detector.py --input-file <path_to_ecg_file> [options]
```

**Required Arguments:**

*   `--input-file <path_to_ecg_file>`: Path to the input ECG file (must be `.csv` or `.wav`).

**Optional Arguments:**

*   `--frequency <Hz>`: Signal frequency in Hz. **Required if using a `.csv` file.** If using a `.wav` file, the frequency is read from the file header by default, but this argument can override it (a warning will be shown if they differ).
*   `--plot`: Generate and save a plot showing the detection steps in the `--output-dir`.
*   `--show-plot`: Display the generated plot interactively (requires a graphical backend).
*   `--output-dir <directory>`: Directory to save the output plot (default: `./qrs_output/`).
*   *(Advanced)*: You can add more command-line arguments to control algorithm parameters like filter cutoffs, window sizes, etc., by modifying the `argparse` section in `run_detector.py`.

**Example (using the provided WAV file and generating a plot):**

```bash
python run_detector.py --input-file PanTompkins/sig100.wav --plot --output-dir qrs_output
```

## Dependencies

*   NumPy
*   SciPy
*   Matplotlib (only if plotting)

Install dependencies using pip:
```bash
pip install numpy scipy matplotlib
```

## Reference

Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm. *IEEE Transactions on Biomedical Engineering*, BME-32(3), 230–236. 