import argparse
import os
from PanTompkins.pt import QRSDetectorOffline, DEFAULT_PLOT_DIR

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Offline Pan-Tompkins QRS Detector')
    parser.add_argument('--input-file', type=str, required=True,
                        help='Path to the input ECG file (.csv or .wav)')
    parser.add_argument('--frequency', type=int, default=None,
                        help='Signal frequency (Hz). Required for CSV, read from header for WAV if not provided.')
    parser.add_argument('--plot', action='store_true', help='Generate and save plot of detection steps.')
    parser.add_argument('--show-plot', action='store_true', help='Show the generated plot interactively.')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_PLOT_DIR,
                        help=f'Directory to save plots (default: {DEFAULT_PLOT_DIR})')

    args = parser.parse_args()

    input_file_path = os.path.abspath(args.input_file)

    if not os.path.exists(input_file_path):
        print(f"Error: Input ECG file not found at '{input_file_path}'")
    else:
        print(f"\nInitializing QRSDetectorOffline with file: {input_file_path}")
        try:
            qrs_detector = QRSDetectorOffline(
                ecg_data_path=input_file_path, # Use absolute path
                signal_frequency=args.frequency,
                plot_data=args.plot or args.show_plot,
                show_plot=args.show_plot,
                plot_dir=args.output_dir
            )

            # Minimal output after run completes
            if qrs_detector.qrs_peaks_indices is not None:
               print(f"\nDetection finished. Found {len(qrs_detector.qrs_peaks_indices)} QRS peaks.")
               if args.plot or args.show_plot:
                   print(f"Plot saved to {os.path.abspath(qrs_detector.plot_path)} (relative to execution dir: {qrs_detector.plot_path})")
            else:
               print("\nDetection process finished, but no QRS peaks were identified.")

        except Exception as e:
            print(f"\nAn error occurred during QRS detection: {e}")
            # Optionally re-raise for more detail e.g., raise 