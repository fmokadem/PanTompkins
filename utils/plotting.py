import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Optional

def plot_detection_results(
    ecg_data_path: str,
    plot_path: Optional[str],
    signal_frequency: Optional[float],
    ecg_data_raw: Optional[np.ndarray],
    filtered_ecg_measurements: Optional[np.ndarray],
    differentiated_ecg_measurements: Optional[np.ndarray],
    squared_ecg_measurements: Optional[np.ndarray],
    integrated_ecg_measurements: Optional[np.ndarray],
    detected_peaks_indices: Optional[np.ndarray],
    qrs_peaks_indices: np.ndarray,
    ecg_data_detected: Optional[np.ndarray],
    show_plot: bool = False
) -> None:
    """Generates and saves/shows a plot showing different stages of the detection process."""

    if plot_path is None:
        print("Warning: Cannot plot data. Plot path not specified.")
        return

    required_data = [
        ecg_data_raw,
        filtered_ecg_measurements,
        differentiated_ecg_measurements,
        squared_ecg_measurements,
        integrated_ecg_measurements,
        ecg_data_detected,
        detected_peaks_indices # Need initial peaks too
    ]
    if any(data is None for data in required_data):
        print("Warning: Cannot plot data. Intermediate processing results missing.")
        return
    # Type assertion after None checks for mypy
    assert ecg_data_raw is not None
    assert filtered_ecg_measurements is not None
    assert differentiated_ecg_measurements is not None
    assert squared_ecg_measurements is not None
    assert integrated_ecg_measurements is not None
    assert detected_peaks_indices is not None
    assert ecg_data_detected is not None

    if ecg_data_raw.shape[1] < 2:
        print("Warning: Cannot plot data. Raw ECG data has unexpected shape.")
        return

    print(f"Plotting detection results to: {plot_path}")

    plt.close('all')
    fig, axes = plt.subplots(6, 1, sharex=True, figsize=(15, 18))
    fig.suptitle(f'Pan-Tompkins QRS Detection Steps\nFile: {os.path.basename(ecg_data_path)}', fontsize=16)

    time_axis = ecg_data_raw[:, 0]

    # Nested helper functions
    def _plot_signal(ax: plt.Axes, time: np.ndarray, data: np.ndarray, title: str, color: str = "salmon") -> None:
        if len(time) != len(data):
            min_len = min(len(time), len(data))
            time = time[:min_len]
            data = data[:min_len]
        if len(time) == 0:
            return
        ax.plot(time, data, color=color, linewidth=0.8)
        ax.set_title(title, fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_ylabel("Amplitude")
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3,3))

    def _plot_peaks(ax: plt.Axes, time: np.ndarray, data: np.ndarray, indices: np.ndarray, color: str = 'black', marker: str = 'o', size: int = 50, label: str = '') -> None:
        indices_in_bounds = indices[(indices >= 0) & (indices < len(time)) & (indices < len(data))]
        if indices_in_bounds.size > 0:
            ax.scatter(time[indices_in_bounds], data[indices_in_bounds], c=color, s=size, marker=marker, zorder=5, label=label)

    # Plotting stages
    _plot_signal(axes[0], time_axis, ecg_data_raw[:, 1], '1. Raw ECG Signal')
    _plot_signal(axes[1], time_axis[:len(filtered_ecg_measurements)], filtered_ecg_measurements, '2. Bandpass Filtered ECG')
    _plot_signal(axes[2], time_axis[:len(differentiated_ecg_measurements)], differentiated_ecg_measurements, '3. Differentiated Signal')
    _plot_signal(axes[3], time_axis[:len(squared_ecg_measurements)], squared_ecg_measurements, '4. Squared Signal')
    _plot_signal(axes[4], time_axis[:len(integrated_ecg_measurements)], integrated_ecg_measurements, '5. Integrated Signal')
    _plot_peaks(axes[4], time_axis[:len(integrated_ecg_measurements)], integrated_ecg_measurements, detected_peaks_indices, color='blue', marker='+', size=60, label='Initial Peaks')

    _plot_signal(axes[5], time_axis, ecg_data_detected[:, 1], '6. Raw ECG with Detected QRS Peaks')
    _plot_peaks(axes[5], time_axis, ecg_data_detected[:, 1], qrs_peaks_indices, color='red', marker='x', size=70, label='Detected QRS')

    axes[4].legend(loc='upper right')
    axes[5].legend(loc='upper right')

    axes[-1].set_xlabel(f"Time (seconds) - Sample Rate: {signal_frequency} Hz")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    try:
        fig.savefig(plot_path)
    except Exception as e:
        print(f"Error saving plot to {plot_path}: {e}")

    if show_plot:
        plt.show()

    plt.close(fig) 