import numpy as np
from typing import Optional

def create_qrs_detection_output(ecg_data_raw: Optional[np.ndarray], qrs_peaks_indices: np.ndarray) -> Optional[np.ndarray]:
    """Creates the output array with raw ECG data and a QRS detection flag column."""
    if ecg_data_raw is None:
        return None

    measurement_qrs_detection_flag = np.zeros((ecg_data_raw.shape[0], 1))

    # Map integrated signal indices back to raw signal indices (approximation).
    qrs_indices_in_raw = qrs_peaks_indices

    valid_indices = qrs_indices_in_raw[(qrs_indices_in_raw >= 0) & (qrs_indices_in_raw < len(measurement_qrs_detection_flag))]
    if valid_indices.size > 0:
        measurement_qrs_detection_flag[valid_indices] = 1

    return np.append(ecg_data_raw, measurement_qrs_detection_flag, axis=1) 