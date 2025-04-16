import numpy as np
from typing import Optional

def findpeaks(data: np.ndarray, spacing: int = 1, limit: Optional[float] = None) -> np.ndarray:
    """
    Detects peaks in data that are separated by at least `spacing` indices.
    Simplified implementation.
    """
    if not isinstance(data, np.ndarray) or data.ndim != 1:
        raise ValueError("`data` must be a 1D NumPy array.")
    if spacing < 1:
        raise ValueError("`spacing` must be at least 1.")

    len_data = data.size
    x = np.zeros(len_data + 2 * spacing)
    pad_value_start = data[0] - 1.e-6 if len_data > 0 else 0
    pad_value_end = data[-1] - 1.e-6 if len_data > 0 else 0
    x[:spacing] = pad_value_start
    x[spacing:spacing + len_data] = data
    x[spacing + len_data:] = pad_value_end

    peak_candidate = np.ones(len_data, dtype=bool)
    for s in range(1, spacing + 1):
        peak_candidate &= (data > x[spacing - s : spacing - s + len_data])
        peak_candidate &= (data > x[spacing + s : spacing + s + len_data])

    ind = np.argwhere(peak_candidate).flatten()

    if limit is not None:
        if len(ind) > 0:
            ind = ind[data[ind] >= limit]
        else:
            ind = np.array([], dtype=int)

    return ind 