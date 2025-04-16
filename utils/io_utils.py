import numpy as np
import os
import wave
from typing import Tuple, Optional

def wav2array(nchannels: int, sampwidth: int, data: bytes) -> np.ndarray:
    """Converts raw WAV byte data to a NumPy array."""
    num_samples, remainder = divmod(len(data), sampwidth * nchannels)
    if remainder > 0:
        raise ValueError('The length of data is not a multiple of sampwidth * num_channels.')
    if sampwidth > 4:
        raise ValueError("sampwidth must not be greater than 4.")

    if sampwidth == 3:
        a = np.empty((num_samples, nchannels, 4), dtype=np.uint8)
        raw_bytes = np.frombuffer(data, dtype=np.uint8)
        a[:, :, :sampwidth] = raw_bytes.reshape(-1, nchannels, sampwidth)
        a[:, :, sampwidth:] = (a[:, :, sampwidth - 1:sampwidth] >> 7) * 255
        result = a.view('<i4').reshape(a.shape[:-1])
    else:
        dt_char = 'u' if sampwidth == 1 else 'i'
        dtype_str = f'<{dt_char}{sampwidth}'
        a = np.frombuffer(data, dtype=dtype_str)
        result = a.reshape(-1, nchannels)

    return result.astype(np.float64)

def read_wav(filepath: str) -> Tuple[int, int, np.ndarray]:
    """Reads a WAV file."""
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

        array = wav2array(nchannels, sampwidth, data_bytes)
        return rate, sampwidth, array
    except wave.Error as e:
         raise ValueError(f"Error reading WAV file headers or data: {e}")

def load_ecg_data(ecg_data_path: str, initial_signal_frequency: Optional[int]) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """Loads ECG data from CSV or WAV file."""
    file_extension = os.path.splitext(ecg_data_path)[1].lower()
    ecg_data_raw = None
    signal_frequency = initial_signal_frequency

    try:
        if file_extension == '.csv':
            if signal_frequency is None:
                raise ValueError("Signal frequency must be provided when loading CSV data.")
            loaded_data = np.loadtxt(ecg_data_path, skiprows=1, delimiter=',')
            if loaded_data.ndim == 1:
                values = loaded_data
                time_index = np.arange(len(values))
                ecg_data_raw = np.vstack((time_index, values)).T
            elif loaded_data.ndim == 2 and loaded_data.shape[1] >= 2:
                ecg_data_raw = loaded_data[:, :2]
            else:
                raise ValueError("CSV data format error. Expected >= 2 columns or 1 value column.")

        elif file_extension == '.wav':
            rate, _, wav_array = read_wav(ecg_data_path)
            if initial_signal_frequency is not None and initial_signal_frequency != rate:
                print(f"Warning: Provided signal frequency ({initial_signal_frequency} Hz) "
                      f"differs from WAV file frequency ({rate} Hz). Using frequency from WAV file.")
            signal_frequency = rate

            if wav_array.ndim == 2:
                if wav_array.shape[1] > 1:
                    print(f"Warning: WAV file has {wav_array.shape[1]} channels. Using only the first channel.")
                ecg_values = wav_array[:, 0]
            elif wav_array.ndim == 1:
                ecg_values = wav_array
            else:
                raise ValueError("Unexpected WAV data array shape.")

            time_axis = np.arange(len(ecg_values)) / signal_frequency
            ecg_data_raw = np.vstack((time_axis, ecg_values)).T
        else:
            raise ValueError(f"Unsupported file type: {file_extension}. Please use .csv or .wav")

    except FileNotFoundError:
        print(f"Error: ECG data file not found at {ecg_data_path}")
        return None, initial_signal_frequency
    except Exception as e:
        print(f"Error loading or processing ECG data from {ecg_data_path}: {e}")
        return None, initial_signal_frequency

    if ecg_data_raw is not None:
        print(f"Successfully loaded {ecg_data_raw.shape[0]} samples from {os.path.basename(ecg_data_path)}")

    return ecg_data_raw, signal_frequency 