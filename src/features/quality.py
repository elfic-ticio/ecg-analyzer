"""ECG signal quality assessment."""

import numpy as np
import neurokit2 as nk


def signal_quality_index(signal: np.ndarray, fs: int) -> float:
    """Estimate the Signal Quality Index (SQI) of an ECG segment.

    Internally calls ``nk.ecg_quality`` after cleaning the signal with
    ``nk.ecg_clean``. Returns the mean quality across all samples.

    Args:
        signal: 1-D ECG signal (raw or pre-filtered).
        fs: Sampling frequency in Hz.

    Returns:
        Mean SQI in ``[0.0, 1.0]``; higher values indicate cleaner signals.
        Returns ``0.0`` on any internal error (e.g. signal too short).
    """
    try:
        cleaned = nk.ecg_clean(signal, sampling_rate=fs, method="neurokit")
        quality = nk.ecg_quality(cleaned, sampling_rate=fs)
        # ecg_quality may return a Series or ndarray
        quality_arr = np.asarray(quality, dtype=float)
        return float(np.nanmean(quality_arr))
    except Exception:
        return 0.0


def is_analyzable(
    signal: np.ndarray,
    fs: int,
    min_sqi: float = 0.5,
) -> bool:
    """Decide whether signal quality is sufficient for HRV analysis.

    Args:
        signal: 1-D ECG signal.
        fs: Sampling frequency in Hz.
        min_sqi: Minimum acceptable SQI. Signals below this threshold are
                 considered too noisy for reliable peak detection.

    Returns:
        ``True`` if ``signal_quality_index(signal, fs) >= min_sqi``.
    """
    return signal_quality_index(signal, fs) >= min_sqi
