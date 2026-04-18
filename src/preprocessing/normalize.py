"""Normalization and resampling utilities for ECG signals."""

from math import gcd

import numpy as np
from scipy.signal import resample_poly as _resample_poly


def zscore_normalize(signal: np.ndarray) -> np.ndarray:
    """Normalize a signal to zero mean and unit variance (z-score).

    Args:
        signal: 1-D ECG signal.

    Returns:
        Z-score normalized signal. Returns an all-zeros array if the
        standard deviation is zero (constant signal).
    """
    std = np.std(signal)
    if std == 0.0:
        return np.zeros_like(signal)
    return (signal - np.mean(signal)) / std


def minmax_normalize(
    signal: np.ndarray,
    feature_range: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """Scale a signal to a specified [min, max] range.

    Args:
        signal: 1-D ECG signal.
        feature_range: Target ``(min, max)`` interval. Defaults to ``(0.0, 1.0)``.

    Returns:
        Scaled signal. Returns an all-zeros array if the signal is constant.
    """
    lo, hi = feature_range
    s_min, s_max = np.min(signal), np.max(signal)
    if s_max == s_min:
        return np.zeros_like(signal)
    scaled = (signal - s_min) / (s_max - s_min)  # → [0, 1]
    return scaled * (hi - lo) + lo


def resample(
    signal: np.ndarray,
    fs_original: int,
    fs_target: int,
) -> np.ndarray:
    """Resample a signal from *fs_original* to *fs_target* Hz.

    Uses ``scipy.signal.resample_poly`` with up/down factors derived from
    the GCD of the two rates, minimising filter artifacts compared to
    naive FFT-based resampling.

    Args:
        signal: 1-D ECG signal.
        fs_original: Original sampling frequency in Hz.
        fs_target: Target sampling frequency in Hz.

    Returns:
        Resampled signal at *fs_target* Hz. Returns a copy when
        ``fs_original == fs_target``.
    """
    if fs_original == fs_target:
        return signal.copy()
    common = gcd(fs_original, fs_target)
    up = fs_target // common
    down = fs_original // common
    return _resample_poly(signal, up, down)
