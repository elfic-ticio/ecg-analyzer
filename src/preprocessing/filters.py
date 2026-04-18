"""Bandpass, notch, and baseline-removal filters for ECG signals."""

import numpy as np
from scipy import signal as sp


def bandpass_filter(
    signal: np.ndarray,
    fs: int,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter.

    Uses ``scipy.signal.filtfilt`` to eliminate phase distortion, which is
    critical for preserving R-peak morphology.

    Args:
        signal: 1-D ECG signal.
        fs: Sampling frequency in Hz.
        lowcut: Lower cutoff frequency in Hz.
        highcut: Upper cutoff frequency in Hz.
        order: Filter order. Higher orders give steeper roll-off but may
               introduce numerical instability for very low cutoffs.

    Returns:
        Filtered signal with the same shape as *signal*.
    """
    nyq = fs / 2.0
    b, a = sp.butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return sp.filtfilt(b, a, signal)


def notch_filter(
    signal: np.ndarray,
    fs: int,
    freq: float = 50.0,
    quality_factor: float = 30.0,
) -> np.ndarray:
    """Apply a zero-phase IIR notch filter to suppress power-line interference.

    Args:
        signal: 1-D ECG signal.
        fs: Sampling frequency in Hz.
        freq: Frequency to notch out in Hz. Use 50.0 for European grids,
              60.0 for North American grids.
        quality_factor: Q factor (center_freq / bandwidth). Higher values
                        produce a narrower notch with less signal distortion.

    Returns:
        Filtered signal with the same shape as *signal*.
    """
    b, a = sp.iirnotch(freq, quality_factor, fs)
    return sp.filtfilt(b, a, signal)


def remove_baseline(
    signal: np.ndarray,
    fs: int,
    method: str = "highpass",
) -> np.ndarray:
    """Remove low-frequency baseline wander from an ECG signal.

    Args:
        signal: 1-D ECG signal.
        fs: Sampling frequency in Hz.
        method: Strategy to use:

            - ``"highpass"`` – 4th-order Butterworth high-pass at 0.5 Hz
              (zero-phase, fast).
            - ``"wavelet"``  – Daubechies-4 wavelet decomposition at level 6;
              the approximation sub-band (baseline estimate) is zeroed and
              the signal is reconstructed. More effective for non-stationary
              baselines but ~10× slower.

    Returns:
        Baseline-corrected signal with the same length as *signal*.

    Raises:
        ValueError: If *method* is not one of ``"highpass"`` or ``"wavelet"``.
    """
    if method == "highpass":
        nyq = fs / 2.0
        b, a = sp.butter(4, 0.5 / nyq, btype="high")
        return sp.filtfilt(b, a, signal)

    if method == "wavelet":
        import pywt  # optional dependency; checked at call time

        coeffs = pywt.wavedec(signal, "db4", level=6)
        coeffs[0] = np.zeros_like(coeffs[0])  # zero the baseline approximation
        reconstructed = pywt.waverec(coeffs, "db4")
        # waverec may return one extra sample due to even-length padding
        return reconstructed[: len(signal)]

    raise ValueError(
        f"Unknown baseline removal method: '{method}'. "
        "Valid options are 'highpass' or 'wavelet'."
    )


def preprocess(
    signal: np.ndarray,
    fs: int,
    config: dict,
) -> np.ndarray:
    """Run the full preprocessing pipeline defined by a configuration dict.

    Steps are applied in order: bandpass → notch → baseline removal.
    Each step can be individually enabled or disabled via *config*.

    Args:
        signal: 1-D raw ECG signal.
        fs: Sampling frequency in Hz.
        config: Dict matching the ``preprocessing`` key of ``default.yaml``.
                Expected structure::

                    bandpass:  {enabled: bool, lowcut: float, highcut: float, order: int}
                    notch:     {enabled: bool, freq: float, quality_factor: float}
                    baseline:  {enabled: bool, method: str}

    Returns:
        Preprocessed signal with the same shape as *signal*.
    """
    out = signal.copy()

    bp = config.get("bandpass", {})
    if bp.get("enabled", True):
        out = bandpass_filter(
            out,
            fs,
            lowcut=bp.get("lowcut", 0.5),
            highcut=bp.get("highcut", 40.0),
            order=bp.get("order", 4),
        )

    notch = config.get("notch", {})
    if notch.get("enabled", True):
        out = notch_filter(
            out,
            fs,
            freq=notch.get("freq", 50.0),
            quality_factor=notch.get("quality_factor", 30.0),
        )

    bl = config.get("baseline", {})
    if bl.get("enabled", True):
        out = remove_baseline(out, fs, method=bl.get("method", "highpass"))

    return out
