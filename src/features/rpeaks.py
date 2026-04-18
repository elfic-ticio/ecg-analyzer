"""R-peak detection and heart rate estimation."""

import numpy as np
import neurokit2 as nk


def detect_rpeaks(
    signal: np.ndarray,
    fs: int,
    method: str = "neurokit",
) -> tuple[np.ndarray, np.ndarray]:
    """Detect R-peaks in an ECG signal using NeuroKit2.

    The signal is first cleaned with ``nk.ecg_clean`` to remove residual
    high-frequency noise before peak detection.

    Args:
        signal: 1-D ECG signal (raw or already filtered).
        fs: Sampling frequency in Hz.
        method: Peak-detection algorithm passed to ``nk.ecg_peaks``.
                Supported values: ``"neurokit"`` (default), ``"pantompkins1985"``,
                ``"nabian2018"``, ``"gamboa2008"``, ``"ssf"``,
                ``"rodrigues2021"``, ``"promac"``.

    Returns:
        A 2-tuple ``(indices, times_s)`` where:

        - **indices** – 1-D ``int64`` array of R-peak sample positions.
        - **times_s** – 1-D ``float64`` array of R-peak times in seconds
          (``indices / fs``).

    Raises:
        ValueError: If fewer than 2 peaks are detected (HRV not computable).
    """
    cleaned = nk.ecg_clean(signal, sampling_rate=fs, method="neurokit")
    _, info = nk.ecg_peaks(cleaned, sampling_rate=fs, method=method,
                            correct_artifacts=True)
    indices = np.asarray(info["ECG_R_Peaks"], dtype=np.int64)
    times_s = indices / float(fs)
    return indices, times_s


def heart_rate(rpeaks: np.ndarray, fs: int) -> float:
    """Compute mean heart rate in beats per minute from R-peak indices.

    Args:
        rpeaks: 1-D integer array of R-peak sample indices (≥ 2 elements).
        fs: Sampling frequency in Hz.

    Returns:
        Mean HR in bpm.

    Raises:
        ValueError: If *rpeaks* contains fewer than 2 elements.
    """
    if len(rpeaks) < 2:
        raise ValueError(
            f"At least 2 R-peaks required to compute HR, got {len(rpeaks)}."
        )
    rr_s = np.diff(rpeaks) / float(fs)  # RR intervals in seconds
    return float(60.0 / np.mean(rr_s))
