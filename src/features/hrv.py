"""Time-domain HRV metrics and RR-irregularity detection."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.features.quality import is_analyzable, signal_quality_index


# ─── RR interval helpers ──────────────────────────────────────────────────────

def compute_rr_intervals(rpeaks: np.ndarray, fs: int) -> np.ndarray:
    """Convert R-peak sample indices to RR intervals in milliseconds.

    Args:
        rpeaks: 1-D integer array of R-peak sample positions (≥ 2 elements).
        fs: Sampling frequency in Hz.

    Returns:
        1-D ``float64`` array of RR intervals in ms.
    """
    return np.diff(rpeaks).astype(np.float64) / fs * 1000.0


# ─── Individual metrics ───────────────────────────────────────────────────────

def sdnn(rr_intervals: np.ndarray) -> float:
    """Standard deviation of NN intervals (ms).

    Args:
        rr_intervals: RR intervals in ms.

    Returns:
        SDNN in ms.
    """
    return float(np.std(rr_intervals, ddof=1))


def rmssd(rr_intervals: np.ndarray) -> float:
    """Root mean square of successive RR differences (ms).

    Args:
        rr_intervals: RR intervals in ms.

    Returns:
        RMSSD in ms.
    """
    diffs = np.diff(rr_intervals)
    return float(np.sqrt(np.mean(diffs ** 2)))


def pnn50(rr_intervals: np.ndarray) -> float:
    """Percentage of successive NN differences greater than 50 ms.

    Args:
        rr_intervals: RR intervals in ms.

    Returns:
        pNN50 as a percentage ``[0, 100]``.
    """
    diffs = np.abs(np.diff(rr_intervals))
    return float(100.0 * np.sum(diffs > 50.0) / len(diffs))


def mean_rr(rr_intervals: np.ndarray) -> float:
    """Mean RR interval (ms).

    Args:
        rr_intervals: RR intervals in ms.

    Returns:
        Mean RR in ms.
    """
    return float(np.mean(rr_intervals))


# ─── Irregularity detector ────────────────────────────────────────────────────

def is_irregular(
    rr_intervals: np.ndarray,
    threshold_cv: float = 0.15,
    threshold_rmssd_ms: float = 100.0,
) -> bool:
    """Detect RR irregularity as a simple proxy for atrial fibrillation.

    Returns ``True`` only when **both** conditions are satisfied:

    1. ``CV = std(RR) / mean(RR) ≥ threshold_cv``
    2. ``RMSSD ≥ threshold_rmssd_ms``

    The dual criterion avoids false positives from respiratory sinus
    arrhythmia, which raises CV but keeps RMSSD modest. True AF typically
    exceeds both thresholds simultaneously.

    Args:
        rr_intervals: RR intervals in ms (≥ 2 elements).
        threshold_cv: Coefficient-of-variation threshold. Default ``0.15``.
        threshold_rmssd_ms: RMSSD threshold in ms. Default ``100.0``.

    Returns:
        ``True`` if both thresholds are exceeded.
    """
    if len(rr_intervals) < 2:
        return False

    mean = np.mean(rr_intervals)
    if mean == 0.0:
        return False

    cv = float(np.std(rr_intervals, ddof=1) / mean)
    rmssd_val = rmssd(rr_intervals)

    return cv >= threshold_cv and rmssd_val >= threshold_rmssd_ms


# ─── Aggregate feature extractor ──────────────────────────────────────────────

def compute_hrv_features(
    rpeaks: np.ndarray,
    fs: int,
    signal: np.ndarray | None = None,
    min_sqi: float = 0.5,
) -> dict[str, Any]:
    """Aggregate all HRV features into a single dictionary.

    When *signal* is provided, the Signal Quality Index is computed first.
    If the SQI falls below *min_sqi*, all HRV metrics are set to ``None``
    and ``analyzable`` is ``False`` — preventing clinically meaningless
    numbers from propagating downstream.

    Args:
        rpeaks: 1-D integer array of R-peak sample positions.
        fs: Sampling frequency in Hz.
        signal: Optional ECG array used for SQI computation. When ``None``,
                SQI is not evaluated and the signal is assumed analyzable.
        min_sqi: SQI gate threshold. Ignored when *signal* is ``None``.

    Returns:
        Dict with the following keys:

        ``mean_rr`` (ms), ``sdnn`` (ms), ``rmssd`` (ms), ``pnn50`` (%),
        ``hr_bpm``, ``irregular`` (bool), ``sqi`` (float | None),
        ``analyzable`` (bool).

        All metric values are ``None`` when ``analyzable`` is ``False``.
    """
    sqi_val: float | None = None
    analyzable_flag: bool = True

    if signal is not None:
        sqi_val = signal_quality_index(signal, fs)
        analyzable_flag = sqi_val >= min_sqi

    _none: dict[str, Any] = {
        "mean_rr": None,
        "sdnn": None,
        "rmssd": None,
        "pnn50": None,
        "hr_bpm": None,
        "irregular": None,
        "sqi": sqi_val,
        "analyzable": False,
    }

    if not analyzable_flag:
        return _none

    rr = compute_rr_intervals(rpeaks, fs)
    hr_bpm = float(60_000.0 / mean_rr(rr))  # 60 000 ms / mean_rr_ms

    return {
        "mean_rr": mean_rr(rr),
        "sdnn": sdnn(rr),
        "rmssd": rmssd(rr),
        "pnn50": pnn50(rr),
        "hr_bpm": hr_bpm,
        "irregular": is_irregular(rr),
        "sqi": sqi_val,
        "analyzable": True,
    }
