"""WFDB record loader and stubs for alternative formats."""

from pathlib import Path
from typing import Any

import numpy as np
import wfdb


def load_wfdb_record(
    path: str | Path,
    channel: int = 0,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    """Load a single channel from a WFDB record.

    Works with any MIT-BIH / PhysioNet record stored on disk. The channel
    is extracted as physical units (mV when available).

    Args:
        path: Path to the record *without* file extension
              (e.g. ``"data/mitdb/100"``).
        channel: Zero-based index of the lead to extract.

    Returns:
        A 3-tuple ``(signal, fs, metadata)`` where:

        - **signal** – 1-D ``float64`` array of the ECG channel.
        - **fs** – Sampling frequency in Hz.
        - **metadata** – Dict with keys:

          ``record_name``, ``fs``, ``n_samples``, ``duration_s``,
          ``units``, ``annotations``.

          ``annotations`` is a list of ``(sample_idx: int, symbol: str)``
          tuples, or ``None`` if no annotation file is found.

    Raises:
        FileNotFoundError: If the record header cannot be found at *path*.
        IndexError: If *channel* exceeds the number of signals in the record.
    """
    path = Path(path)
    record = wfdb.rdrecord(str(path), channels=[channel])

    if record.p_signal is None:
        raise ValueError(
            f"Record '{path}' has no physical signal. "
            "Check that the .dat file is present."
        )

    signal: np.ndarray = record.p_signal[:, 0].astype(np.float64)
    fs: int = int(record.fs)

    units = "unknown"
    if record.units and len(record.units) > 0:
        units = record.units[0]

    annotations: list[tuple[int, str]] | None = None
    try:
        ann = wfdb.rdann(str(path), "atr")
        annotations = list(zip(ann.sample.tolist(), ann.symbol))
    except Exception:
        pass

    metadata: dict[str, Any] = {
        "record_name": record.record_name,
        "fs": fs,
        "n_samples": len(signal),
        "duration_s": len(signal) / fs,
        "units": units,
        "annotations": annotations,
    }

    return signal, fs, metadata


# ─── Stubs for future formats ─────────────────────────────────────────────────

def load_csv_record(
    path: str | Path,
    fs: int,
    signal_col: int = 0,
    time_col: int | None = None,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    """Load an ECG signal from a CSV file.

    Not implemented in Phase 1. Raises ``NotImplementedError``.

    Args:
        path: Path to the CSV file.
        fs: Sampling frequency in Hz (must be provided externally).
        signal_col: Column index (or name) containing the ECG samples.
        time_col: Optional column index with timestamps; ignored when ``None``.
    """
    raise NotImplementedError("CSV loader is planned for a future phase.")


def load_edf_record(
    path: str | Path,
    channel: int = 0,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    """Load an ECG signal from an EDF/BDF file (requires ``mne`` or ``pyedflib``).

    Not implemented in Phase 1. Raises ``NotImplementedError``.

    Args:
        path: Path to the EDF file.
        channel: Zero-based channel index to extract.
    """
    raise NotImplementedError("EDF loader is planned for a future phase.")
