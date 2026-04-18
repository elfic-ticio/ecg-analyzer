"""Unit tests for src.features.rpeaks."""

import numpy as np
import neurokit2 as nk
import pytest

from src.features.rpeaks import detect_rpeaks, heart_rate

FS = 500  # Hz — used throughout
DURATION = 10  # seconds
HR_TARGET = 70  # bpm


def _synthetic_ecg(seed: int = 0) -> np.ndarray:
    """Generate a reproducible 10-second ECG at 70 bpm."""
    return nk.ecg_simulate(
        duration=DURATION,
        sampling_rate=FS,
        heart_rate=HR_TARGET,
        noise=0.05,
        random_state=seed,
    )


class TestDetectRpeaks:
    def test_peak_count_in_expected_range(self):
        """At 70 bpm over 10 s expect 10–13 R-peaks (tolerates edge effects)."""
        ecg = _synthetic_ecg()
        indices, _ = detect_rpeaks(ecg, FS)
        assert 10 <= len(indices) <= 13, (
            f"Expected 10–13 peaks for {HR_TARGET} bpm / {DURATION} s, "
            f"got {len(indices)}"
        )

    def test_times_equal_indices_over_fs(self):
        """times_s must satisfy times_s[i] == indices[i] / fs exactly."""
        ecg = _synthetic_ecg()
        indices, times_s = detect_rpeaks(ecg, FS)
        np.testing.assert_allclose(times_s, indices / FS, rtol=1e-6)

    def test_indices_are_sorted(self):
        """R-peak indices must be strictly increasing."""
        ecg = _synthetic_ecg()
        indices, _ = detect_rpeaks(ecg, FS)
        assert np.all(np.diff(indices) > 0), "R-peak indices are not monotonically increasing"

    def test_indices_within_signal_bounds(self):
        """Every R-peak index must be a valid sample position."""
        ecg = _synthetic_ecg()
        indices, _ = detect_rpeaks(ecg, FS)
        assert np.all(indices >= 0) and np.all(indices < len(ecg))

    def test_indices_dtype_is_integer(self):
        ecg = _synthetic_ecg()
        indices, _ = detect_rpeaks(ecg, FS)
        assert np.issubdtype(indices.dtype, np.integer)


class TestHeartRate:
    def test_hr_close_to_simulated_rate(self):
        """HR estimate must be within ±10 bpm of the simulated 70 bpm."""
        ecg = nk.ecg_simulate(
            duration=30, sampling_rate=FS, heart_rate=HR_TARGET,
            noise=0.05, random_state=1,
        )
        indices, _ = detect_rpeaks(ecg, FS)
        hr = heart_rate(indices, FS)
        assert abs(hr - HR_TARGET) <= 10, (
            f"HR estimate {hr:.1f} bpm is too far from target {HR_TARGET} bpm"
        )

    def test_raises_with_single_peak(self):
        """heart_rate must raise ValueError when fewer than 2 peaks are given."""
        with pytest.raises(ValueError, match="2 R-peaks required"):
            heart_rate(np.array([100]), FS)

    def test_raises_with_empty_array(self):
        with pytest.raises(ValueError):
            heart_rate(np.array([], dtype=int), FS)

    def test_known_rr_interval(self):
        """Two peaks 500 ms apart (fs=1000) → 120 bpm."""
        rpeaks = np.array([0, 500], dtype=int)
        hr = heart_rate(rpeaks, fs=1000)
        assert abs(hr - 120.0) < 0.01
