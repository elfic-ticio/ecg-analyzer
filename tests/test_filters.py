"""Unit tests for src.preprocessing.filters."""

import numpy as np
import pytest

from src.preprocessing.filters import (
    bandpass_filter,
    notch_filter,
    preprocess,
    remove_baseline,
)

FS = 360  # MIT-BIH native sampling rate


def _sine(freq_hz: float, fs: int = FS, duration: float = 10.0) -> np.ndarray:
    """Generate a pure sine wave."""
    t = np.arange(int(duration * fs)) / fs
    return np.sin(2 * np.pi * freq_hz * t)


def _power_ratio(filtered: np.ndarray, raw: np.ndarray) -> float:
    # Trim 5 % from each end to exclude filtfilt edge transients.
    # The 0.5 Hz high-pass pole has a time constant of ~0.32 s; with 10 s
    # signals and 5 % trim (~0.5 s) the transient is safely excluded.
    trim = max(1, len(filtered) // 20)
    f = filtered[trim:-trim]
    r = raw[trim:-trim]
    return float(np.var(f) / np.var(r))


# ─── bandpass_filter ──────────────────────────────────────────────────────────

class TestBandpassFilter:
    def test_attenuates_below_lowcut(self):
        """0.1 Hz sine must be reduced to < 1% of its original power."""
        raw = _sine(0.1)
        out = bandpass_filter(raw, FS, lowcut=0.5, highcut=40.0)
        assert _power_ratio(out, raw) < 0.01

    def test_attenuates_above_highcut(self):
        """100 Hz sine (2.5× highcut) must be reduced to < 1% of its original power.

        60 Hz is too close to the 40 Hz cutoff for a 4th-order Butterworth
        (theoretical attenuation only ~96% at 1.5× cutoff). At 2.5× cutoff
        attenuation exceeds 99.9%, giving reliable headroom.
        """
        raw = _sine(100.0)
        out = bandpass_filter(raw, FS, lowcut=0.5, highcut=40.0)
        assert _power_ratio(out, raw) < 0.01

    def test_preserves_passband(self):
        """10 Hz sine (well inside passband) must retain > 90% of its power."""
        raw = _sine(10.0)
        out = bandpass_filter(raw, FS, lowcut=0.5, highcut=40.0)
        assert _power_ratio(out, raw) > 0.90

    def test_output_shape(self):
        raw = _sine(10.0)
        out = bandpass_filter(raw, FS)
        assert out.shape == raw.shape


# ─── notch_filter ─────────────────────────────────────────────────────────────

class TestNotchFilter:
    def test_attenuates_50hz(self):
        """50 Hz component must be reduced to < 5% of its power."""
        raw = _sine(50.0)
        out = notch_filter(raw, FS, freq=50.0)
        assert _power_ratio(out, raw) < 0.05

    def test_attenuates_60hz(self):
        """60 Hz notch variant must suppress the 60 Hz component."""
        raw = _sine(60.0)
        out = notch_filter(raw, FS, freq=60.0)
        assert _power_ratio(out, raw) < 0.05

    def test_passes_10hz(self):
        """10 Hz sine must not be substantially affected by a 50 Hz notch."""
        raw = _sine(10.0)
        out = notch_filter(raw, FS, freq=50.0)
        assert _power_ratio(out, raw) > 0.90


# ─── preprocess ───────────────────────────────────────────────────────────────

_DEFAULT_CONFIG = {
    "bandpass": {"enabled": True, "lowcut": 0.5, "highcut": 40.0, "order": 4},
    "notch": {"enabled": True, "freq": 50.0, "quality_factor": 30.0},
    "baseline": {"enabled": True, "method": "highpass"},
}


class TestPreprocess:
    def test_output_shape_preserved(self):
        sig = _sine(10.0) + 0.1 * _sine(0.1) + 0.05 * _sine(50.0)
        out = preprocess(sig, FS, _DEFAULT_CONFIG)
        assert out.shape == sig.shape

    def test_disabled_steps_return_copy(self):
        """All steps disabled → output values equal input values."""
        rng = np.random.default_rng(42)
        sig = rng.standard_normal(FS * 5)
        cfg = {
            "bandpass": {"enabled": False},
            "notch": {"enabled": False},
            "baseline": {"enabled": False},
        }
        out = preprocess(sig, FS, cfg)
        np.testing.assert_array_equal(out, sig)

    def test_does_not_mutate_input(self):
        sig = _sine(10.0)
        original = sig.copy()
        preprocess(sig, FS, _DEFAULT_CONFIG)
        np.testing.assert_array_equal(sig, original)

    def test_invalid_baseline_method(self):
        sig = _sine(10.0)
        cfg = {"bandpass": {"enabled": False}, "notch": {"enabled": False},
               "baseline": {"enabled": True, "method": "unknown_method"}}
        with pytest.raises(ValueError, match="unknown_method"):
            preprocess(sig, FS, cfg)
