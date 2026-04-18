"""ECG plotting utilities."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# Shared style constants
_BLUE = "#1f77b4"
_GRAY = "#7f7f7f"
_RED = "#d62728"

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def plot_ecg(
    signal: np.ndarray,
    fs: int,
    rpeaks: np.ndarray | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot an ECG signal with optional R-peak markers.

    Args:
        signal: 1-D ECG signal to plot.
        fs: Sampling frequency in Hz. Used to build the time axis.
        rpeaks: Optional 1-D integer array of R-peak sample indices.
                Red triangles are drawn at each detected peak.
        title: Optional axes title. Defaults to ``"ECG Signal"``.
        ax: Optional ``Axes`` object to draw into. A new ``Figure`` with a
            single subplot is created when ``None``.

    Returns:
        The ``Figure`` containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 3))
    else:
        fig = ax.get_figure()

    t = np.arange(len(signal)) / float(fs)

    ax.plot(t, signal, color=_BLUE, linewidth=0.8, label="ECG")

    if rpeaks is not None and len(rpeaks) > 0:
        valid = rpeaks[(rpeaks >= 0) & (rpeaks < len(signal))]
        ax.scatter(
            valid / float(fs),
            signal[valid],
            color=_RED,
            marker="^",
            s=45,
            zorder=5,
            label="R-peaks",
        )
        ax.legend(loc="upper right", fontsize=8, framealpha=0.7)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title or "ECG Signal")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())

    fig.tight_layout()
    return fig


def plot_before_after(
    raw: np.ndarray,
    filtered: np.ndarray,
    fs: int,
    title: str = "ECG: Before vs After Preprocessing",
) -> plt.Figure:
    """Plot raw and preprocessed ECG signals in aligned subplots.

    Both panels share the same X axis (time in seconds) to facilitate
    visual comparison of filter effects.

    Args:
        raw: 1-D raw ECG signal.
        filtered: 1-D preprocessed ECG signal. Must have the same length as
                  *raw* for aligned axes.
        fs: Sampling frequency in Hz.
        title: Overall figure suptitle.

    Returns:
        The ``Figure`` with two vertically stacked subplots.
    """
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(12, 5), sharex=True,
        gridspec_kw={"hspace": 0.08},
    )

    t = np.arange(len(raw)) / float(fs)

    ax_top.plot(t, raw, color=_GRAY, linewidth=0.7)
    ax_top.set_ylabel("Amplitude (raw)")
    ax_top.set_title("Raw")
    ax_top.grid(True, linestyle="--", alpha=0.35)

    ax_bot.plot(t, filtered, color=_BLUE, linewidth=0.7)
    ax_bot.set_ylabel("Amplitude (filtered)")
    ax_bot.set_xlabel("Time (s)")
    ax_bot.set_title("Filtered")
    ax_bot.grid(True, linestyle="--", alpha=0.35)
    ax_bot.xaxis.set_minor_locator(mticker.AutoMinorLocator())

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig
