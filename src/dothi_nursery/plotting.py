"""Matplotlib figures reproducing the notebook's output.

Two figures, exactly as in ``Dothi_model_for_RSE.ipynb``:

1. ``plot_timecourse`` -- two step plots over time (symptomatic cases, and
   symptomatic + removed "yield loss") with the four control strategies
   overlaid, seasonal shading, and a line marking the control month.
2. ``plot_snapshots`` -- a 4x3 grid of field maps (one row per strategy;
   columns are just-before / just-after / at-30-months).
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")  # headless backend, safe inside a web server
import matplotlib.colors as clt
import matplotlib.patches as patches
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11})

# Grid value -> colour (0 S, 1 E, 2 I, 4 empty).
_FIELD_CMAP = clt.ListedColormap(["b", "plum", "r", "k"])
_LEGEND = [
    ("b", "Susceptible"),
    ("plum", "Asymptomatic"),
    ("r", "Symptomatic"),
    ("k", "Empty / removed"),
]

# Seasonal (transmission) windows to shade, in absolute months.
_SEASONS = [(3, 11), (15, 23), (27, 30)]


def _shade_seasons(ax, control):
    for start, end in _SEASONS:
        ax.axvspan(start, end, color="red", alpha=0.1)
    ax.axvline(control, 0, 1, color="C0")


def plot_timecourse(result: dict):
    """Symptomatic and yield-loss time courses for the four strategies."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    colors = ["r", "b", "plum", "k"]
    labels = result["labels"]

    for s, color, label in zip(result["series"], colors, labels):
        ax[0].step(s["t"], s["symptomatic"], color=color, label=label, where="post")
        ax[1].step(s["t"], s["yield_loss"], color=color, label=label, where="post")

    for a in ax:
        _shade_seasons(a, result["control"])
        a.set_xlabel("Months")
        a.set_xlim([0, 30])
        a.set_xticks([0, 6, 12, 18, 24, 30])

    # Round axis tops up to a tidy number (and never collapse to zero height).
    top_inf = max(20, math.ceil(result["max_symptomatic"] / 20) * 20)
    top_yld = max(20, math.ceil(result["max_yield"] / 50) * 50)
    ax[0].set_ylim([0, top_inf])
    ax[1].set_ylim([0, top_yld])
    ax[0].set_ylabel("Symptomatic cases")
    ax[1].set_ylabel("Symptomatic cases + removed")
    ax[1].legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    return fig


def plot_snapshots(result: dict):
    """4x3 grid of field maps: rows = strategies, cols = time points."""
    gs = result["gridsaves"]
    width, length = result["width"], result["length"]
    labels = result["labels"]

    # gridsaves index of the "after control" / "at 30 months" snapshot per row.
    after_idx = [1, 4, 7, 10]
    end_idx = [2, 5, 8, 11]
    col_titles = ["Just before control", "Just after control", "At 30 months"]

    fig, ax = plt.subplots(4, 3, figsize=(9, 11))

    def show(a, gridimg):
        a.imshow(
            gridimg,
            interpolation="none",
            cmap=_FIELD_CMAP,
            vmin=0,
            vmax=4,
            extent=[0, width, 0, length],
            origin="lower",
            zorder=0,
        )
        a.axhline(1, c="w", lw=2)          # separates the external row from the field
        a.set_xticks([])
        a.set_yticks([])

    for row in range(4):
        show(ax[row, 0], gs[0])            # every row shares the pre-control state
        show(ax[row, 1], gs[after_idx[row]])
        show(ax[row, 2], gs[end_idx[row]])
        ax[row, 0].set_ylabel(labels[row], fontsize=10)

    for col in range(3):
        ax[0, col].set_title(col_titles[col])

    legend_handles = [patches.Patch(color=c, label=l) for c, l in _LEGEND]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def placeholder(message: str):
    """A blank figure carrying an instruction, shown before the first run."""
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=14, color="#555")
    ax.axis("off")
    return fig
