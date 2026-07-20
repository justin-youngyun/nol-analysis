#!/usr/bin/env python
"""
Summary figure for the NOL discrimination-index analysis.

Reads the per-animal table written by nol_analysis.py and draws:
  Panel A: discrimination index per animal (bar of the mean with SEM and the
           individual points jittered on top, chance line at zero)
  Panel B: DI against wall time, to check that discrimination is not driven by
           thigmotaxis (object time is already excluded from the wall denom)
  Panel C: DI against total distance, to check it is not driven by locomotion

The figure is intentionally descriptive. It shows the distribution of the
metric and its relationship to exploration, it does not run group comparisons.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Figure style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "legend.frameon": False,
    "figure.dpi": 150,
})

BAR_COLOR = "#3B7DB0"
POINT_EDGE = "black"


def _jitter(n: int, width: float = 0.12) -> np.ndarray:
    if n <= 1:
        return np.zeros(max(n, 0))
    return np.linspace(-width, width, n)


def _panel_di_per_animal(ax, df: pd.DataFrame) -> None:
    vals = df["DI"].dropna().to_numpy(dtype=float)
    if vals.size == 0:
        ax.text(0.5, 0.5, "no passing animals", transform=ax.transAxes, ha="center")
        return
    m = float(np.mean(vals))
    sem = float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
    ax.bar(0, m, color=BAR_COLOR, alpha=0.55, edgecolor=BAR_COLOR, linewidth=1.5, width=0.6, zorder=1)
    ax.errorbar(0, m, yerr=sem, fmt="none", ecolor=BAR_COLOR, elinewidth=1.5, capsize=4, zorder=2)
    x = _jitter(vals.size)
    ax.scatter(x, vals, s=45, facecolor="white", edgecolor=POINT_EDGE, linewidth=1.0, zorder=3)
    ax.axhline(0, color="black", linestyle=":", linewidth=0.8)
    ax.set_xticks([0])
    ax.set_xticklabels([f"cohort\n(n={vals.size})"])
    ax.set_ylabel("Discrimination index (DI)")
    ax.set_title("DI per animal")


def _panel_scatter(ax, df: pd.DataFrame, xcol: str, xlabel: str, title: str) -> None:
    s = df[[xcol, "DI"]].dropna()
    if s.empty:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
        return
    ax.scatter(
        s[xcol], s["DI"], s=50, facecolor=BAR_COLOR, edgecolor=POINT_EDGE,
        linewidth=0.6, alpha=0.85,
    )
    if len(s) >= 3:
        slope, intercept, r, p, _ = stats.linregress(s[xcol], s["DI"])
        xs = np.linspace(s[xcol].min(), s[xcol].max(), 50)
        ax.plot(xs, slope * xs + intercept, color="black", linewidth=1.0, linestyle="--")
        ax.text(0.03, 0.97, f"r={r:+.2f}, p={p:.3f}", transform=ax.transAxes,
                ha="left", va="top", fontsize=8)
    ax.axhline(0, color="black", linestyle=":", linewidth=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("DI")
    ax.set_title(title)


def plot_summary(per_animal: pd.DataFrame, out_path: Path, title: str = "NOL discrimination summary") -> Path:
    """Draw the three-panel summary figure and save it to out_path."""
    out_path = Path(out_path)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), facecolor="white")
    _panel_di_per_animal(axes[0], per_animal)
    _panel_scatter(axes[1], per_animal, "time_wall_pct", "Wall time (%), object excluded", "DI vs thigmotaxis")
    _panel_scatter(axes[2], per_animal, "distance_m", "Total distance (m)", "DI vs locomotion")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _latest_per_animal_csv(path: Path) -> Path:
    if path.is_file():
        return path
    hits = sorted(glob.glob(str(path / "NOL_per_animal_*.csv")))
    if not hits:
        raise FileNotFoundError(f"No NOL_per_animal_*.csv found under {path}")
    return Path(hits[-1])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Draw the NOL summary figure from a per-animal table.")
    p.add_argument("--input", required=True, help="A per-animal CSV, or a directory containing one.")
    p.add_argument("--output", required=True, help="Output image path (.png or .pdf).")
    p.add_argument("--title", default="NOL discrimination summary")
    ns = p.parse_args(argv if argv is not None else sys.argv[1:])

    csv_path = _latest_per_animal_csv(Path(ns.input))
    df = pd.read_csv(csv_path)
    out = plot_summary(df, Path(ns.output), title=ns.title)
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
