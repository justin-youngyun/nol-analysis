#!/usr/bin/env python
"""
Shared pieces for the SCI splenocyte figures: the cohort split, the palette, and
the group-scatter panel both flow figures are built from.

The cohort assignments come from the surgery sheet and are the same for every
panel; which animals actually have data, and which are excluded, is per-panel
and lives with the script that reads that panel.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Editable vector output. Text stays text in the PDF (TrueType, not Type 3
# outlines) and in the SVG (<text> elements, not glyph paths), so every label,
# p-value, point and bracket can be selected and changed in Illustrator,
# Inkscape or PowerPoint. Liberation Sans shares Arial's metrics, so a layout
# computed here holds on a machine that renders the same text in Arial.
matplotlib.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Liberation Sans",
    "mathtext.it": "Liberation Sans:italic",
    "mathtext.bf": "Liberation Sans:bold",
    "mathtext.sf": "Liberation Sans",
    "mathtext.cal": "Liberation Sans:italic",
    "mathtext.tt": "Liberation Mono",
})

# Superscripts are written by mathtext, which names only the font it drew them
# with. Give those runs the same fallback chain as the rest of the text, so a
# viewer without Liberation Sans reaches for Arial rather than its own default.
_SVG_FALLBACK = "'Arial', 'Liberation Sans', 'Helvetica', sans-serif"

# ---------------------------------------------------------------------------
# Cohort split (surgery sheet). 501, 503 and 504 were assigned but have not
# appeared in any export so far.
# ---------------------------------------------------------------------------

COHORTS: dict[int, tuple[str, str]] = {
    497: ("Uninjured", "Uninjured"),
    498: ("Uninjured", "Uninjured"),
    499: ("Uninjured", "Uninjured"),

    483: ("6 h", "Vehicle"), 484: ("6 h", "Vehicle"),
    487: ("6 h", "Vehicle"), 488: ("6 h", "Vehicle"),
    503: ("6 h", "Vehicle"), 504: ("6 h", "Vehicle"),

    481: ("6 h", "NM72"), 485: ("6 h", "NM72"), 486: ("6 h", "NM72"),
    501: ("6 h", "NM72"), 502: ("6 h", "NM72"),

    489: ("24 h", "Vehicle"), 491: ("24 h", "Vehicle"),
    492: ("24 h", "Vehicle"), 495: ("24 h", "Vehicle"),

    490: ("24 h", "NM72"), 493: ("24 h", "NM72"),
    494: ("24 h", "NM72"), 496: ("24 h", "NM72"),
}

# Group order along x, with the spacing that separates the timepoint blocks.
GROUPS: list[tuple[str, str, float]] = [
    ("Uninjured", "Uninjured", 0.00),
    ("6 h", "Vehicle", 1.35),
    ("6 h", "NM72", 2.15),
    ("24 h", "Vehicle", 3.50),
    ("24 h", "NM72", 4.30),
]

BRACKETS = {"6 h": (1.35, 2.15), "24 h": (3.50, 4.30)}

SHORT_TICKS = ["Un", "Veh", "NM72", "Veh", "NM72"]

# Uninjured dropped: the drug-vs-vehicle contrast on its own.
DRUG_GROUPS: list[tuple[str, str, float]] = [
    ("6 h", "Vehicle", 0.00),
    ("6 h", "NM72", 0.80),
    ("24 h", "Vehicle", 2.10),
    ("24 h", "NM72", 2.90),
]
DRUG_BRACKETS = {"6 h": (0.00, 0.80), "24 h": (2.10, 2.90)}
DRUG_TICKS = ["Veh", "NM72", "Veh", "NM72"]

# ---------------------------------------------------------------------------
# Palette. Categorical slots 1 and 2 carry treatment identity; uninjured is
# context, so it takes the de-emphasis gray. Both modes are stepped for their
# own surface rather than flipped.
# ---------------------------------------------------------------------------

LIGHT = {
    "surface": "#ffffff",
    "text": "#0b0b0b",
    "text_secondary": "#52514e",
    "text_muted": "#7a7973",
    "grid": "#e6e5e1",
    "Uninjured": "#84837c",
    "Vehicle": "#2a78d6",
    "NM72": "#eb6834",
    "band": "#84837c",
    "flag": "#d03b3b",
}

DARK = {
    "surface": "#1a1a19",
    "text": "#ffffff",
    "text_secondary": "#c3c2b7",
    "text_muted": "#97968c",
    "grid": "#343431",
    "Uninjured": "#97968c",
    "Vehicle": "#3987e5",
    "NM72": "#d95926",
    "band": "#97968c",
    "flag": "#e66767",
}


def save_figure(fig, out_path, facecolor: str, dpi: int = 200) -> Path:
    """PNG to look at; PDF and SVG to edit. Returns the PNG path."""
    base = Path(out_path).with_suffix("")
    for ext, kw in ((".png", {"dpi": dpi}), (".pdf", {}), (".svg", {})):
        fig.savefig(base.with_suffix(ext), bbox_inches="tight", facecolor=facecolor, **kw)
    svg = base.with_suffix(".svg")
    svg.write_text(svg.read_text().replace("'Liberation Sans'\"", _SVG_FALLBACK + '"')
                                  .replace("'Liberation Sans';", _SVG_FALLBACK + ";"))
    return base.with_suffix(".png")


def palette(dark: bool = False) -> dict:
    return DARK if dark else LIGHT


def jitter(n: int, width: float = 0.13) -> np.ndarray:
    if n <= 1:
        return np.zeros(max(n, 0))
    return np.linspace(-width, width, n)


def mean_sem(vals: np.ndarray) -> tuple[float, float]:
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(vals))
    sem = float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
    return mean, sem


def group_values(data: pd.DataFrame, timepoint: str, treatment: str, col: str) -> np.ndarray:
    v = data.loc[
        (data["timepoint"] == timepoint) & (data["treatment"] == treatment), col
    ].to_numpy(dtype=float)
    return v[~np.isnan(v)]


def legend_handles(c: dict, treatments: tuple[str, ...] = ("Uninjured", "Vehicle", "NM72")) -> list:
    return [
        plt.Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor=c[t],
                   markeredgecolor=c["surface"], markeredgewidth=1.2, label=t)
        for t in treatments
    ]


def draw_panel(ax, data: pd.DataFrame, col: str, c: dict, *, ylabel: str | None = None,
               panel_title: str | None = None, subtitle: str | None = None,
               compact: bool = False, show_ticklabels: bool = True,
               show_brackets: bool = True, show_legend: bool = False,
               show_n: bool = True, baseline: bool = True,
               headroom: float = 1.30, title_color: str | None = None,
               subtitle_color: str | None = None,
               groups: list | None = None, brackets: dict | None = None,
               ticks: list | None = None) -> None:
    """One measure per group: individual points over a mean crossbar with SEM."""
    groups = groups if groups is not None else GROUPS
    brackets = brackets if brackets is not None else BRACKETS
    ax.set_facecolor(c["surface"])

    if baseline:
        unin = group_values(data, "Uninjured", "Uninjured", col)
        if unin.size:
            u_mean, u_sem = mean_sem(unin)
            ax.axhspan(u_mean - u_sem, u_mean + u_sem, color=c["band"], alpha=0.13,
                       zorder=0, linewidth=0)
            ax.axhline(u_mean, color=c["band"], linewidth=0.9, alpha=0.55, zorder=0)

    finite = data[col].to_numpy(dtype=float)
    finite = finite[~np.isnan(finite)]
    ymax = float(np.max(finite)) if finite.size else 1.0
    top = (ymax * headroom) or 1.0

    marker = 42 if compact else 62
    for timepoint, treatment, x in groups:
        vals = group_values(data, timepoint, treatment, col)
        if vals.size == 0:
            continue
        hue = c[treatment]
        mean, sem = mean_sem(vals)
        ax.errorbar(x, mean, yerr=sem, fmt="none", ecolor=hue,
                    elinewidth=1.4, capsize=5, capthick=1.4, zorder=2)
        ax.hlines(mean, x - 0.23, x + 0.23, color=hue, linewidth=2.4, zorder=3)
        ax.scatter(x + jitter(vals.size), vals, s=marker, facecolor=hue, alpha=0.92,
                   edgecolor=c["surface"], linewidth=1.2, zorder=4)
        if show_n:
            ax.text(x, top * 0.03, f"n={vals.size}", ha="center", va="bottom",
                    fontsize=7.5 if compact else 8.5, color=c["text_muted"])

    xs = [x for _tp, _tr, x in groups]
    pad = 0.75 if len(groups) > 4 else 0.62
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(0, top)
    ax.set_xticks(xs)
    if show_ticklabels:
        default_ticks = ticks if ticks is not None else SHORT_TICKS
        labels = default_ticks if compact else [tr for _tp, tr, _x in groups]
        ax.set_xticklabels(labels, color=c["text_secondary"],
                           fontsize=8.5 if compact else 10)
    else:
        ax.set_xticklabels([])
    if ylabel:
        ax.set_ylabel(ylabel, color=c["text"], fontsize=9.5 if compact else 11)
    ax.yaxis.grid(True, color=c["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(c["grid"])
    ax.tick_params(colors=c["text_secondary"], length=4, width=0.8,
                   labelsize=8.5 if compact else 10)

    if show_brackets:
        for timepoint, (lo, hi) in brackets.items():
            ax.annotate("", xy=(lo, -0.105), xytext=(hi, -0.105),
                        xycoords=("data", "axes fraction"),
                        textcoords=("data", "axes fraction"),
                        arrowprops=dict(arrowstyle="-", color=c["text_muted"], linewidth=0.9))
            ax.text((lo + hi) / 2, -0.155, timepoint, ha="center", va="top",
                    transform=ax.get_xaxis_transform(),
                    fontsize=8.5 if compact else 10, color=c["text_secondary"])

    if show_legend:
        leg = ax.legend(handles=legend_handles(c), loc="upper left", frameon=False,
                        fontsize=9.5, handletextpad=0.4, borderaxespad=0.2, ncol=3,
                        columnspacing=1.4)
        for t in leg.get_texts():
            t.set_color(c["text_secondary"])

    if panel_title:
        ax.set_title(panel_title, fontsize=9.5 if compact else 10.5, fontweight="bold",
                     color=title_color or c["text"], loc="left", pad=16 if subtitle else 8)
    if subtitle:
        ax.text(0.0, 1.015, subtitle, transform=ax.transAxes, fontsize=8,
                color=subtitle_color or c["text_muted"], ha="left", va="bottom")
