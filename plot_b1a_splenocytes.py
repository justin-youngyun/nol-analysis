#!/usr/bin/env python
"""
Splenic B-1 populations after SCI, by cohort.

Plots the two columns of the FlowJo B-1 table against the cohort split
(uninjured, and vehicle vs NM72 at 6 h and 24 h):

  IgM+   Cells / Single Cells / Live / CD45+ / B Cells / IgM+ IgDlo /
         CD43+ B220lo, labelled B-1a, Freq. of B Cells
  IgM-   the IgM- column of that same table

Every animal is drawn, because n is 3-4 per group and a bar alone would hide
that. Each group gets its individual points, a mean crossbar, and SEM whiskers;
the uninjured mean +/- SEM is carried across the injured groups as a shaded
baseline band.

The figure is descriptive. --stats annotates the vehicle-vs-NM72 contrasts, but
note that at these group sizes the comparison is badly underpowered.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Data, transcribed from the cohort sheet and the FlowJo tables
# ---------------------------------------------------------------------------

# (animal, timepoint, treatment, IgM+ %, IgM- %). NaN = on the cohort sheet but
# no .fcs in the exported table.
RECORDS: list[tuple[int, str, str, float, float]] = [
    (497, "Uninjured", "Uninjured", 3.53, 3.16),
    (498, "Uninjured", "Uninjured", 5.68, 2.63),
    (499, "Uninjured", "Uninjured", 4.24, 2.89),

    (483, "6 h", "Vehicle", 4.23, 1.63),
    (484, "6 h", "Vehicle", 8.60, 2.79),
    (487, "6 h", "Vehicle", 4.70, 2.35),
    (488, "6 h", "Vehicle", 3.12, 3.41),
    (503, "6 h", "Vehicle", np.nan, np.nan),
    (504, "6 h", "Vehicle", np.nan, np.nan),

    (481, "6 h", "NM72", 4.44, 1.93),
    (485, "6 h", "NM72", 3.99, 1.27),
    (486, "6 h", "NM72", 4.25, 2.09),
    (501, "6 h", "NM72", np.nan, np.nan),
    (502, "6 h", "NM72", 5.48, 3.14),

    (489, "24 h", "Vehicle", 17.60, 5.88),
    (491, "24 h", "Vehicle", 6.68, 1.71),
    (492, "24 h", "Vehicle", 8.15, 2.36),
    (495, "24 h", "Vehicle", 7.01, 3.51),

    (490, "24 h", "NM72", 6.21, 2.02),
    (493, "24 h", "NM72", 4.92, 3.44),
    (494, "24 h", "NM72", 2.93, 2.29),
    (496, "24 h", "NM72", 4.63, 1.35),
]

EXCLUDED: dict[int, str] = {489: "excluded by request"}

# What each column is and how to label it. flowjo_mean/sd are that table's own
# footer over every acquired sample (489 included) and are used only as a
# transcription check, never as an analysis input.
MEASURES: dict[str, dict] = {
    "igm_pos": {
        "column": "igm_pos_pct",
        "title": "Splenic B-1a frequency after SCI",
        "panel": "IgM$^+$  (B-1a)",
        "ylabel": "B-1a (% of B cells)",
        "gating": "Live / CD45$^+$ / B cells / IgM$^+$ IgD$^{lo}$ / CD43$^+$ B220$^{lo}$",
        "flowjo_mean": 5.81,
        "flowjo_sd": 3.26,
    },
    "igm_neg": {
        "column": "igm_neg_pct",
        "title": "Splenic IgM$^-$ B-1 fraction after SCI",
        "panel": "IgM$^-$",
        "ylabel": "IgM$^-$ (% of B cells)",
        "gating": "IgM$^-$ column of the same B-1 table  ·  read as % of B cells",
        "flowjo_mean": 2.62,
        "flowjo_sd": 1.05,
    },
}

FLOWJO_N = 19

# Group order along x, with the spacing that separates the timepoint blocks.
GROUPS: list[tuple[str, str, float]] = [
    ("Uninjured", "Uninjured", 0.00),
    ("6 h", "Vehicle", 1.35),
    ("6 h", "NM72", 2.15),
    ("24 h", "Vehicle", 3.50),
    ("24 h", "NM72", 4.30),
]

BRACKETS = {"6 h": (1.35, 2.15), "24 h": (3.50, 4.30)}

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
}


def build_frame() -> pd.DataFrame:
    """Tidy table of the cohort, with acquisition and exclusion flags."""
    df = pd.DataFrame(
        RECORDS, columns=["animal", "timepoint", "treatment", "igm_pos_pct", "igm_neg_pct"]
    )
    df["acquired"] = df["igm_pos_pct"].notna()
    df["excluded"] = df["animal"].isin(EXCLUDED)
    df["exclusion_reason"] = df["animal"].map(EXCLUDED).fillna("")
    return df


def check_transcription(df: pd.DataFrame) -> bool:
    """Warn if the transcribed values drift from either FlowJo table footer."""
    all_ok = True
    for key, meta in MEASURES.items():
        vals = df.loc[df["acquired"], meta["column"]].to_numpy(dtype=float)
        n, mean, sd = vals.size, float(np.mean(vals)), float(np.std(vals, ddof=1))
        ok = (n == FLOWJO_N
              and abs(mean - meta["flowjo_mean"]) < 0.01
              and abs(sd - meta["flowjo_sd"]) < 0.01)
        all_ok &= ok
        tag = "matches" if ok else "DOES NOT MATCH"
        print(f"Transcription check [{key}]: n={n} mean={mean:.2f} SD={sd:.2f} "
              f"({tag} FlowJo n={FLOWJO_N} mean={meta['flowjo_mean']} SD={meta['flowjo_sd']})")
        if not ok:
            print(f"  ^ re-check the {key} column against the FlowJo table before using "
                  "this figure.", file=sys.stderr)
    return all_ok


def analysis_set(df: pd.DataFrame) -> pd.DataFrame:
    """Acquired, not excluded."""
    return df[df["acquired"] & ~df["excluded"]].copy()


def _jitter(n: int, width: float = 0.13) -> np.ndarray:
    if n <= 1:
        return np.zeros(max(n, 0))
    return np.linspace(-width, width, n)


def _mean_sem(vals: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(vals))
    sem = float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
    return mean, sem


def _group_values(data: pd.DataFrame, timepoint: str, treatment: str, col: str) -> np.ndarray:
    return data.loc[
        (data["timepoint"] == timepoint) & (data["treatment"] == treatment), col
    ].to_numpy(dtype=float)


def summarize(data: pd.DataFrame, measure: str) -> pd.DataFrame:
    """Per-group n, mean, SD, SEM, in plotting order."""
    col = MEASURES[measure]["column"]
    rows = []
    for timepoint, treatment, _x in GROUPS:
        vals = _group_values(data, timepoint, treatment, col)
        if vals.size == 0:
            continue
        mean, sem = _mean_sem(vals)
        rows.append({
            "timepoint": timepoint,
            "treatment": treatment,
            "n": vals.size,
            "mean": mean,
            "sd": float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan,
            "sem": sem,
        })
    return pd.DataFrame(rows)


def contrasts(data: pd.DataFrame, measure: str) -> pd.DataFrame:
    """Welch's t-test, vehicle vs NM72 at each injured timepoint."""
    col = MEASURES[measure]["column"]
    rows = []
    for timepoint in ("6 h", "24 h"):
        veh = _group_values(data, timepoint, "Vehicle", col)
        nm = _group_values(data, timepoint, "NM72", col)
        if veh.size < 2 or nm.size < 2:
            continue
        t, p = stats.ttest_ind(veh, nm, equal_var=False)
        rows.append({
            "timepoint": timepoint,
            "n_vehicle": veh.size,
            "n_nm72": nm.size,
            "diff": float(np.mean(nm) - np.mean(veh)),
            "t": float(t),
            "p": float(p),
        })
    return pd.DataFrame(rows)


def _p_label(p: float) -> str:
    return f"p = {p:.3f}" if p >= 0.001 else "p < 0.001"


def draw_panel(ax, data: pd.DataFrame, measure: str, c: dict, show_stats: bool = False,
               panel_title: str | None = None, show_legend: bool = True) -> None:
    """Draw one measure onto ax."""
    meta = MEASURES[measure]
    col = meta["column"]
    ax.set_facecolor(c["surface"])

    unin = data.loc[data["treatment"] == "Uninjured", col].to_numpy(dtype=float)
    if unin.size:
        u_mean, u_sem = _mean_sem(unin)
        ax.axhspan(u_mean - u_sem, u_mean + u_sem, color=c["band"], alpha=0.13,
                   zorder=0, linewidth=0)
        ax.axhline(u_mean, color=c["band"], linewidth=0.9, alpha=0.55, zorder=0)

    ymax = float(np.nanmax(data[col])) if len(data) else 1.0
    top = ymax * (1.45 if show_stats else 1.30)

    for timepoint, treatment, x in GROUPS:
        vals = _group_values(data, timepoint, treatment, col)
        if vals.size == 0:
            continue
        hue = c[treatment]
        mean, sem = _mean_sem(vals)
        ax.errorbar(x, mean, yerr=sem, fmt="none", ecolor=hue,
                    elinewidth=1.4, capsize=5, capthick=1.4, zorder=2)
        ax.hlines(mean, x - 0.23, x + 0.23, color=hue, linewidth=2.4, zorder=3)
        ax.scatter(x + _jitter(vals.size), vals, s=62, facecolor=hue, alpha=0.92,
                   edgecolor=c["surface"], linewidth=1.3, zorder=4)
        ax.text(x, top * 0.035, f"n={vals.size}", ha="center", va="bottom",
                fontsize=8.5, color=c["text_muted"])

    ax.set_xlim(-0.75, 5.05)
    ax.set_ylim(0, top)
    ax.set_xticks([x for _tp, _tr, x in GROUPS])
    ax.set_xticklabels([tr for _tp, tr, _x in GROUPS], color=c["text_secondary"])
    ax.set_ylabel(meta["ylabel"], color=c["text"], fontsize=11)
    ax.yaxis.grid(True, color=c["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(c["grid"])
    ax.tick_params(colors=c["text_secondary"], length=4, width=0.8)

    for timepoint, (lo, hi) in BRACKETS.items():
        ax.annotate("", xy=(lo, -0.105), xytext=(hi, -0.105),
                    xycoords=("data", "axes fraction"), textcoords=("data", "axes fraction"),
                    arrowprops=dict(arrowstyle="-", color=c["text_muted"], linewidth=0.9))
        ax.text((lo + hi) / 2, -0.155, timepoint, ha="center", va="top",
                transform=ax.get_xaxis_transform(), fontsize=10, color=c["text_secondary"])

    if show_stats:
        for _, row in contrasts(data, measure).iterrows():
            lo, hi = BRACKETS[row["timepoint"]]
            y = float(data.loc[data["timepoint"] == row["timepoint"], col].max()) + top * 0.085
            ax.plot([lo, lo, hi, hi], [y, y + top * 0.022, y + top * 0.022, y],
                    color=c["text_muted"], linewidth=0.9)
            ax.text((lo + hi) / 2, y + top * 0.032, _p_label(row["p"]), ha="center",
                    va="bottom", fontsize=8.5, color=c["text_secondary"])

    if show_legend:
        handles = [
            plt.Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor=c[t],
                       markeredgecolor=c["surface"], markeredgewidth=1.2, label=t)
            for t in ("Uninjured", "Vehicle", "NM72")
        ]
        leg = ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9.5,
                        handletextpad=0.4, borderaxespad=0.2, ncol=3, columnspacing=1.4)
        for text in leg.get_texts():
            text.set_color(c["text_secondary"])

    if panel_title:
        ax.set_title(panel_title, fontsize=10.5, fontweight="bold", color=c["text"],
                     loc="left", pad=10)


def _footnote(c: dict, fig, extra: list[str] | None = None) -> None:
    notes = ["489 excluded (24 h vehicle)",
             "501, 503, 504 on the cohort sheet, not in the export",
             "shaded band = uninjured mean ± SEM"]
    if extra:
        notes += extra
    fig.text(0.085, 0.01, "  ·  ".join(notes), fontsize=8, color=c["text_muted"], ha="left")


def plot_measure(data: pd.DataFrame, measure: str, out_path: Path, dark: bool = False,
                 show_stats: bool = False, title: str | None = None) -> Path:
    """Single-panel figure for one measure."""
    c = DARK if dark else LIGHT
    meta = MEASURES[measure]
    out_path = Path(out_path)

    fig, ax = plt.subplots(figsize=(7.0, 5.0), facecolor=c["surface"])
    draw_panel(ax, data, measure, c, show_stats=show_stats)
    ax.set_title(f"{meta['gating']}  ·  mean ± SEM, every animal shown",
                 fontsize=9, color=c["text_secondary"], loc="left", pad=14)
    fig.suptitle(title or meta["title"], fontsize=13, fontweight="bold", color=c["text"],
                 x=0.085, ha="left", y=0.995)
    _footnote(c, fig)
    fig.tight_layout(rect=[0.02, 0.055, 1, 0.955])
    fig.savefig(out_path, bbox_inches="tight", facecolor=c["surface"], dpi=200)
    plt.close(fig)
    return out_path


def plot_both(data: pd.DataFrame, out_path: Path, dark: bool = False,
              show_stats: bool = False,
              title: str = "Splenic B-1 populations after SCI") -> Path:
    """Two-panel figure, IgM+ beside IgM-, each on its own y-scale."""
    c = DARK if dark else LIGHT
    out_path = Path(out_path)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), facecolor=c["surface"])
    for ax, measure, legend in zip(axes, ("igm_pos", "igm_neg"), (True, False)):
        draw_panel(ax, data, measure, c, show_stats=show_stats,
                   panel_title=MEASURES[measure]["panel"], show_legend=legend)
    fig.suptitle(title, fontsize=13, fontweight="bold", color=c["text"],
                 x=0.045, ha="left", y=0.995)
    _footnote(c, fig, extra=["panels have independent y-scales"])
    fig.tight_layout(rect=[0.01, 0.055, 1, 0.945])
    fig.savefig(out_path, bbox_inches="tight", facecolor=c["surface"], dpi=200)
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Plot splenic B-1 frequencies by SCI cohort.")
    p.add_argument("--measure", choices=["igm_pos", "igm_neg", "both"], default="both",
                   help="Which column to plot. 'both' draws a two-panel figure.")
    p.add_argument("--output", default=None,
                   help="Output image path. Defaults to outputs/b1_<measure>.png.")
    p.add_argument("--csv", default=None, help="Also write the tidy per-animal table here.")
    p.add_argument("--dark", action="store_true", help="Render on the dark surface.")
    p.add_argument("--stats", action="store_true",
                   help="Annotate vehicle-vs-NM72 Welch t-tests (underpowered at this n).")
    p.add_argument("--title", default=None)
    ns = p.parse_args(argv if argv is not None else sys.argv[1:])

    df = build_frame()
    check_transcription(df)
    data = analysis_set(df)

    for key in (MEASURES if ns.measure == "both" else [ns.measure]):
        print(f"\n[{key}] per group (acquired, 489 excluded):")
        print(summarize(data, key).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        con = contrasts(data, key)
        if not con.empty:
            print(f"[{key}] vehicle vs NM72, Welch's t-test "
                  "(n=3-4 per group: descriptive, not a powered test):")
            print(con.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    out_path = Path(ns.output) if ns.output else Path(f"outputs/b1_{ns.measure}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if ns.measure == "both":
        plot_both(data, out_path, dark=ns.dark, show_stats=ns.stats,
                  title=ns.title or "Splenic B-1 populations after SCI")
    else:
        plot_measure(data, ns.measure, out_path, dark=ns.dark, show_stats=ns.stats,
                     title=ns.title)
    print(f"\nSaved: {out_path}")

    if ns.csv:
        csv_path = Path(ns.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
