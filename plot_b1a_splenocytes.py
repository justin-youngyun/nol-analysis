#!/usr/bin/env python
"""
Splenic B-1a frequency after SCI, by cohort.

Plots the FlowJo readout

    Cells / Single Cells / Live / CD45+ / B Cells / IgM+ IgDlo / CD43+ B220lo
    B-1a, Freq. of B Cells

against the cohort split (uninjured, and vehicle vs NM72 at 6 h and 24 h).

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
# Data, transcribed from the cohort sheet and the FlowJo table
# ---------------------------------------------------------------------------

# (animal, timepoint, treatment, B-1a % of B cells). NaN = on the cohort sheet
# but no .fcs in the exported table.
RECORDS: list[tuple[int, str, str, float]] = [
    (497, "Uninjured", "Uninjured", 3.53),
    (498, "Uninjured", "Uninjured", 5.68),
    (499, "Uninjured", "Uninjured", 4.24),

    (483, "6 h", "Vehicle", 4.23),
    (484, "6 h", "Vehicle", 8.60),
    (487, "6 h", "Vehicle", 4.70),
    (488, "6 h", "Vehicle", 3.12),
    (503, "6 h", "Vehicle", np.nan),
    (504, "6 h", "Vehicle", np.nan),

    (481, "6 h", "NM72", 4.44),
    (485, "6 h", "NM72", 3.99),
    (486, "6 h", "NM72", 4.25),
    (501, "6 h", "NM72", np.nan),
    (502, "6 h", "NM72", 5.48),

    (489, "24 h", "Vehicle", 17.60),
    (491, "24 h", "Vehicle", 6.68),
    (492, "24 h", "Vehicle", 8.15),
    (495, "24 h", "Vehicle", 7.01),

    (490, "24 h", "NM72", 6.21),
    (493, "24 h", "NM72", 4.92),
    (494, "24 h", "NM72", 2.93),
    (496, "24 h", "NM72", 4.63),
]

EXCLUDED: dict[int, str] = {489: "excluded by request"}

# The FlowJo table footer, over every acquired sample including 489. Used as a
# transcription check, not as an analysis input.
FLOWJO_MEAN, FLOWJO_SD, FLOWJO_N = 5.81, 3.26, 19

# Group order along x, with the spacing that separates the timepoint blocks.
GROUPS: list[tuple[str, str, float]] = [
    ("Uninjured", "Uninjured", 0.00),
    ("6 h", "Vehicle", 1.35),
    ("6 h", "NM72", 2.15),
    ("24 h", "Vehicle", 3.50),
    ("24 h", "NM72", 4.30),
]

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

GATING = "Live / CD45$^+$ / B cells / IgM$^+$ IgD$^{lo}$ / CD43$^+$ B220$^{lo}$"


def build_frame() -> pd.DataFrame:
    """Tidy table of the cohort, with acquisition and exclusion flags."""
    df = pd.DataFrame(RECORDS, columns=["animal", "timepoint", "treatment", "b1a_pct"])
    df["acquired"] = df["b1a_pct"].notna()
    df["excluded"] = df["animal"].isin(EXCLUDED)
    df["exclusion_reason"] = df["animal"].map(EXCLUDED).fillna("")
    return df


def check_transcription(df: pd.DataFrame) -> None:
    """Warn if the transcribed values drift from the FlowJo table footer."""
    vals = df.loc[df["acquired"], "b1a_pct"].to_numpy(dtype=float)
    n, mean, sd = vals.size, float(np.mean(vals)), float(np.std(vals, ddof=1))
    ok = n == FLOWJO_N and abs(mean - FLOWJO_MEAN) < 0.01 and abs(sd - FLOWJO_SD) < 0.01
    tag = "matches" if ok else "DOES NOT MATCH"
    print(f"Transcription check: n={n} mean={mean:.2f} SD={sd:.2f} "
          f"({tag} FlowJo n={FLOWJO_N} mean={FLOWJO_MEAN} SD={FLOWJO_SD})")
    if not ok:
        print("  ^ re-check the values against the FlowJo table before using this figure.",
              file=sys.stderr)


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


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    """Per-group n, mean, SD, SEM, in plotting order."""
    rows = []
    for timepoint, treatment, _x in GROUPS:
        vals = data.loc[
            (data["timepoint"] == timepoint) & (data["treatment"] == treatment), "b1a_pct"
        ].to_numpy(dtype=float)
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


def contrasts(data: pd.DataFrame) -> pd.DataFrame:
    """Welch's t-test, vehicle vs NM72 at each injured timepoint."""
    rows = []
    for timepoint in ("6 h", "24 h"):
        veh = data.loc[(data["timepoint"] == timepoint) & (data["treatment"] == "Vehicle"),
                       "b1a_pct"].to_numpy(dtype=float)
        nm = data.loc[(data["timepoint"] == timepoint) & (data["treatment"] == "NM72"),
                      "b1a_pct"].to_numpy(dtype=float)
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


def plot_b1a(data: pd.DataFrame, out_path: Path, dark: bool = False,
             show_stats: bool = False, title: str = "Splenic B-1a frequency after SCI") -> Path:
    """Draw the cohort figure and save it to out_path."""
    c = DARK if dark else LIGHT
    out_path = Path(out_path)

    fig, ax = plt.subplots(figsize=(7.0, 5.0), facecolor=c["surface"])
    ax.set_facecolor(c["surface"])

    # Uninjured baseline band, carried across the injured groups.
    unin = data.loc[data["treatment"] == "Uninjured", "b1a_pct"].to_numpy(dtype=float)
    if unin.size:
        u_mean, u_sem = _mean_sem(unin)
        ax.axhspan(u_mean - u_sem, u_mean + u_sem, xmin=0.0, xmax=1.0,
                   color=c["band"], alpha=0.13, zorder=0, linewidth=0)
        ax.axhline(u_mean, color=c["band"], linewidth=0.9, alpha=0.55, zorder=0)

    ymax = float(np.nanmax(data["b1a_pct"])) if len(data) else 1.0
    top = ymax * (1.45 if show_stats else 1.30)

    for timepoint, treatment, x in GROUPS:
        vals = data.loc[
            (data["timepoint"] == timepoint) & (data["treatment"] == treatment), "b1a_pct"
        ].to_numpy(dtype=float)
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

    # Axes and chrome
    ax.set_xlim(-0.75, 5.05)
    ax.set_ylim(0, top)
    ax.set_xticks([x for _tp, _tr, x in GROUPS])
    ax.set_xticklabels([tr for _tp, tr, _x in GROUPS], color=c["text_secondary"])
    ax.set_ylabel("B-1a (% of B cells)", color=c["text"], fontsize=11)
    ax.yaxis.grid(True, color=c["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(c["grid"])
    ax.tick_params(colors=c["text_secondary"], length=4, width=0.8)

    # Timepoint brackets under the injured pairs
    for timepoint, (lo, hi) in {"6 h": (1.35, 2.15), "24 h": (3.50, 4.30)}.items():
        ax.annotate("", xy=(lo, -0.105), xytext=(hi, -0.105),
                    xycoords=("data", "axes fraction"), textcoords=("data", "axes fraction"),
                    arrowprops=dict(arrowstyle="-", color=c["text_muted"], linewidth=0.9))
        ax.text((lo + hi) / 2, -0.155, timepoint, ha="center", va="top",
                transform=ax.get_xaxis_transform(), fontsize=10, color=c["text_secondary"])

    if show_stats:
        con = contrasts(data)
        for _, row in con.iterrows():
            lo, hi = {"6 h": (1.35, 2.15), "24 h": (3.50, 4.30)}[row["timepoint"]]
            grp = data[(data["timepoint"] == row["timepoint"])]
            y = float(grp["b1a_pct"].max()) + top * 0.085
            ax.plot([lo, lo, hi, hi], [y, y + top * 0.022, y + top * 0.022, y],
                    color=c["text_muted"], linewidth=0.9)
            ax.text((lo + hi) / 2, y + top * 0.032, _p_label(row["p"]), ha="center",
                    va="bottom", fontsize=8.5, color=c["text_secondary"])

    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor=c[t],
                   markeredgecolor=c["surface"], markeredgewidth=1.2, label=t)
        for t in ("Uninjured", "Vehicle", "NM72")
    ]
    leg = ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9.5,
                    handletextpad=0.4, borderaxespad=0.2, ncol=3, columnspacing=1.4)
    for text in leg.get_texts():
        text.set_color(c["text_secondary"])

    fig.suptitle(title, fontsize=13, fontweight="bold", color=c["text"], x=0.085, ha="left", y=0.995)
    ax.set_title(f"{GATING}  ·  mean ± SEM, every animal shown",
                 fontsize=9, color=c["text_secondary"], loc="left", pad=14)

    notes = ["489 excluded (24 h vehicle)", "501, 503, 504 on the cohort sheet, not in the export"]
    if unin.size:
        notes.append("shaded band = uninjured mean ± SEM")
    fig.text(0.085, 0.01, "  ·  ".join(notes), fontsize=8, color=c["text_muted"], ha="left")

    fig.tight_layout(rect=[0.02, 0.055, 1, 0.955])
    fig.savefig(out_path, bbox_inches="tight", facecolor=c["surface"], dpi=200)
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Plot splenic B-1a frequency by SCI cohort.")
    p.add_argument("--output", default="outputs/b1a_splenocytes.png",
                   help="Output image path (.png or .pdf).")
    p.add_argument("--csv", default=None, help="Also write the tidy per-animal table here.")
    p.add_argument("--dark", action="store_true", help="Render on the dark surface.")
    p.add_argument("--stats", action="store_true",
                   help="Annotate vehicle-vs-NM72 Welch t-tests (underpowered at this n).")
    p.add_argument("--title", default="Splenic B-1a frequency after SCI")
    ns = p.parse_args(argv if argv is not None else sys.argv[1:])

    df = build_frame()
    check_transcription(df)
    data = analysis_set(df)

    print("\nPer group (acquired, 489 excluded):")
    print(summarize(data).to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    con = contrasts(data)
    if not con.empty:
        print("\nVehicle vs NM72, Welch's t-test "
              "(n=3-4 per group: treat as descriptive, not a powered test):")
        print(con.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    out_path = Path(ns.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_b1a(data, out_path, dark=ns.dark, show_stats=ns.stats, title=ns.title)
    print(f"\nSaved: {out_path}")

    if ns.csv:
        csv_path = Path(ns.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
