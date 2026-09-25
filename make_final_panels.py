#!/usr/bin/env python
"""
The figure set for presenting this cohort: the result, its specificity control,
the injury response, the Treg compartment, and the target-adjacent null.

Deliberately not every population. CD3 and CD4 frequencies are left out
because they track the acquisition-quality gradient (see
outputs/flow_qc_runorder.png), and a panel nobody should interpret does not
belong in a set people read quickly.

Significance is annotated as exact p, not stars: at n=3-4 the distance between
p=0.02 and p=0.06 is not a category boundary and should not be drawn as one.

The multiple-comparison procedure is a choice, so it is a flag:

    python3 make_final_panels.py                       # Dunnett's T3 (default)
    python3 make_final_panels.py --correction tukey
    python3 make_final_panels.py --correction games-howell
    python3 make_final_panels.py --correction none     # uncorrected Welch

Every run writes PNG, PDF and SVG with live text, so labels, p-values, points
and brackets stay editable in Illustrator, Inkscape or PowerPoint, plus a CSV of
every pairwise p under every procedure for the panels shown.
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

import posthoc as ph
import sci_cohorts as sc

LONG = Path("outputs/prism/per_animal_long.csv")
ORDER = ["Uninjured", "6 h Vehicle", "6 h NM72", "24 h Vehicle", "24 h NM72"]

# (panel label, population as the export names it, y-axis label, pairs to bracket).
# The bracketed pairs are the planned comparisons, fixed before choosing a
# correction; they are drawn whatever their p, so a contrast that fails
# correction is shown failing rather than quietly disappearing.
PANELS = [
    ("B-1a (IgM$^+$)", "B-1a (IgM+)", "% of B cells",
     [("Uninjured", "24 h Vehicle"), ("24 h Vehicle", "24 h NM72")]),
    ("IgM$^-$  (specificity control)", "IgM-", "% of B cells", []),
    # Both arms rise over uninjured at 6 h, so both brackets are drawn: showing
    # only the vehicle one would imply the drug arm was not elevated.
    ("Neutrophils", "Neutrophils", "% of live leukocytes",
     [("Uninjured", "6 h Vehicle"), ("Uninjured", "6 h NM72"),
      ("6 h Vehicle", "6 h NM72"),
      ("6 h Vehicle", "24 h Vehicle"), ("6 h NM72", "24 h NM72")]),
    ("Tregs", "Tregs", "% of CD4$^+$", []),
    ("CD25$^+$CD127$^-$", "CD25+CD127-", "% of CD4$^+$", []),
    ("MerTK on red pulp macrophages", "MerTK Median (M1 Like)", "median fluorescence", []),
]

TITLE = "Splenic immune response to SCI, and the effect of NM72"

# Footnote lines. Plain strings, so they can be edited here or in the SVG.
NOTE_EXCLUSIONS = "489 and 497 excluded on live-leukocyte yield."
NOTE_483_UNCORRECTED = ("C: animal 483's CD3 gate was tightened after initial analysis (its "
                        "neutrophils 3.87% to 5.67%); with the original gate, 6 h vehicle vs NM72 "
                        "is p = 0.26. Mann-Whitney p = 0.11, exact permutation p = 0.09.")
NOTE_483_CORRECTED = ("C: animal 483's CD3 gate was tightened after initial analysis (its "
                      "neutrophils 3.87% to 5.67%), moving the uncorrected 6 h vehicle-vs-NM72 p "
                      "from 0.26 to 0.048; neither survives correction.")
NOTE_TUKEY = ("Tukey assumes equal SDs across groups: Brown-Forsythe p = 0.30-0.79 for A-E, "
              "0.048 for F.")
NOTE_OMITTED = ("CD3 and CD4 frequencies are omitted: those track the acquisition-quality "
                "gradient (Spearman 0.74, p=0.001) and cannot be separated from it at 6 h. D-F do "
                "not (|rho| < 0.41, p > 0.12), so they are shown.")

XS = {("Uninjured" if tp == "Uninjured" else f"{tp} {tr}"): x for tp, tr, x in sc.GROUPS}


def _stats(long: pd.DataFrame, pop: str) -> dict:
    d = long[long.population == pop]
    return ph.pairwise({g: d[d.group == g]["value"].to_numpy(float) for g in ORDER})


def _p(res: dict, a: str, b: str, col: str) -> float:
    for r in res["rows"]:
        if {r["group_a"], r["group_b"]} == {a, b}:
            return r[col]
    return np.nan


def _bracket(ax, x1, x2, y, text, color, drop=0.02):
    h = ax.get_ylim()[1] * drop
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color=color, linewidth=1.0,
            clip_on=False, zorder=6)
    ax.text((x1 + x2) / 2, y + h * 1.15, text, ha="center", va="bottom", fontsize=8.5,
            color=color, zorder=6)


def draw(ax, long: pd.DataFrame, res: dict, label: str, pop: str, ylab: str,
         brackets: list, c: dict, col: str, corrected: bool) -> None:
    d = long[long.population == pop]
    vals = {g: d[d.group == g]["value"].dropna().to_numpy(float) for g in XS}
    present = {g: v for g, v in vals.items() if v.size}
    ax.set_facecolor(c["surface"])

    un = present.get("Uninjured", np.array([]))
    if un.size > 1:
        m, sem = sc.mean_sem(un)
        ax.axhspan(m - sem, m + sem, color=c["band"], alpha=0.13, zorder=0, linewidth=0)
        ax.axhline(m, color=c["band"], linewidth=0.9, alpha=0.55, zorder=0)
    elif un.size == 1:
        ax.axhline(float(un[0]), color=c["band"], linewidth=0.9, alpha=0.4,
                   linestyle=":", zorder=0)
        ax.text(0.99, 0.90, "uninjured n=1: no baseline comparison possible",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color=c["flag"])

    ymax = max(v.max() for v in present.values())
    top = ymax * (1.52 if brackets else 1.28)
    ax.set_ylim(0, top)

    for g, v in present.items():
        x = XS[g]
        arm = "Uninjured" if g == "Uninjured" else ("NM72" if "NM72" in g else "Vehicle")
        hue = c[arm]
        m, s = sc.mean_sem(v)
        ax.errorbar(x, m, yerr=s, fmt="none", ecolor=hue, elinewidth=1.5, capsize=5,
                    capthick=1.5, zorder=2)
        ax.hlines(m, x - 0.24, x + 0.24, color=hue, linewidth=2.6, zorder=3)
        ax.scatter(x + sc.jitter(v.size, 0.11), v, s=58, facecolor=hue, alpha=0.92,
                   edgecolor=c["surface"], linewidth=1.3, zorder=4)
        ax.text(x, top * 0.022, f"n={v.size}", ha="center", va="bottom", fontsize=8,
                color=c["text_muted"])

    # Brackets sit just above the points they span, but two that overlap
    # horizontally must not share a height or their labels collide.
    placed: list[tuple[float, float, float]] = []
    step = ymax * 0.145
    for a, b in brackets:
        if a not in present or b not in present:
            continue
        pv = _p(res, a, b, col)
        if np.isnan(pv):
            continue
        x1, x2 = sorted((XS[a], XS[b]))
        y = max(present[a].max(), present[b].max()) + ymax * 0.09
        for px1, px2, py in placed:
            if x1 <= px2 and px1 <= x2 and y < py + step:
                y = py + step
        txt = f"p = {pv:.3f}" if pv >= 0.001 else "p < 0.001"
        _bracket(ax, x1, x2, y, txt, c["text"] if pv < 0.05 else c["text_muted"])
        placed.append((x1, x2, y))
    if placed:
        top = max(top, max(y for _a, _b, y in placed) + ymax * 0.16)
    if not brackets and present:
        any_sig = any(r[col] < 0.05 for r in res["rows"])
        msg = ("a comparison reaches p < 0.05; see CSV" if any_sig else
               "no comparison reaches " + ("adjusted " if corrected else "") + "p < 0.05")
        ax.text(0.99, 0.97, msg, transform=ax.transAxes, ha="right", va="top",
                fontsize=8.5, color=c["text_muted"])

    ax.set_xlim(-0.8, 5.1)
    ax.set_ylim(0, top)
    ax.set_xticks(list(XS.values()))
    ax.set_xticklabels(["Uninj", "Veh", "NM72", "Veh", "NM72"], color=c["text_secondary"],
                       fontsize=9.5)
    ax.set_ylabel(ylab, color=c["text"], fontsize=10)
    ax.yaxis.grid(True, color=c["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(c["grid"])
    ax.tick_params(colors=c["text_secondary"], length=4, width=0.8, labelsize=9.5)
    for tp, (lo, hi) in sc.BRACKETS.items():
        ax.annotate("", xy=(lo, -0.10), xytext=(hi, -0.10),
                    xycoords=("data", "axes fraction"), textcoords=("data", "axes fraction"),
                    arrowprops=dict(arrowstyle="-", color=c["text_muted"], linewidth=0.9))
        ax.text((lo + hi) / 2, -0.15, tp, ha="center", va="top",
                transform=ax.get_xaxis_transform(), fontsize=9.5, color=c["text_secondary"])
    ax.set_title(label, fontsize=11, fontweight="bold", color=c["text"], loc="left", pad=8)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Presentation figure set.")
    p.add_argument("--correction", choices=list(ph.METHODS), default="dunnett-t3",
                   help="Multiple-comparison procedure for the p-values drawn.")
    p.add_argument("--outdir", default="outputs/final")
    p.add_argument("--dark", action="store_true")
    ns = p.parse_args(argv if argv is not None else sys.argv[1:])

    col, method = ph.METHODS[ns.correction]
    corrected = ns.correction != "none"
    long = pd.read_csv(LONG)
    c = sc.palette(ns.dark)
    outdir = Path(ns.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = {pop: _stats(long, pop) for _l, pop, _y, _b in PANELS}
    rows = [{"population": pop, **r, "dropped_n_lt_2": ",".join(res["dropped"])}
            for pop, res in results.items() for r in res["rows"]]
    pd.DataFrame(rows).to_csv(outdir / "posthoc_all_methods.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(17.0, 9.0), facecolor=c["surface"])
    for ax, (label, pop, ylab, br), letter in zip(axes.flatten(), PANELS, "ABCDEF"):
        draw(ax, long, results[pop], label, pop, ylab, br, c, col, corrected)
        ax.text(-0.17, 1.06, letter, transform=ax.transAxes, fontsize=15,
                fontweight="bold", color=c["text"], va="top")

    leg = fig.legend(handles=sc.legend_handles(c), loc="upper left", frameon=False,
                     fontsize=10, ncol=3, bbox_to_anchor=(0.038, 0.962),
                     handletextpad=0.4, columnspacing=1.6)
    for t in leg.get_texts():
        t.set_color(c["text_secondary"])
    fig.suptitle(TITLE, fontsize=14, fontweight="bold", color=c["text"],
                 x=0.038, ha="left", y=0.995)

    head = (f"Mean ± SEM, every animal shown. {method}"
            + (", all pairwise comparisons across the five groups; adjusted p shown, grey where "
               "p ≥ 0.05." if corrected else "; exact p, grey where p ≥ 0.05.")
            + "  ·  " + NOTE_EXCLUSIONS)
    lines = [head, NOTE_483_CORRECTED if corrected else NOTE_483_UNCORRECTED]
    if ns.correction == "tukey":
        lines.append(NOTE_TUKEY)
    lines.append(NOTE_OMITTED)
    fig.text(0.038, 0.012, "\n".join(lines), fontsize=8.5, color=c["text_muted"], ha="left")
    fig.tight_layout(rect=[0.01, 0.022 * len(lines) + 0.01, 1, 0.925], h_pad=3.6, w_pad=3.2)

    out = sc.save_figure(fig, outdir / f"NM72_summary_panels_{ns.correction}", c["surface"], dpi=300)
    plt.close(fig)
    print(f"Saved: {out} (+ .pdf, .svg) and {outdir / 'posthoc_all_methods.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
