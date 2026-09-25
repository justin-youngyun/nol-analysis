#!/usr/bin/env python
"""
The presentation figures, cut and sized for 16:9 slides.

make_final_panels.py lays the six panels out as one page. A slide needs them
in ones and twos, drawn larger, with the title, legend and footnotes left to
the slide as live text. The panels come from the same draw() call, so a slide
and the summary figure cannot disagree.

Two backup figures are drawn here too, for what the summary figure only
mentions in its footnote: which readouts track sample viability, and the
live-leukocyte exclusion.

Each figure is written as PNG, which the slide shows, and SVG, which
PowerPoint edits once you right-click the figure and choose Convert to Shape.

    python3 slide_figures.py                       # Dunnett's T3
    python3 slide_figures.py --correction tukey
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker  # noqa: F401  (FuncFormatter)
import numpy as np
import pandas as pd
from lxml import etree
from scipy import stats

import make_final_panels as fp
import plot_flow_panel as pf
import posthoc as ph
import sci_cohorts as sc

OUTDIR = Path("outputs/slides")
BY_LETTER = dict(zip("ABCDEF", fp.PANELS))

# name: (panel letters, render size in inches, grid). Rendered a little smaller
# than the slide shows them, so the text lands at a readable size.
SETS = {
    "overview": ("ABCDEF", (13.0, 6.6), (2, 3)),
    "b1a": ("AB", (10.4, 3.2), (1, 2)),
    "neutrophils": ("C", (5.9, 4.1), (1, 1)),
    "tregs": ("DE", (10.4, 3.2), (1, 2)),
    "mertk": ("F", (5.9, 4.1), (1, 1)),
}

# Readouts checked against viability for the backup slide: the two left out of
# the summary figure first, then the T cell / myeloid panel readouts it shows.
QUALITY_PANELS = [
    ("CD3$^+$ T cells", "CD3+ T cells", "% of live leukocytes"),
    ("CD4$^+$", "CD4+", "% of CD3$^+$"),
    ("Neutrophils", "Neutrophils", "% of live leukocytes"),
    ("Tregs", "Tregs", "% of CD4$^+$"),
    ("CD25$^+$CD127$^-$", "CD25+CD127-", "% of CD4$^+$"),
    ("MerTK", "MerTK Median (M1 Like)", "median fluorescence"),
]

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"


def _style(s: str | None) -> dict:
    out = {}
    for item in (s or "").split(";"):
        if ":" in item:
            k, v = item.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def office_svg(path: Path) -> None:
    """Replace every <use> with the path it points to.

    Matplotlib draws markers and ticks once and places copies with <use>.
    Inlining them makes each point a plain shape, which is what PowerPoint's
    Convert to Shape handles best.
    """
    tree = etree.parse(str(path))
    root = tree.getroot()
    defs = {el.get("id"): el for el in root.iter(f"{{{SVG_NS}}}path") if el.get("id")}
    for use in list(root.iter(f"{{{SVG_NS}}}use")):
        src = defs.get(use.get(f"{{{XLINK_NS}}}href", "").lstrip("#"))
        if src is None:
            continue
        shape = etree.Element(f"{{{SVG_NS}}}path")
        shape.set("d", src.get("d"))
        move = f"translate({use.get('x', '0')} {use.get('y', '0')})"
        shape.set("transform", f"{use.get('transform')} {move}" if use.get("transform") else move)
        # The referenced path's own style wins; the <use> supplies the rest.
        merged = {**_style(use.get("style")), **_style(src.get("style"))}
        shape.set("style", "; ".join(f"{k}: {v}" for k, v in merged.items()))
        if use.get("clip-path"):
            shape.set("clip-path", use.get("clip-path"))
        use.getparent().replace(use, shape)
    tree.write(str(path), xml_declaration=True, encoding="utf-8", standalone=False)


def _save(fig, name: str, c: dict, outdir: Path) -> Path:
    png = sc.save_figure(fig, outdir / name, c["surface"], dpi=300)
    plt.close(fig)
    office_svg(png.with_suffix(".svg"))
    return png


def panel_figure(long: pd.DataFrame, results: dict, name: str, method: str,
                 c: dict, outdir: Path) -> Path:
    letters, figsize, grid = SETS[name]
    col = ph.METHODS[method][0]
    fig, axes = plt.subplots(*grid, figsize=figsize, facecolor=c["surface"], squeeze=False)
    for ax, letter in zip(axes.flat, letters):
        label, pop, ylab, brackets = BY_LETTER[letter]
        fp.draw(ax, long, results[pop], label, pop, ylab, brackets, c, col, method != "none")
        ax.text(-0.17, 1.06, letter, transform=ax.transAxes, fontsize=15,
                fontweight="bold", color=c["text"], va="top")
    fig.tight_layout(h_pad=3.2, w_pad=3.2)
    suffix = "" if method == "dunnett-t3" else f"_{method}"
    return _save(fig, f"{name}{suffix}", c, outdir)


def quality_table() -> pd.DataFrame:
    """Per-animal viability and clean-gate-block membership, T cell / myeloid tube."""
    counts = pd.read_csv(pf.DEFAULT_COUNTS, keep_default_na=False, na_values=[""])
    counts["count"] = counts["count"].astype(int)
    counts, _notes = pf.resolve_duplicates(counts)
    wide = pf.build_table(counts)
    meta = pd.read_csv(pf.DEFAULT_META)
    sid = counts.drop_duplicates("sample").set_index("sample")["sampleID"].astype(str)
    wide["sampleID"] = wide["sample"].map(sid)
    wide = wide.merge(meta.astype({"sampleID": str})[["sampleID", "date", "btim"]],
                      on="sampleID", how="left")
    wide = wide.dropna(subset=["treatment"]).copy()
    wide["live_ok"] = wide[pf.LIVE] >= 1000
    wide["block"] = pf.detect_outlier_block(wide[wide["live_ok"]]).reindex(
        wide.index, fill_value=False)
    wide["viability"] = 100 * wide[pf.LIVE].astype(float) / wide["Cells/Single Cells"].astype(float)
    wide["live_events"] = wide[pf.LIVE].astype(float)
    wide["cd4cd8"] = 100 * (wide[pf.CD4].astype(float) + wide[f"{pf.CD3}/CD8+"].astype(float)) \
        / wide[pf.CD3].astype(float)
    return wide.set_index("animal")[["timepoint", "treatment", "live_ok", "block", "btim",
                                     "viability", "live_events", "cd4cd8"]]


def quality_cd4cd8() -> pd.Series:
    """CD4+ plus CD8+ as % of CD3+, the gating-quality readout, for analysed animals."""
    q = quality_table()
    return q.loc[q["live_ok"], "cd4cd8"]


def quality_figure(long: pd.DataFrame, q: pd.DataFrame, c: dict, outdir: Path) -> Path:
    """Each readout against viability. The ones that track it cannot be read at 6 h."""
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.9), facecolor=c["surface"])
    ok = q[q["live_ok"]]
    for ax, (title, pop, ylab) in zip(axes.flat, QUALITY_PANELS):
        y = long[long.population == pop].set_index("animal")["value"]
        d = ok.join(y.rename("value"), how="inner").dropna(subset=["value"])
        rho, p = stats.spearmanr(d["viability"], d["value"])
        tracks = p < 0.05
        ax.set_facecolor(c["surface"])
        for _a, r in d.iterrows():
            ax.scatter(r["viability"], r["value"], s=46, facecolor=c[r["treatment"]],
                       edgecolor=c["text"] if r["block"] else c["surface"],
                       linewidth=1.5 if r["block"] else 0.9, zorder=3)
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color=c["text"],
                     pad=17)
        ax.text(0, 1.03, f"Spearman ρ = {rho:+.2f}, p = {p:.3f}", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=9,
                color=c["flag"] if tracks else c["text_muted"],
                fontweight="bold" if tracks else "normal")
        ax.set_ylabel(ylab, fontsize=9, color=c["text_secondary"])
        ax.set_xlim(0, 75)
        ax.set_ylim(bottom=0)
        ax.yaxis.grid(True, color=c["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(c["grid"])
        ax.tick_params(colors=c["text_secondary"], length=3, width=0.8, labelsize=8.5)
        if pop.startswith("MerTK"):
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
                lambda v, _p: f"{v / 1000:.0f}k" if v else "0"))
    fig.supxlabel("Viability: live leukocytes, % of singlets", fontsize=9.5,
                  color=c["text_secondary"], y=0.075)
    handles = sc.legend_handles(c) + [plt.Line2D(
        [], [], marker="o", linestyle="", markersize=8, markerfacecolor=c["surface"],
        markeredgecolor=c["text"], markeredgewidth=1.5, label="run in the clean-gate block")]
    leg = fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9,
                     bbox_to_anchor=(0.5, -0.01), handletextpad=0.3, columnspacing=1.2)
    for t in leg.get_texts():
        t.set_color(c["text_secondary"])
    fig.tight_layout(rect=[0, 0.09, 1, 1], h_pad=1.6, w_pad=1.4)
    return _save(fig, "quality_vs_viability", c, outdir)


def exclusion_figure(q: pd.DataFrame, c: dict, outdir: Path, threshold: int = 1000) -> Path:
    d = q.sort_values("live_events")
    fig, ax = plt.subplots(figsize=(7.4, 4.3), facecolor=c["surface"])
    ax.set_facecolor(c["surface"])
    x = np.arange(len(d))
    failed = (d["live_events"] < threshold).to_numpy()
    colors = [matplotlib.colors.to_rgba(c[t], 0.28 if f else 0.9)
              for t, f in zip(d["treatment"], failed)]
    ax.bar(x, d["live_events"], color=colors, width=0.68, zorder=2)
    ax.axhline(threshold, color=c["flag"], linewidth=1.2, linestyle="--", zorder=3)
    # Labelled outside the plot, at the line's right end, so it covers no bar.
    ax.text(1.01, threshold, "1,000", transform=ax.get_yaxis_transform(), ha="left",
            va="center", fontsize=9.5, color=c["flag"], fontweight="bold")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(a) for a in d.index], fontsize=9, rotation=90)
    ax.set_ylabel("Live leukocyte events (log scale)", color=c["text"], fontsize=10)
    ax.set_xlabel("Animal", color=c["text_secondary"], fontsize=10)
    ax.yaxis.grid(True, color=c["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(c["grid"])
    ax.tick_params(colors=c["text_secondary"], length=4, width=0.8)
    # After tick_params, which would reset them: excluded animals in the flag colour.
    for lab, f in zip(ax.get_xticklabels(), failed):
        lab.set_color(c["flag"] if f else c["text_secondary"])
        lab.set_fontweight("bold" if f else "normal")
    fig.tight_layout()
    return _save(fig, "exclusions", c, outdir)


def render(methods=("dunnett-t3", "tukey"), outdir: Path = OUTDIR) -> dict[str, Path]:
    """Draw every slide figure; returns {name: png path}."""
    outdir.mkdir(parents=True, exist_ok=True)
    c = sc.palette(False)
    long = pd.read_csv(fp.LONG)
    results = {pop: fp._stats(long, pop) for _l, pop, _y, _b in fp.PANELS}
    out = {}
    for method in methods:
        names = SETS if method == "dunnett-t3" else ["overview"]
        for name in names:
            png = panel_figure(long, results, name, method, c, outdir)
            out[png.stem] = png
    q = quality_table()
    out["quality_vs_viability"] = quality_figure(long, q, c, outdir)
    out["exclusions"] = exclusion_figure(q, c, outdir)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Figures sized for 16:9 slides.")
    p.add_argument("--correction", choices=list(ph.METHODS), default=None,
                   help="Draw the panel figures with one procedure only.")
    p.add_argument("--outdir", default=str(OUTDIR))
    ns = p.parse_args(argv if argv is not None else sys.argv[1:])
    methods = (ns.correction,) if ns.correction else ("dunnett-t3", "tukey")
    for name, png in render(methods, Path(ns.outdir)).items():
        print(f"Saved: {png} (+ .svg)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
