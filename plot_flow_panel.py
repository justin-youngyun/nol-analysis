#!/usr/bin/env python
"""
T cell and myeloid splenocyte populations after SCI, by cohort.

Reads a FlowJo workspace (.wsp) directly -- the workspace caches an event count
on every gate node, so the whole hierarchy can be recovered without the .fcs
files -- and draws each population against the cohort split.

Counts are cached to a CSV so the figures rebuild without the workspace.

Two things the workspace makes visible and a screenshot of a stats table would
not:

  * the myeloid branch hangs off a FlowJo NotNode (CD3+-, i.e. CD3-negative)
    rather than a Population node, so it is easy to miss when scraping;
  * per-sample event counts, which is what the live-leukocyte exclusion below
    is judged on.
"""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import sci_cohorts as sc
import plot_b1a_splenocytes as b1a
import posthoc as ph

DEFAULT_COUNTS = Path("data/sci_flow_counts.csv")
DEFAULT_META = Path("data/sci_flow_samples.csv")
DEFAULT_MFI = Path("data/sci_flow_mfi.csv")
SPL_GROUP = "Spl"
NODE_TAGS = {"Population", "NotNode", "OrNode", "AndNode"}

LIVE = "Cells/Single Cells/Live Leukocytes"
CD3 = f"{LIVE}/CD3+"
CD4 = f"{CD3}/CD4+"
CD3N = f"{LIVE}/CD3+-"
MAC = f"{CD3N}/CD11b+F4_80+"
LY6G = f"{MAC}/Eosinophils-/Ly6G-"
GRAN = f"{CD3N}/CD11b+F4_80-"
RPM = f"{CD3N}/Red pulp Macs"

# (title, numerator path, denominator path, denominator label)
T_PANELS = [
    ("CD3$^+$ T cells", CD3, LIVE, "% of live leukocytes"),
    ("CD4$^+$", CD4, CD3, "% of CD3$^+$"),
    ("CD8$^+$", f"{CD3}/CD8+", CD3, "% of CD3$^+$"),
    ("CD25$^+$CD127$^-$", f"{CD4}/CD25+CD127-", CD4, "% of CD4$^+$"),
    ("Tregs", f"{CD4}/CD25+CD127-/Tregs", CD4, "% of CD4$^+$"),
]

# CD3+- is the exact complement of CD3+, so it is a denominator here, not a panel.
M_PANELS = [
    ("CD11b$^+$F4/80$^+$", MAC, LIVE, "% of live leukocytes"),
    ("CD11b$^+$F4/80$^-$", GRAN, LIVE, "% of live leukocytes"),
    ("Red pulp macrophages", RPM, LIVE, "% of live leukocytes"),
    ("Neutrophils", f"{GRAN}/Neutrophils", LIVE, "% of live leukocytes"),
    ("Monocytes", f"{GRAN}/Monocytes", LIVE, "% of live leukocytes"),
    ("Eosinophils", f"{MAC}/Eosinophils", MAC, "% of CD11b$^+$F4/80$^+$"),
    ("Ly6G$^-$", LY6G, MAC, "% of CD11b$^+$F4/80$^+$"),
    ("M1-like (Ly6G$^-$)", f"{LY6G}/M1 Like", LY6G, "% of Ly6G$^-$"),
    ("M2-like (Ly6G$^-$)", f"{LY6G}/M2 Like", LY6G, "% of Ly6G$^-$"),
    ("M1-like (red pulp)", f"{RPM}/M1 Like", RPM, "% of red pulp macs"),
    ("M2-like (red pulp)", f"{RPM}/M2 Like", RPM, "% of red pulp macs"),
]

# The B cell panel is a separate stain on the same animals, merged by animal ID.
B1A_PANELS = [
    ("B-1a  (IgM$^+$)", "b1a_igm_pos", "% of B cells  ·  B cell panel"),
    ("IgM$^-$", "b1a_igm_neg", "% of B cells  ·  B cell panel"),
]


def add_b1a(wide: pd.DataFrame) -> list:
    """Merge the B cell panel in by animal, and report any coverage mismatch.

    Its own exclusion (489) already matches this panel's; 497 differs but is
    uninjured, so the two sets coincide for a drug-vs-vehicle read.
    """
    src = b1a.build_frame().set_index("animal")
    keep = src[~src["excluded"]]
    wide["b1a_igm_pos"] = wide["animal"].map(keep["igm_pos_pct"])
    wide["b1a_igm_neg"] = wide["animal"].map(keep["igm_neg_pct"])
    missing = sorted(int(a) for a in wide.loc[wide["b1a_igm_pos"].isna(), "animal"])
    if missing:
        print(f"No B cell panel data for: {missing}")
    return B1A_PANELS


# Every comparison worth reporting, tagged by what it asks.
CONTRAST_SET = [
    ("Uninjured", "Uninjured", "6 h", "Vehicle", "injury"),
    ("Uninjured", "Uninjured", "6 h", "NM72", "injury"),
    ("Uninjured", "Uninjured", "24 h", "Vehicle", "injury"),
    ("Uninjured", "Uninjured", "24 h", "NM72", "injury"),
    ("6 h", "Vehicle", "24 h", "Vehicle", "time"),
    ("6 h", "NM72", "24 h", "NM72", "time"),
    ("6 h", "Vehicle", "6 h", "NM72", "drug"),
    ("24 h", "Vehicle", "24 h", "NM72", "drug"),
]

# Populations that belong on one graph together. Anything not named here still
# gets its own single-population figure.
FAMILIES: dict[str, list[str]] = {
    "T cells and Tregs": ["CD3$^+$ T cells", "CD4$^+$", "CD8$^+$", "CD4:CD8 ratio",
                          "CD25$^+$CD127$^-$", "Tregs"],
    "B cells": ["B-1a  (IgM$^+$)", "IgM$^-$"],
    "Macrophages": ["CD11b$^+$F4/80$^+$", "CD11b$^+$F4/80$^-$", "Red pulp macrophages"],
    "Granulocytes and monocytes": ["Neutrophils", "Monocytes"],
    "MerTK on red pulp macrophages": ["MerTK Median (M1 Like)", "MerTK Median (M2 Like)"],
    "Macrophage subsets (low events)": ["Eosinophils", "Ly6G$^-$", "M1-like (Ly6G$^-$)",
                                        "M2-like (Ly6G$^-$)", "M1-like (red pulp)",
                                        "M2-like (red pulp)"],
}


def _slug(label: str) -> str:
    """A filename from a panel label, with the TeX markup taken back out."""
    t = (label.replace("$^+$", "pos").replace("$^-$", "neg").replace("$^{lo}$", "lo")
              .replace("/", "-").replace(":", "-").replace(" ", "_"))
    return re.sub(r"[^A-Za-z0-9_.-]", "", t).strip("_")


def _plain(label: str) -> str:
    """The label as a person would type it, for CSV headers."""
    return (label.replace("$^+$", "+").replace("$^-$", "-").replace("$^{lo}$", "lo")
                 .replace("  ", " ").strip())


# A frequency computed from a handful of events is noise, not a measurement.
MIN_NUMERATOR = 20
MIN_DENOMINATOR = 100


# ---------------------------------------------------------------------------
# Reading the workspace
# ---------------------------------------------------------------------------

def parse_wsp(wsp_path: Path, group: str = SPL_GROUP) -> pd.DataFrame:
    """Tidy (sample, gate path, count) table for one workspace group.

    FlowJo hangs boolean-NOT gates off NotNode rather than Population, so the
    walk has to accept both or whole branches vanish silently.
    """
    root = ET.parse(wsp_path).getroot()
    wanted = {sr.get("sampleID")
              for g in root.iter("GroupNode") if g.get("name") == group
              for sr in g.iter("SampleRef")}
    if not wanted:
        raise SystemExit(f"No group named {group!r} in {wsp_path}")

    rows = []

    def walk(node, path, sample, sid):
        subs = node.find("Subpopulations")
        if subs is None:
            return
        for child in subs:
            if child.tag not in NODE_TAGS:
                continue
            p = path + [child.get("name")]
            rows.append({"sample": sample, "sampleID": sid, "path": "/".join(p),
                         "pop": child.get("name"), "kind": child.tag,
                         "depth": len(p), "count": int(child.get("count") or 0),
                         "parent_path": "/".join(path)})
            walk(child, p, sample, sid)

    for sn in root.iter("SampleNode"):
        sid = sn.get("sampleID")
        if sid not in wanted:
            continue
        sample = sn.get("name")
        rows.append({"sample": sample, "sampleID": sid, "path": "", "pop": "(all events)",
                     "kind": "SampleNode", "depth": 0,
                     "count": int(sn.get("count") or 0), "parent_path": ""})
        walk(sn, [], sample, sid)

    return pd.DataFrame(rows)


def parse_metadata(wsp_path: Path, group: str = SPL_GROUP) -> pd.DataFrame:
    """Acquisition date and start time per sample, from the cached FCS keywords."""
    root = ET.parse(wsp_path).getroot()
    wanted = {sr.get("sampleID")
              for g in root.iter("GroupNode") if g.get("name") == group
              for sr in g.iter("SampleRef")}
    rows = []
    for smp in root.iter("Sample"):
        sn = smp.find("SampleNode")
        if sn is None or sn.get("sampleID") not in wanted:
            continue
        kw = {k.get("name"): k.get("value") for k in smp.iter("Keyword")}
        rows.append({"sample": sn.get("name"), "sampleID": str(sn.get("sampleID")),
                     "date": kw.get("$DATE"), "btim": kw.get("$BTIM"),
                     "flowrate": kw.get("$FLOWRATE"), "cytometer": kw.get("$CYT")})
    return pd.DataFrame(rows)


def detect_outlier_block(wide: pd.DataFrame) -> pd.Series:
    """Flag a technical cluster using CD4+CD8 as a fraction of CD3+.

    In spleen, CD4 and CD8 should account for most CD3+ events. A sample where
    they account for half of them has a CD3 gate holding something that is not
    a T cell, so the sum is a gating-quality readout that does not depend on
    the biology under test. Split at the largest gap rather than a fixed cut.
    """
    total = (wide[CD4].astype(float) + wide[f"{CD3}/CD8+"].astype(float))
    frac = 100.0 * total / wide[CD3].astype(float).where(wide[CD3].astype(float) > 0)
    ok = frac.dropna().sort_values()
    if len(ok) < 4:
        return pd.Series(False, index=wide.index)
    gaps = ok.diff()
    if gaps.max() < 5:
        return pd.Series(False, index=wide.index)
    seed = frac >= ok.loc[gaps.idxmax()]

    # A technical problem develops and clears over a run rather than hitting a
    # fixed set of tubes, so samples that sit between flagged ones belong to the
    # same episode even when their own value lands mid-gap. Widen the seed to the
    # contiguous acquisition-order span it covers.
    if "btim" in wide.columns and wide["btim"].notna().any():
        order = wide.sort_values(["date", "btim"]).index
        pos = {idx: i for i, idx in enumerate(order)}
        flagged = [pos[i] for i in wide.index[seed.fillna(False)]]
        if flagged:
            lo, hi = min(flagged), max(flagged)
            span = {idx for idx, i in pos.items() if lo <= i <= hi}
            return pd.Series([i in span for i in wide.index], index=wide.index)
    return seed.fillna(False)


def parse_mfi(wsp_path: Path, group: str = SPL_GROUP) -> pd.DataFrame:
    """Non-frequency statistics (MFI, medians) stored on gate nodes.

    FlowJo hangs these off the node's Subpopulations element rather than the
    node itself, beside the child gates, so a walk that only reads gate nodes
    steps straight past them.
    """
    root = ET.parse(wsp_path).getroot()
    wanted = {sr.get("sampleID")
              for g in root.iter("GroupNode") if g.get("name") == group
              for sr in g.iter("SampleRef")}
    detector = {}
    for smp in root.iter("Sample"):
        sn = smp.find("SampleNode")
        if sn is None or sn.get("sampleID") not in wanted:
            continue
        kw = {k.get("name"): k.get("value") for k in smp.iter("Keyword")}
        for key, val in kw.items():
            m = re.fullmatch(r"\$P(\d+)N", key)
            if m and kw.get(f"$P{m.group(1)}S"):
                detector[val] = kw[f"$P{m.group(1)}S"]
        break

    rows = []

    def walk(node, path, sample, sid):
        subs = node.find("Subpopulations")
        if subs is None:
            return
        for child in subs:
            if child.tag == "Statistic" and child.get("name") != "fj.stat.freqof":
                det = (child.get("id") or "").replace("Comp-", "")
                rows.append({"sample": sample, "sampleID": sid, "path": "/".join(path),
                             "stat": child.get("name"), "detector": det,
                             "marker": detector.get(det, det),
                             "value": float(child.get("value") or "nan")})
            if child.tag in NODE_TAGS:
                walk(child, path + [child.get("name")], sample, sid)

    for sn in root.iter("SampleNode"):
        if sn.get("sampleID") in wanted:
            walk(sn, [], sn.get("name"), sn.get("sampleID"))
    return pd.DataFrame(rows)


def resolve_duplicates(counts: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """A sample imported twice keeps the entry with more events."""
    notes = []
    live = counts[counts.path == LIVE].set_index("sampleID")["count"].to_dict()
    keep = {}
    for sample, grp in counts.groupby("sample"):
        sids = sorted(grp["sampleID"].unique())
        if len(sids) > 1:
            best = max(sids, key=lambda s: live.get(s, 0))
            notes.append(f"{sample}: {len(sids)} entries in the workspace "
                         f"(live leukocytes {', '.join(str(live.get(s, 0)) for s in sids)}); "
                         f"kept the largest")
            keep[sample] = best
        else:
            keep[sample] = sids[0]
    return counts[counts["sampleID"].isin(keep.values())].copy(), notes


def animal_id(sample: str) -> int | None:
    stem = Path(str(sample)).stem
    return int(stem) if stem.isdigit() else None


def build_table(counts: pd.DataFrame) -> pd.DataFrame:
    """Wide per-sample table: one column of counts per gate path, plus cohort."""
    counts = counts.copy()
    counts["animal"] = counts["sample"].map(animal_id)
    wide = counts.pivot_table(index=["sample", "animal"], columns="path",
                              values="count", aggfunc="first").reset_index()
    wide.columns.name = None
    wide["timepoint"] = wide["animal"].map(lambda a: sc.COHORTS.get(a, (None, None))[0])
    wide["treatment"] = wide["animal"].map(lambda a: sc.COHORTS.get(a, (None, None))[1])
    return wide


def add_frequency(wide: pd.DataFrame, key: str, num: str, den: str) -> pd.DataFrame:
    """Percentage of one gate in another, blanked where the counts are too thin."""
    n = wide[num].astype(float)
    d = wide[den].astype(float)
    pct = 100.0 * n / d.where(d > 0)
    wide[key] = pct.where((d >= MIN_DENOMINATOR) & ((n >= MIN_NUMERATOR) | (n == 0)))
    wide[key + "__n"] = n
    wide[key + "__d"] = d
    return wide


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _footnote(fig, c, notes: list[str]) -> None:
    fig.text(0.05, 0.008, "  ·  ".join(notes), fontsize=8, color=c["text_muted"], ha="left")


def plot_grid(data: pd.DataFrame, panels: list, out_path: Path, title: str,
              ncols: int, dark: bool = False, notes: list[str] | None = None,
              drug_only: bool = False) -> Path:
    c = sc.palette(dark)
    gkw = dict(groups=sc.DRUG_GROUPS, brackets=sc.DRUG_BRACKETS, ticks=sc.DRUG_TICKS,
               baseline=False) if drug_only else {}
    treatments = ("Vehicle", "NM72") if drug_only else ("Uninjured", "Vehicle", "NM72")
    nrows = int(np.ceil(len(panels) / ncols))
    # extra height for the title band, so a long title cannot run into the legend
    head = 0.75
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.55 * ncols, 3.15 * nrows + head),
                             facecolor=c["surface"], squeeze=False)
    h_total = 3.15 * nrows + head
    flat = axes.flatten()

    for i, (label, key, denom_label) in enumerate(panels):
        ax = flat[i]
        bottom = i >= len(panels) - ncols
        missing = int(data[key].isna().sum())
        thin = missing > len(data) / 3
        sc.draw_panel(ax, data, key, c, panel_title=label,
                      subtitle=(f"{denom_label}  ·  too few events in {missing}/{len(data)}"
                                if thin else denom_label),
                      compact=True, show_ticklabels=bottom, show_brackets=bottom,
                      show_n=True,
                      title_color=c["flag"] if thin else None,
                      subtitle_color=c["flag"] if thin else None, **gkw)
    for ax in flat[len(panels):]:
        ax.set_visible(False)

    leg = fig.legend(handles=sc.legend_handles(c, treatments), loc="upper left", frameon=False,
                     fontsize=9.5, ncol=3, bbox_to_anchor=(0.05, 1 - 0.42 / h_total),
                     handletextpad=0.4, columnspacing=1.4)
    for t in leg.get_texts():
        t.set_color(c["text_secondary"])
    fig.suptitle(title, fontsize=13, fontweight="bold", color=c["text"],
                 x=0.05, ha="left", y=1 - 0.10 / h_total)
    _footnote(fig, c, notes or [])
    fig.tight_layout(rect=[0.005, 0.035, 1, 1 - (head - 0.05) / h_total],
                     h_pad=2.6, w_pad=1.8)
    sc.save_figure(fig, out_path, c["surface"])
    plt.close(fig)
    return out_path


def plot_qc(wide: pd.DataFrame, threshold: int, out_path: Path, dark: bool = False) -> Path:
    """Live-leukocyte yield per animal, log scale, with the exclusion line."""
    c = sc.palette(dark)
    d = wide.dropna(subset=["treatment"]).sort_values(LIVE)
    fig, ax = plt.subplots(figsize=(9.5, 4.6), facecolor=c["surface"])
    ax.set_facecolor(c["surface"])

    x = np.arange(len(d))
    failed = (d[LIVE] < threshold).to_numpy()
    colors = [matplotlib.colors.to_rgba(c[t], 0.3 if f else 0.9)
              for t, f in zip(d["treatment"], failed)]
    ax.bar(x, d[LIVE], color=colors, width=0.68, zorder=2)
    ax.axhline(threshold, color=c["flag"], linewidth=1.2, linestyle="--", zorder=3)
    ax.text(len(d) - 0.4, threshold * 1.18, f"exclusion threshold = {threshold:,}",
            ha="right", va="bottom", fontsize=9, color=c["flag"])

    for xi, (_, r) in zip(x, d.iterrows()):
        if r[LIVE] < threshold:
            ax.text(xi, r[LIVE] * 1.35, "excluded", ha="center", va="bottom",
                    fontsize=8, color=c["flag"], rotation=90)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(a)) for a in d["animal"]], color=c["text_secondary"],
                       fontsize=9)
    ax.set_ylabel("Live leukocyte events", color=c["text"], fontsize=11)
    ax.set_xlabel("Animal", color=c["text_secondary"], fontsize=10)
    ax.yaxis.grid(True, color=c["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(c["grid"])
    ax.tick_params(colors=c["text_secondary"], length=4, width=0.8)

    leg = ax.legend(handles=sc.legend_handles(c), loc="upper left", frameon=False,
                    fontsize=9.5, ncol=3, handletextpad=0.4, columnspacing=1.4)
    for t in leg.get_texts():
        t.set_color(c["text_secondary"])
    fig.suptitle("Live leukocyte yield per animal", fontsize=13, fontweight="bold",
                 color=c["text"], x=0.06, ha="left", y=0.99)
    ax.set_title("log scale  ·  bars below the line are dropped from every population",
                 fontsize=9, color=c["text_secondary"], loc="left", pad=10)
    fig.tight_layout(rect=[0.01, 0.02, 1, 0.94])
    sc.save_figure(fig, out_path, c["surface"])
    plt.close(fig)
    return out_path


def plot_acquisition_qc(wide: pd.DataFrame, out_path: Path, dark: bool = False) -> Path:
    """Gating- and sample-quality readouts against the order the samples were run in.

    Drawn because the cohort effect and the sample quality are not independent
    here. Run order is the x axis only because it is the axis a whole-session
    problem would show up on; what the panels actually compare is quality, and a
    cohort group whose samples all sit at one end of that range cannot be
    compared with one at the other end. Acquisition settings that differ between
    tubes are marked, since those are a cause rather than a symptom.
    """
    c = sc.palette(dark)
    d = wide.dropna(subset=["btim"]).sort_values(["date", "btim"]).reset_index(drop=True)
    cd3 = d[CD3].astype(float)
    metrics = [
        ("CD4$^+$ + CD8$^+$\n(% of CD3$^+$)",
         100 * (d[CD4].astype(float) + d[f"{CD3}/CD8+"].astype(float)) / cd3),
        ("Live leukocytes\n(% of singlets)",
         100 * d[LIVE].astype(float) / d["Cells/Single Cells"].astype(float)),
        ("Singlets (% of cells)\nscatter only, no antibody",
         100 * d["Cells/Single Cells"].astype(float) / d["Cells"].astype(float)),
        ("CD3$^+$\n(% of live)", 100 * cd3 / d[LIVE].astype(float)),
    ]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(10.5, 10.4), sharex=True,
                             facecolor=c["surface"])
    x = np.arange(len(d))
    flagged = d["tech_block"].to_numpy()
    lo, hi = (np.where(flagged)[0].min() - 0.5, np.where(flagged)[0].max() + 0.5) \
        if flagged.any() else (None, None)

    for ax, (label, series) in zip(axes, metrics):
        ax.set_facecolor(c["surface"])
        if lo is not None:
            ax.axvspan(lo, hi, color=c["flag"], alpha=0.10, zorder=0, linewidth=0)
        for xi, val, tr in zip(x, series, d["treatment"]):
            ax.scatter(xi, val, s=70, facecolor=c[tr], edgecolor=c["surface"],
                       linewidth=1.3, zorder=3)
        ax.plot(x, series, color=c["text_muted"], linewidth=0.9, alpha=0.5, zorder=2)
        ax.set_ylabel(label, color=c["text"], fontsize=9.5)
        ax.yaxis.grid(True, color=c["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(c["grid"])
        ax.tick_params(colors=c["text_secondary"], length=4, width=0.8, labelsize=9)

    if lo is not None:
        axes[0].text((lo + hi) / 2, 1.04, "high-quality block",
                     transform=axes[0].get_xaxis_transform(), ha="center", va="bottom",
                     fontsize=9.5, color=c["flag"], fontweight="bold")
    rates = d["flowrate"].fillna("")
    odd = rates[rates != ""].value_counts()
    minority = odd.index[-1] if len(odd) > 1 else None
    if minority is not None:
        for ax in axes:
            for xi, r in zip(x, rates):
                if r == minority:
                    ax.axvline(xi, color=c["flag"], linewidth=8, alpha=0.10, zorder=0)
        axes[0].text(0.995, 1.04, f"vertical bars: flow rate = {minority} "
                     f"({int(odd.iloc[-1])} tubes; all others {odd.index[0]})",
                     transform=axes[0].transAxes, ha="right", va="bottom",
                     fontsize=8.5, color=c["flag"])

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([f"{int(a)}\n{t[:5]}" for a, t in zip(d["animal"], d["btim"])],
                             fontsize=8, color=c["text_secondary"])
    axes[-1].set_xlabel("Animal, in acquisition order", color=c["text_secondary"], fontsize=10)

    leg = fig.legend(handles=sc.legend_handles(c), loc="upper right", frameon=False,
                     fontsize=9.5, ncol=3, bbox_to_anchor=(0.995, 0.997),
                     handletextpad=0.4, columnspacing=1.4)
    for t in leg.get_texts():
        t.set_color(c["text_secondary"])
    fig.suptitle("Sample and gating quality across the run", fontsize=13, fontweight="bold",
                 color=c["text"], x=0.045, ha="left", y=0.995)
    axes[0].set_title("CD4 and CD8 should account for most CD3$^+$ events in spleen; where they "
                      "do not, the CD3 gate is holding something else.\nThat readout tracks "
                      "viability and doublet rate, so it is sample quality, not biology.",
                      fontsize=9, color=c["text_secondary"], loc="left", pad=26)
    fig.tight_layout(rect=[0.01, 0.01, 1, 0.94])
    sc.save_figure(fig, out_path, c["surface"])
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------

def _hedges_g(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Standardised NM72-minus-vehicle difference with a 95% interval.

    Effect size rather than raw units so populations on different denominators
    sit on one axis. At n=3-4 the interval is wide by construction; that width
    is the point, not a defect to hide.
    """
    na, nb = a.size, b.size
    if na < 2 or nb < 2:
        return np.nan, np.nan, np.nan
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    if sp == 0:
        return np.nan, np.nan, np.nan
    d = (np.mean(b) - np.mean(a)) / sp
    g = d * (1 - 3 / (4 * (na + nb) - 9))          # small-sample correction
    se = np.sqrt((na + nb) / (na * nb) + g ** 2 / (2 * (na + nb - 2)))
    t = stats.t.ppf(0.975, na + nb - 2)
    return g, g - t * se, g + t * se


def plot_contrast(data: pd.DataFrame, panels: list, out_path: Path, dark: bool = False,
                  confounded: set[str] | None = None) -> Path:
    """NM72 minus vehicle, per population, at each timepoint."""
    c = sc.palette(dark)
    confounded = confounded or set()
    rows = []
    for label, key, _dl in panels:
        for tp in ("6 h", "24 h"):
            veh = sc.group_values(data, tp, "Vehicle", key)
            nm = sc.group_values(data, tp, "NM72", key)
            g, lo, hi = _hedges_g(veh, nm)
            pval = (stats.ttest_ind(veh, nm, equal_var=False)[1]
                    if veh.size > 1 and nm.size > 1 else np.nan)
            rows.append({"label": label, "tp": tp, "g": g, "lo": lo, "hi": hi,
                         "p": pval, "n": f"{veh.size}v{nm.size}",
                         "bcell": label in {l for l, _k, _d in B1A_PANELS}})
    df = pd.DataFrame(rows).dropna(subset=["g"])
    labels = [l for l, _k, _d in panels if l in set(df["label"])]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 0.42 * len(labels) + 3.0),
                             sharey=True, facecolor=c["surface"])
    for ax, tp in zip(axes, ("6 h", "24 h")):
        ax.set_facecolor(c["surface"])
        bad = tp in confounded
        sub = df[df.tp == tp].set_index("label")
        y = np.arange(len(labels))
        ax.axvline(0, color=c["text_muted"], linewidth=1.0, zorder=1)
        for yi, lab in zip(y, labels):
            if lab not in sub.index:
                continue
            r = sub.loc[lab]
            dim = bad and not r["bcell"]
            hue = c["text_muted"] if dim else (c["NM72"] if r["g"] > 0 else c["Vehicle"])
            ax.plot([r["lo"], r["hi"]], [yi, yi], color=hue, linewidth=2.0,
                    alpha=0.45 if dim else 0.9, zorder=2)
            ax.scatter(r["g"], yi, s=70, facecolor=hue, edgecolor=c["surface"],
                       linewidth=1.3, alpha=0.5 if dim else 1.0, zorder=3)
            sig = "" if (np.isnan(r["p"]) or r["p"] >= 0.05) else f"  p={r['p']:.3f}"
            ax.text(1.01, yi, r["n"] + sig, transform=ax.get_yaxis_transform(),
                    va="center", ha="left", fontsize=7.5,
                    color=c["flag"] if sig else c["text_muted"],
                    fontweight="bold" if sig else "normal")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9, color=c["text_secondary"])
        ax.invert_yaxis()
        ax.xaxis.grid(True, color=c["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(c["grid"])
        ax.tick_params(colors=c["text_secondary"], length=4, width=0.8, labelsize=9)
        ax.set_xlabel("← favours vehicle      Hedges' g      favours NM72 →",
                      color=c["text_secondary"], fontsize=9)
        ax.set_title(f"{tp}" + ("   ·  grey = confounded" if bad else "   ·  quality-balanced"),
                     fontsize=11, fontweight="bold", loc="left", pad=8,
                     color=c["flag"] if bad else c["text"])

    fig.suptitle("NM72 vs vehicle, by population", fontsize=13, fontweight="bold",
                 color=c["text"], x=0.045, ha="left", y=0.995)
    nsig = int((df["p"] < 0.05).sum())
    tail = (f"{nsig} population(s) reach uncorrected Welch p<0.05 (none survives Dunnett's T3 or "
            f"Tukey), flagged at right; the g interval "
            "uses a pooled SD and can be wider than the Welch test it sits beside"
            if nsig else "no population reaches Welch p<0.05")
    fig.text(0.045, 0.012,
             "Hedges' g with 95% CI  ·  n per group at right (vehicle v NM72)  ·  " + tail
             + "  ·  grey = T cell / myeloid QC cluster; the B cell panel is a separate "
               "stain and is never greyed",
             fontsize=8, color=c["text_muted"], ha="left")
    fig.tight_layout(rect=[0.005, 0.045, 1, 0.94])
    sc.save_figure(fig, out_path, c["surface"])
    plt.close(fig)
    return out_path


def plot_single(data: pd.DataFrame, label: str, key: str, denom: str, out_path: Path,
                dark: bool = False, drug_only: bool = False) -> Path:
    """One population, one figure."""
    c = sc.palette(dark)
    gkw = dict(groups=sc.DRUG_GROUPS, brackets=sc.DRUG_BRACKETS, ticks=sc.DRUG_TICKS,
               baseline=False) if drug_only else {}
    treatments = ("Vehicle", "NM72") if drug_only else ("Uninjured", "Vehicle", "NM72")
    fig, ax = plt.subplots(figsize=(4.6, 4.4), facecolor=c["surface"])
    sc.draw_panel(ax, data, key, c, ylabel=denom, compact=False, show_ticklabels=True,
                  show_brackets=True, show_legend=False, show_n=True, **gkw)
    leg = ax.legend(handles=sc.legend_handles(c, treatments), loc="upper left",
                    frameon=False, fontsize=8.5, ncol=len(treatments),
                    handletextpad=0.35, borderaxespad=0.2, columnspacing=1.0)
    for t in leg.get_texts():
        t.set_color(c["text_secondary"])
    fig.suptitle(_plain(label), fontsize=12, fontweight="bold", color=c["text"],
                 x=0.02, ha="left", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    sc.save_figure(fig, out_path, c["surface"])
    plt.close(fig)
    return out_path


def plot_timecourse(data: pd.DataFrame, panels: list, out_path: Path, title: str,
                    ncols: int, dark: bool = False, notes: list[str] | None = None) -> Path:
    """Uninjured -> 6 h -> 24 h, one line per arm.

    The question this answers is whether a perturbation at 6 h has come back to
    the uninjured level by 24 h, so the baseline is drawn as the shared point
    both arms start from rather than as a separate column.
    """
    c = sc.palette(dark)
    nrows = int(np.ceil(len(panels) / ncols))
    head = 0.75
    h_total = 3.15 * nrows + head
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.55 * ncols, h_total),
                             facecolor=c["surface"], squeeze=False)
    flat = axes.flatten()
    XS = {"Uninjured": 0.0, "6 h": 1.0, "24 h": 2.0}

    for i, (label, key, denom) in enumerate(panels):
        ax = flat[i]
        ax.set_facecolor(c["surface"])
        un = sc.group_values(data, "Uninjured", "Uninjured", key)
        u_mean, u_sem = sc.mean_sem(un)
        if un.size:
            ax.axhspan(u_mean - u_sem, u_mean + u_sem, color=c["band"], alpha=0.13,
                       zorder=0, linewidth=0)
            ax.axhline(u_mean, color=c["band"], linewidth=0.9, alpha=0.55, zorder=0)
            ax.scatter(np.full(un.size, XS["Uninjured"]) + sc.jitter(un.size, 0.06), un,
                       s=34, facecolor=c["Uninjured"], edgecolor=c["surface"],
                       linewidth=1.0, zorder=4)
            ax.errorbar(XS["Uninjured"], u_mean, yerr=u_sem, fmt="o", color=c["Uninjured"],
                        markersize=7, elinewidth=1.4, capsize=4, zorder=5)
        for arm, dx in (("Vehicle", -0.06), ("NM72", 0.06)):
            xs, ms, es = [], [], []
            for tp in ("6 h", "24 h"):
                v = sc.group_values(data, tp, arm, key)
                if v.size == 0:
                    continue
                m, sem = sc.mean_sem(v)
                xs.append(XS[tp] + dx); ms.append(m); es.append(sem)
                ax.scatter(np.full(v.size, XS[tp] + dx) + sc.jitter(v.size, 0.06), v,
                           s=34, facecolor=c[arm], alpha=0.85, edgecolor=c["surface"],
                           linewidth=1.0, zorder=4)
            if xs:
                ax.plot(xs, ms, color=c[arm], linewidth=2.0, zorder=3)
                ax.errorbar(xs, ms, yerr=es, fmt="o", color=c[arm], markersize=7,
                            elinewidth=1.4, capsize=4, zorder=5)
        ax.set_xlim(-0.35, 2.35)
        ax.set_xticks(list(XS.values()))
        ax.set_xticklabels(["Uninj", "6 h", "24 h"], color=c["text_secondary"], fontsize=9)
        ax.set_ylim(bottom=0)
        ax.yaxis.grid(True, color=c["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(c["grid"])
        ax.tick_params(colors=c["text_secondary"], length=4, width=0.8, labelsize=9)
        ax.set_title(label, fontsize=9.5, fontweight="bold", color=c["text"], loc="left", pad=16)
        ax.text(0.0, 1.015, denom, transform=ax.transAxes, fontsize=8,
                color=c["text_muted"], ha="left", va="bottom")
    for ax in flat[len(panels):]:
        ax.set_visible(False)

    leg = fig.legend(handles=sc.legend_handles(c), loc="upper left", frameon=False,
                     fontsize=9.5, ncol=3, bbox_to_anchor=(0.05, 1 - 0.42 / h_total),
                     handletextpad=0.4, columnspacing=1.4)
    for t in leg.get_texts():
        t.set_color(c["text_secondary"])
    fig.suptitle(title, fontsize=13, fontweight="bold", color=c["text"],
                 x=0.05, ha="left", y=1 - 0.10 / h_total)
    _footnote(fig, c, (notes or []) + ["separate animals per timepoint, not a within-animal trajectory"])
    fig.tight_layout(rect=[0.005, 0.035, 1, 1 - (head - 0.05) / h_total], h_pad=2.6, w_pad=1.8)
    sc.save_figure(fig, out_path, c["surface"])
    plt.close(fig)
    return out_path


def export_prism(data: pd.DataFrame, panels: list, outdir: Path) -> None:
    """Per-animal values, group summaries, contrasts, and one table per graph.

    Prism builds a graph from a table whose columns are the groups, so each
    population also gets its own file laid out that way: paste it straight in.
    """
    pdir = outdir / "prism"
    (pdir / "per_graph").mkdir(parents=True, exist_ok=True)

    long_rows, summary_rows, contrast_rows = [], [], []
    for label, key, denom in panels:
        plain = _plain(label)
        for _, r in data.iterrows():
            v = r[key]
            if pd.isna(v):
                continue
            nev = r.get(key + "__n", np.nan)
            long_rows.append({"animal": int(r["animal"]), "timepoint": r["timepoint"],
                              "treatment": r["treatment"],
                              "group": f"{r['timepoint']} {r['treatment']}".replace(
                                  "Uninjured Uninjured", "Uninjured"),
                              "population": plain, "denominator": _plain(denom or ""),
                              "value": float(v),
                              "events": float(nev) if pd.notna(nev) else np.nan})
        wide_cols = {}
        for tp, tr, _x in sc.GROUPS:
            gname = f"{tp} {tr}".replace("Uninjured Uninjured", "Uninjured")
            vals = sc.group_values(data, tp, tr, key)
            wide_cols[gname] = list(vals)
            mean, sem = sc.mean_sem(vals)
            ev = data.loc[(data["timepoint"] == tp) & (data["treatment"] == tr),
                          key + "__n"] if key + "__n" in data.columns else pd.Series(dtype=float)
            summary_rows.append({"population": plain, "denominator": _plain(denom or ""),
                                 "group": gname, "n": vals.size,
                                 "median_events": float(np.nanmedian(ev)) if len(ev.dropna()) else np.nan,
                                 "mean": mean,
                                 "sd": float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan,
                                 "sem": sem})
        width = max((len(v) for v in wide_cols.values()), default=0)
        pd.DataFrame({k: v + [np.nan] * (width - len(v)) for k, v in wide_cols.items()}).to_csv(
            pdir / "per_graph" / f"{_slug(label)}.csv", index=False)

        # Both kinds of comparison, not only the drug one: the injury and time
        # contrasts are where the neutrophil result lives, and leaving them out
        # made "nothing but B-1a is significant" read as broader than it was.
        for (tp1, tr1, tp2, tr2, kind) in CONTRAST_SET:
            a = sc.group_values(data, tp1, tr1, key)
            b = sc.group_values(data, tp2, tr2, key)
            if a.size < 2 or b.size < 2:
                continue
            t, pv = stats.ttest_ind(a, b, equal_var=False)
            g, lo, hi = _hedges_g(a, b)
            contrast_rows.append({
                "population": plain, "kind": kind,
                "group_a": f"{tp1} {tr1}".replace("Uninjured Uninjured", "Uninjured"),
                "group_b": f"{tp2} {tr2}".replace("Uninjured Uninjured", "Uninjured"),
                "n_a": a.size, "n_b": b.size,
                "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
                "ratio_b_over_a": float(np.mean(b) / np.mean(a)) if np.mean(a) else np.nan,
                "welch_t": float(t), "welch_p": float(pv),
                "hedges_g": g, "g_ci_low": lo, "g_ci_high": hi})

    pd.DataFrame(long_rows).to_csv(pdir / "per_animal_long.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(pdir / "group_summary.csv", index=False)
    con = pd.DataFrame(contrast_rows).sort_values(["kind", "welch_p"])
    # Benjamini-Hochberg within each family of comparisons, since this is a wide
    # screen: a raw p near 0.05 among dozens of tests is not one result in twenty.
    con["bh_q"] = np.nan
    for kind, grp in con.groupby("kind"):
        pv = grp["welch_p"].to_numpy()
        order = np.argsort(pv)
        m = len(pv)
        q = np.empty(m)
        q[order] = np.minimum.accumulate((pv[order] * m / np.arange(1, m + 1))[::-1])[::-1]
        con.loc[grp.index, "bh_q"] = np.clip(q, 0, 1)
    con.to_csv(pdir / "contrasts_all.csv", index=False)

    ph_rows = []
    order = ["Uninjured", "6 h Vehicle", "6 h NM72", "24 h Vehicle", "24 h NM72"]
    lf = pd.DataFrame(long_rows)
    for pop, d in lf.groupby("population", sort=False):
        res = ph.pairwise({g: d[d.group == g]["value"].to_numpy(float) for g in order})
        ph_rows += [{"population": pop, **r, "dropped_n_lt_2": ",".join(res["dropped"])}
                    for r in res["rows"]]
    pd.DataFrame(ph_rows).to_csv(pdir / "posthoc_all_methods.csv", index=False)
    con[con["kind"] == "drug"].to_csv(pdir / "contrasts_vehicle_vs_nm72.csv", index=False)
    print(f"Prism bundle: {pdir}/ "
          f"(per_animal_long, group_summary, contrasts, per_graph/*.csv)")


def summarize(data: pd.DataFrame, key: str, label: str) -> pd.DataFrame:
    rows = []
    for tp, tr, _x in sc.GROUPS:
        v = sc.group_values(data, tp, tr, key)
        mean, sem = sc.mean_sem(v)
        rows.append({"population": label,
                     "group": f"{tp} {tr}".replace("Uninjured Uninjured", "Uninjured"),
                     "n": v.size, "mean": mean, "sem": sem})
    return pd.DataFrame(rows)


def contrasts(data: pd.DataFrame, key: str, label: str) -> pd.DataFrame:
    rows = []
    for tp in ("6 h", "24 h"):
        veh = sc.group_values(data, tp, "Vehicle", key)
        nm = sc.group_values(data, tp, "NM72", key)
        if veh.size < 2 or nm.size < 2:
            continue
        t, p = stats.ttest_ind(veh, nm, equal_var=False)
        rows.append({"population": label, "timepoint": tp, "n_veh": veh.size,
                     "n_nm72": nm.size, "diff": float(np.mean(nm) - np.mean(veh)),
                     "p": float(p)})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Plot FlowJo T cell / myeloid populations by SCI cohort.")
    p.add_argument("--wsp", default=None, help="FlowJo workspace to read counts from.")
    p.add_argument("--counts", default=str(DEFAULT_COUNTS),
                   help="Cached counts CSV (written when --wsp is given, else read).")
    p.add_argument("--meta", default=str(DEFAULT_META),
                   help="Cached per-sample acquisition metadata CSV.")
    p.add_argument("--mfi", default=str(DEFAULT_MFI),
                   help="Cached non-frequency (MFI) statistics CSV.")
    p.add_argument("--group", default=SPL_GROUP, help="Workspace group to use.")
    p.add_argument("--min-live", type=int, default=1000,
                   help="Drop animals with fewer live leukocyte events than this.")
    p.add_argument("--outdir", default="outputs")
    p.add_argument("--dark", action="store_true")
    p.add_argument("--drug-only", action="store_true",
                   help="Drop the uninjured group: vehicle vs NM72 alone.")
    ns = p.parse_args(argv if argv is not None else sys.argv[1:])

    counts_path = Path(ns.counts)
    meta_path = Path(ns.meta)
    mfi_path = Path(ns.mfi)
    if ns.wsp:
        counts = parse_wsp(Path(ns.wsp), group=ns.group)
        meta = parse_metadata(Path(ns.wsp), group=ns.group)
        mfi = parse_mfi(Path(ns.wsp), group=ns.group)
        counts_path.parent.mkdir(parents=True, exist_ok=True)
        counts.to_csv(counts_path, index=False)
        meta.to_csv(meta_path, index=False)
        mfi.to_csv(mfi_path, index=False)
        print(f"Parsed {ns.wsp} -> {counts_path}, {meta_path}")
    else:
        if not counts_path.exists():
            raise SystemExit(f"{counts_path} not found; pass --wsp to build it.")
        counts = pd.read_csv(counts_path, keep_default_na=False, na_values=[""])
        counts["count"] = counts["count"].astype(int)
        meta = pd.read_csv(meta_path) if meta_path.exists() else pd.DataFrame()
        mfi = pd.read_csv(mfi_path) if mfi_path.exists() else pd.DataFrame()

    counts, dup_notes = resolve_duplicates(counts)
    for n in dup_notes:
        print(f"Duplicate sample  {n}")

    wide = build_table(counts)
    if len(meta):
        sid = counts.drop_duplicates("sample").set_index("sample")["sampleID"].astype(str)
        wide["sampleID"] = wide["sample"].map(sid)
        keep = [c for c in ("sampleID", "date", "btim", "flowrate") if c in meta.columns]
        wide = wide.merge(meta.astype({"sampleID": str})[keep], on="sampleID", how="left")
    for c in ("date", "btim", "flowrate"):
        if c not in wide.columns:
            wide[c] = None
    unmapped = wide[wide["treatment"].isna()]["animal"].tolist()
    if unmapped:
        print(f"Not in the cohort sheet, skipped: {unmapped}")
    wide = wide.dropna(subset=["treatment"]).copy()

    # Exclusions, judged on the live leukocyte gate.
    ranked = wide[["animal", LIVE]].sort_values(LIVE)
    print("\nLive leukocyte events, ascending:")
    print("  " + ", ".join(f"{int(a)}: {int(v):,}" for a, v in ranked.values))
    failed = wide[wide[LIVE] < ns.min_live]
    print(f"\nExcluded at --min-live {ns.min_live:,}: "
          f"{', '.join(str(int(a)) for a in failed['animal']) or 'none'}")
    qc_frame = wide.copy()
    wide["live_ok"] = wide[LIVE] >= ns.min_live

    wide["tech_block"] = detect_outlier_block(wide[wide["live_ok"]]).reindex(wide.index, fill_value=False)
    block = wide[wide["tech_block"]]
    confound = ""
    if len(block):
        in_order = wide.sort_values(["date", "btim"]).reset_index(drop=True)
        pos = np.where(in_order["tech_block"].to_numpy())[0]
        contiguous = len(pos) == (pos.max() - pos.min() + 1)
        groups = block.groupby(["timepoint", "treatment"]).size()
        whole = [f"{tp} {tr}" for (tp, tr), n in groups.items()
                 if n == int((wide["timepoint"] == tp).mul(wide["treatment"] == tr).sum())]
        print("\n" + "!" * 72)
        print(f"Technical cluster: {sorted(int(a) for a in block['animal'])}")
        print(f"  CD4+CD8 as % of CD3+ separates these from the rest with no overlap.")
        print(f"  Consecutive in acquisition order: {contiguous}")
        print(f"  Cohort groups falling ENTIRELY inside it: {whole or 'none'}")
        if whole:
            confound = (f"QC cluster {sorted(int(a) for a in block['animal'])} "
                        f"wholly contains {', '.join(whole)} - comparisons confounded")
            print("  -> Any cohort difference involving those groups is confounded with "
                  "the acquisition window.")
        print("!" * 72)

    all_panels = [("tcell", T_PANELS), ("myeloid", M_PANELS)]
    specs: dict[str, list] = {}
    for kind, panels in all_panels:
        built = []
        for label, num, den, den_label in panels:
            key = f"pct::{label}"
            add_frequency(wide, key, num, den)
            built.append((label, key, den_label))
        specs[kind] = built

    # CD4:CD8 is denominator-free, so it survives a shift in total T cell number.
    for _key in [c for c in wide.columns if c.startswith("pct::")]:
        wide.loc[~wide["live_ok"], _key] = np.nan
    specs["bcell"] = add_b1a(wide)

    # Non-frequency statistics, when the workspace carries any.
    mfi_specs = []
    if len(mfi):
        mfi = mfi.astype({"sampleID": str})
        for (path, marker, stat), grp in mfi.groupby(["path", "marker", "stat"]):
            label = f"{marker} {stat} ({path.split('/')[-1]})"
            key = f"mfi::{label}"
            # keyed on sampleID, not sample name: a sample imported twice shares
            # its name, and resolve_duplicates has already picked which id wins
            lookup = grp.drop_duplicates("sampleID").set_index("sampleID")["value"]
            wide[key] = wide["sampleID"].map(lookup)
            wide.loc[~wide["live_ok"], key] = np.nan
            if path in wide.columns:
                wide[key + "__n"] = wide[path].astype(float)
            got = int(wide[key].notna().sum())
            mfi_specs.append((label, key, f"{stat} fluorescence intensity"))
            if got < len(wide):
                print(f"  {label}: present for {got}/{len(wide)} animals")
    specs["mfi"] = mfi_specs

    wide["pct::CD4:CD8 ratio"] = (wide[CD4].astype(float)
                                  / wide[f"{CD3}/CD8+"].astype(float).where(
                                      wide[f"{CD3}/CD8+"].astype(float) > 0))
    wide.loc[~wide["live_ok"], "pct::CD4:CD8 ratio"] = np.nan
    specs["tcell"].append(("CD4:CD8 ratio", "pct::CD4:CD8 ratio", "ratio of counts"))

    outdir = Path(ns.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    thin = []
    summaries, contrast_rows = [], []
    for kind, panels in specs.items():
        for label, key, _dl in panels:
            if key + "__n" in wide.columns:
                nmin = int(np.nanmin(wide[key + "__n"]))
                dropped = int(wide[key].isna().sum())
                if dropped:
                    thin.append(f"{label} (n/a in {dropped})")
                elif nmin < 100:
                    thin.append(f"{label} (min {nmin} events)")
            summaries.append(summarize(wide, key, label))
            contrast_rows.append(contrasts(wide, key, label))

    summary = pd.concat(summaries, ignore_index=True)
    contrast = pd.concat([c for c in contrast_rows if not c.empty], ignore_index=True)
    summary.to_csv(outdir / "flow_panel_summary.csv", index=False)
    contrast.to_csv(outdir / "flow_panel_contrasts.csv", index=False)
    wide.drop(columns=[c for c in wide.columns if c.endswith(("__n", "__d"))]).to_csv(
        outdir / "flow_panel_per_animal.csv", index=False)

    print("\nPer group (excluded animals dropped):")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\nVehicle vs NM72 (Welch; n=2-4 per group, descriptive only):")
    print(contrast.sort_values("p").to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if thin:
        print("\nLow event counts, read these with care: " + "; ".join(thin))

    excluded_note = (", ".join(str(int(a)) for a in failed["animal"]) or "none")
    notes = [f"excluded: {excluded_note} (<{ns.min_live:,} live leukocytes)",
             "panels have independent y-scales",
             f"frequencies blanked below {MIN_NUMERATOR} numerator events"]
    if not ns.drug_only:
        notes.insert(1, "shaded band = uninjured mean ± SEM")
    if confound:
        notes.insert(0, "SEE QC FIGURE: " + confound)

    suffix = "_drug" if ns.drug_only else ""
    sub = wide[wide["treatment"] != "Uninjured"] if ns.drug_only else wide
    tag = "  ·  NM72 vs vehicle" if ns.drug_only else ""
    plot_grid(sub, specs["tcell"], outdir / f"flow_tcell{suffix}.png",
              "Splenic T cell populations after SCI" + tag, ncols=3, dark=ns.dark,
              notes=notes, drug_only=ns.drug_only)
    plot_grid(sub, specs["myeloid"], outdir / f"flow_myeloid{suffix}.png",
              "Splenic myeloid populations after SCI" + tag, ncols=4, dark=ns.dark,
              notes=notes, drug_only=ns.drug_only)
    plot_grid(sub, specs["bcell"] + specs["tcell"] + specs["myeloid"],
              outdir / f"flow_all{suffix}.png",
              "Splenic populations after SCI, both panels" + tag, ncols=5, dark=ns.dark,
              notes=notes + ["B-1a / IgM- are a separate stain, merged by animal"],
              drug_only=ns.drug_only)
    plot_qc(qc_frame, ns.min_live, outdir / "flow_qc_live.png", dark=ns.dark)
    if wide["btim"].notna().any():
        plot_acquisition_qc(wide, outdir / "flow_qc_runorder.png", dark=ns.dark)

    # Which timepoints are safe for a drug-vs-vehicle read: a timepoint whose two
    # arms differ on sample quality cannot separate drug from prep.
    bad_tp = set()
    for tp in ("6 h", "24 h"):
        at_tp = wide[wide["live_ok"] & (wide["timepoint"] == tp)]
        arms = at_tp.groupby("treatment")["tech_block"].mean()
        if len(arms) == 2 and abs(arms.diff().iloc[-1]) >= 0.5:
            bad_tp.add(tp)
        # Acquisition settings that differ between the two arms are on their own
        # enough to sink the comparison, however the QC block came out.
        if "flowrate" in at_tp.columns and at_tp["flowrate"].notna().any():
            byarm = at_tp.groupby("treatment")["flowrate"].apply(lambda v: set(v.dropna()))
            if len(byarm) == 2 and byarm.iloc[0] != byarm.iloc[1]:
                bad_tp.add(tp)
    print(f"\nTimepoints unusable for drug-vs-vehicle (arms differ on quality): "
          f"{sorted(bad_tp) or 'none'}")
    every = specs["bcell"] + specs["tcell"] + specs["myeloid"] + specs["mfi"]
    by_label = {l: (l, k, d) for l, k, d in every}

    fam_dir = outdir / f"families{suffix}"
    fam_dir.mkdir(parents=True, exist_ok=True)
    placed = set()
    for fam, labels in FAMILIES.items():
        chosen = [by_label[l] for l in labels if l in by_label]
        if not chosen:
            continue
        placed.update(l for l, _k, _d in chosen)
        plot_grid(sub, chosen, fam_dir / f"{_slug(fam)}.png", fam + tag,
                  ncols=min(3, len(chosen)), dark=ns.dark, notes=notes,
                  drug_only=ns.drug_only)

    ind_dir = outdir / f"individual{suffix}"
    ind_dir.mkdir(parents=True, exist_ok=True)
    for label, key, denom in every:
        plot_single(sub, label, key, denom, ind_dir / f"{_slug(label)}.png",
                    dark=ns.dark, drug_only=ns.drug_only)
    leftover = [l for l, _k, _d in every if l not in placed]
    if leftover:
        print(f"Not in any family, single figures only: {[_plain(l) for l in leftover]}")
    print(f"Figures: {len(FAMILIES)} family plots in {fam_dir}/, "
          f"{len(every)} single plots in {ind_dir}/")

    tc = [by_label[l] for l in ["Neutrophils", "CD11b$^+$F4/80$^-$", "CD11b$^+$F4/80$^+$",
                                "Red pulp macrophages", "B-1a  (IgM$^+$)", "IgM$^-$",
                                "CD3$^+$ T cells", "CD4$^+$", "CD8$^+$"] if l in by_label]
    if tc and not ns.drug_only:
        plot_timecourse(wide, tc, outdir / "flow_timecourse.png",
                        "Time course: uninjured to 6 h to 24 h", ncols=3,
                        dark=ns.dark, notes=notes)

    export_prism(wide, every, outdir)

    plot_contrast(wide, specs["bcell"] + specs["tcell"] + specs["myeloid"],
                  outdir / "flow_contrast.png", dark=ns.dark, confounded=bad_tp)
    print(f"\nSaved figures to {outdir}/ (flow_tcell, flow_myeloid, flow_qc_live, "
          f"flow_qc_runorder) and 3 CSVs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
