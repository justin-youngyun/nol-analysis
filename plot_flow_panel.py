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

DEFAULT_COUNTS = Path("data/sci_flow_counts.csv")
DEFAULT_META = Path("data/sci_flow_samples.csv")
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
              ncols: int, dark: bool = False, notes: list[str] | None = None) -> Path:
    c = sc.palette(dark)
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.55 * ncols, 3.15 * nrows),
                             facecolor=c["surface"], squeeze=False)
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
                      subtitle_color=c["flag"] if thin else None)
    for ax in flat[len(panels):]:
        ax.set_visible(False)

    leg = fig.legend(handles=sc.legend_handles(c), loc="upper right", frameon=False,
                     fontsize=9.5, ncol=3, bbox_to_anchor=(0.995, 0.995),
                     handletextpad=0.4, columnspacing=1.4)
    for t in leg.get_texts():
        t.set_color(c["text_secondary"])
    fig.suptitle(title, fontsize=13, fontweight="bold", color=c["text"],
                 x=0.05, ha="left", y=0.995)
    _footnote(fig, c, notes or [])
    fig.tight_layout(rect=[0.005, 0.035, 1, 0.955], h_pad=2.6, w_pad=1.8)
    fig.savefig(out_path, bbox_inches="tight", facecolor=c["surface"], dpi=200)
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
    fig.savefig(out_path, bbox_inches="tight", facecolor=c["surface"], dpi=200)
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
    fig.savefig(out_path, bbox_inches="tight", facecolor=c["surface"], dpi=200)
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
            rows.append({"label": label, "tp": tp, "g": g, "lo": lo, "hi": hi,
                         "n": f"{veh.size}v{nm.size}"})
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
            hue = c["text_muted"] if bad else (c["NM72"] if r["g"] > 0 else c["Vehicle"])
            ax.plot([r["lo"], r["hi"]], [yi, yi], color=hue, linewidth=2.0,
                    alpha=0.45 if bad else 0.9, zorder=2)
            ax.scatter(r["g"], yi, s=70, facecolor=hue, edgecolor=c["surface"],
                       linewidth=1.3, alpha=0.5 if bad else 1.0, zorder=3)
            ax.text(1.01, yi, r["n"], transform=ax.get_yaxis_transform(),
                    va="center", ha="left", fontsize=7.5, color=c["text_muted"])
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
        ax.set_title(f"{tp}" + ("   ·  CONFOUNDED, do not interpret" if bad else
                                "   ·  quality-balanced"),
                     fontsize=11, fontweight="bold", loc="left", pad=8,
                     color=c["flag"] if bad else c["text"])

    fig.suptitle("NM72 vs vehicle, by population", fontsize=13, fontweight="bold",
                 color=c["text"], x=0.045, ha="left", y=0.995)
    fig.text(0.045, 0.012,
             "Hedges' g with 95% CI  ·  n per group shown at right (vehicle v NM72)  ·  "
             "at n=3-4 every interval crosses zero; none of these is a positive result",
             fontsize=8, color=c["text_muted"], ha="left")
    fig.tight_layout(rect=[0.005, 0.045, 1, 0.94])
    fig.savefig(out_path, bbox_inches="tight", facecolor=c["surface"], dpi=200)
    plt.close(fig)
    return out_path


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
    p.add_argument("--group", default=SPL_GROUP, help="Workspace group to use.")
    p.add_argument("--min-live", type=int, default=1000,
                   help="Drop animals with fewer live leukocyte events than this.")
    p.add_argument("--outdir", default="outputs")
    p.add_argument("--dark", action="store_true")
    ns = p.parse_args(argv if argv is not None else sys.argv[1:])

    counts_path = Path(ns.counts)
    meta_path = Path(ns.meta)
    if ns.wsp:
        counts = parse_wsp(Path(ns.wsp), group=ns.group)
        meta = parse_metadata(Path(ns.wsp), group=ns.group)
        counts_path.parent.mkdir(parents=True, exist_ok=True)
        counts.to_csv(counts_path, index=False)
        meta.to_csv(meta_path, index=False)
        print(f"Parsed {ns.wsp} -> {counts_path}, {meta_path}")
    else:
        if not counts_path.exists():
            raise SystemExit(f"{counts_path} not found; pass --wsp to build it.")
        counts = pd.read_csv(counts_path, keep_default_na=False, na_values=[""])
        counts["count"] = counts["count"].astype(int)
        meta = pd.read_csv(meta_path) if meta_path.exists() else pd.DataFrame()

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
    wide = wide[wide[LIVE] >= ns.min_live].copy()

    wide["tech_block"] = detect_outlier_block(wide)
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
    wide["pct::CD4:CD8 ratio"] = (wide[CD4].astype(float)
                                  / wide[f"{CD3}/CD8+"].astype(float).where(
                                      wide[f"{CD3}/CD8+"].astype(float) > 0))
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
             "shaded band = uninjured mean ± SEM",
             "panels have independent y-scales",
             f"frequencies blanked below {MIN_NUMERATOR} numerator events"]
    if confound:
        notes.insert(0, "SEE QC FIGURE: " + confound)

    plot_grid(wide, specs["tcell"], outdir / "flow_tcell.png",
              "Splenic T cell populations after SCI", ncols=3, dark=ns.dark, notes=notes)
    plot_grid(wide, specs["myeloid"], outdir / "flow_myeloid.png",
              "Splenic myeloid populations after SCI", ncols=4, dark=ns.dark, notes=notes)
    plot_qc(qc_frame, ns.min_live, outdir / "flow_qc_live.png", dark=ns.dark)
    if wide["btim"].notna().any():
        plot_acquisition_qc(wide, outdir / "flow_qc_runorder.png", dark=ns.dark)

    # Which timepoints are safe for a drug-vs-vehicle read: a timepoint whose two
    # arms differ on sample quality cannot separate drug from prep.
    bad_tp = set()
    for tp in ("6 h", "24 h"):
        sub = wide[wide["timepoint"] == tp]
        arms = sub.groupby("treatment")["tech_block"].mean()
        if len(arms) == 2 and abs(arms.diff().iloc[-1]) >= 0.5:
            bad_tp.add(tp)
        # Acquisition settings that differ between the two arms are on their own
        # enough to sink the comparison, however the QC block came out.
        if "flowrate" in sub.columns and sub["flowrate"].notna().any():
            byarm = sub.groupby("treatment")["flowrate"].apply(lambda v: set(v.dropna()))
            if len(byarm) == 2 and byarm.iloc[0] != byarm.iloc[1]:
                bad_tp.add(tp)
    print(f"\nTimepoints unusable for drug-vs-vehicle (arms differ on quality): "
          f"{sorted(bad_tp) or 'none'}")
    plot_contrast(wide, specs["tcell"] + specs["myeloid"],
                  outdir / "flow_contrast.png", dark=ns.dark, confounded=bad_tp)
    print(f"\nSaved figures to {outdir}/ (flow_tcell, flow_myeloid, flow_qc_live, "
          f"flow_qc_runorder) and 3 CSVs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
