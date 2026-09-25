#!/usr/bin/env python
"""
The summary panels, taken apart so PowerPoint can rebuild them as editable objects.

Each panel (A-F of make_final_panels.py) is drawn by the same draw() call as the
summary figure, on its own canvas with fixed margins, so panels line up when
laid side by side. Matplotlib does the layout; then every text (title, axis
label, tick labels, n=, p-values, notes) and every bracket and timepoint line
is recorded, with its position, size and colour, and removed from the image.
What is left, points, error bars, bands, grid and axes, is written as PNG and
SVG. make_slides.py puts the image back and rebuilds the rest as native text
boxes and lines, grouped per panel.

    python3 slide_figures.py      # writes the panel images and prints the counts
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import pandas as pd
from lxml import etree

import make_final_panels as fp
import plot_flow_panel as pf
import posthoc as ph
import sci_cohorts as sc

OUTDIR = Path("outputs/slides")
BY_LETTER = dict(zip("ABCDEF", fp.PANELS))

# Fixed canvas margins in inches (left, right, top, bottom): room for the widest
# tick labels (MerTK), the title and letter above, and the timepoint labels below.
MARGINS = (0.95, 0.15, 0.5, 0.68)


@dataclass
class TextItem:
    """One text, as laid out by matplotlib. Inches from the canvas top-left."""
    text: str          # with ^{...} marking superscripts
    x0: float
    y0: float
    x1: float
    y1: float
    size: float        # points
    color: str
    bold: bool
    rotation: float
    align: str         # left, center, right
    kind: str          # tick, title, label, pvalue, note, n, letter, timepoint


@dataclass
class LineItem:
    points: list[tuple[float, float]]   # inches from the canvas top-left
    color: str
    width: float       # points


@dataclass
class Panel:
    letter: str
    png: Path
    svg: Path
    width: float       # canvas, inches
    height: float
    texts: list[TextItem] = field(default_factory=list)
    lines: list[LineItem] = field(default_factory=list)


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


def markup(s: str) -> str:
    """Matplotlib mathtext to plain text, superscripts marked ^{...}, a real minus."""
    def math(m: re.Match) -> str:
        body = m.group(1).replace(r"\mathdefault", "")
        body = re.sub(r"\^\{([^}]*)\}|\^(.)",
                      lambda k: "\0" + (k.group(1) or k.group(2)).replace("-", "−") + "\1",
                      body)
        body = body.replace("{", "").replace("}", "")
        return body.replace("\0", "^{").replace("\1", "}")
    out = re.sub(r"\$([^$]*)\$", math, s)
    return re.sub(r" {2,}", " ", out)


def _is_bold(weight) -> bool:
    if isinstance(weight, str):
        return weight in ("bold", "heavy", "extra bold", "black", "demibold", "semibold")
    return weight >= 600


def _visible_ticklabels(axis, lo: float, hi: float):
    span = hi - lo
    for loc, label in zip(axis.get_ticklocs(), axis.get_ticklabels()):
        if lo - 1e-9 * span <= loc <= hi + 1e-9 * span and label.get_text().strip():
            yield label


def _kind(t: mtext.Text, ax, ticks: set) -> str:
    s = t.get_text()
    if id(t) in ticks:
        return "tick"
    if t is ax.title or t is ax._left_title:
        return "title"
    if t is ax.yaxis.label:
        return "label"
    if s.startswith("p ") or s.startswith("p<") or s.startswith("p="):
        return "pvalue"
    if s.startswith("n="):
        return "n"
    if s in ("6 h", "24 h"):
        return "timepoint"
    if len(s) == 1 and s.isupper():
        return "letter"
    return "note"


def render_panel(letter: str, size: tuple[float, float], method: str, results: dict,
                 long: pd.DataFrame, outdir: Path = OUTDIR) -> Panel:
    """Draw one panel, lift its text and lines out, and save what remains."""
    label, pop, ylab, brackets = BY_LETTER[letter]
    col = ph.METHODS[method][0]
    c = sc.palette(False)
    w, h = size
    left, right, top, bottom = MARGINS
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([left / w, bottom / h, (w - left - right) / w, (h - top - bottom) / h])
    fp.draw(ax, long, results[pop], label, pop, ylab, brackets, c, col, method != "none")
    ax.text(-0.17, 1.06, letter, transform=ax.transAxes, fontsize=15, fontweight="bold",
            color=c["text"], va="top")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    dpi = fig.dpi
    panel = Panel(letter, Path(), Path(), w, h)

    ylo, yhi = ax.get_ylim()
    xlo, xhi = ax.get_xlim()
    candidates = list(_visible_ticklabels(ax.xaxis, xlo, xhi)) + \
        list(_visible_ticklabels(ax.yaxis, ylo, yhi))
    ticks = set(id(t) for t in candidates)
    # Tick labels outside the view limits exist but are never drawn: skip them all
    # here, the drawn ones are already in.
    every_tick = {id(t) for axis in (ax.xaxis, ax.yaxis)
                  for t in axis.get_ticklabels(minor=False) + axis.get_ticklabels(minor=True)}
    for t in fig.findobj(mtext.Text):
        if id(t) not in every_tick and t.get_visible() and t.get_text().strip():
            candidates.append(t)
    for t in candidates:
        bb = t.get_window_extent(renderer)
        rot = t.get_rotation() % 360
        panel.texts.append(TextItem(
            text=markup(t.get_text()),
            x0=bb.x0 / dpi, x1=bb.x1 / dpi, y0=h - bb.y1 / dpi, y1=h - bb.y0 / dpi,
            size=float(t.get_fontsize()), color=mcolors.to_hex(t.get_color())[1:].upper(),
            bold=_is_bold(t.get_fontweight()), rotation=rot,
            align="center" if rot else t.get_horizontalalignment(),
            kind=_kind(t, ax, ticks)))
        if id(t) not in ticks:
            t.set_visible(False)
    ax.tick_params(labelbottom=False, labelleft=False)

    for ln in ax.get_lines():
        if ln.get_label() in (fp.BRACKET, fp.TIMEPOINT):
            pts = ln.get_transform().transform(ln.get_xydata())
            panel.lines.append(LineItem(
                points=[(x / dpi, h - y / dpi) for x, y in pts],
                color=mcolors.to_hex(ln.get_color())[1:].upper(),
                width=float(ln.get_linewidth())))
            ln.set_visible(False)

    outdir.mkdir(parents=True, exist_ok=True)
    tag = "" if method == "dunnett-t3" else f"_{method}"
    base = outdir / f"panel_{letter}_{round(w * 100)}x{round(h * 100)}{tag}"
    panel.png, panel.svg = base.with_suffix(".png"), base.with_suffix(".svg")
    fig.savefig(panel.png, dpi=300, transparent=True)
    fig.savefig(panel.svg, transparent=True)
    plt.close(fig)
    office_svg(panel.svg)
    return panel


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
    return wide.set_index("animal")[["timepoint", "treatment", "live_ok", "block", "btim",
                                     "viability", "live_events"]]


def main() -> int:
    long = pd.read_csv(fp.LONG)
    results = {pop: fp._stats(long, pop) for _l, pop, _y, _b in fp.PANELS}
    for letter in "ABCDEF":
        p = render_panel(letter, (5.9, 4.3), "dunnett-t3", results, long)
        kinds = sorted({t.kind for t in p.texts})
        print(f"{p.png}: {len(p.texts)} texts ({', '.join(kinds)}), {len(p.lines)} lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
