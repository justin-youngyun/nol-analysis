#!/usr/bin/env python
"""
The slide deck: the summary panels, and the statistics behind them. Plots only.

    python3 make_slides.py        # -> outputs/slides/NM72_spleen_flow.pptx

Slides: all six panels (A-F), then neutrophils (C), B-1a with its IgM- fraction
(A-B), Tregs with CD25+CD127- (D-E), MerTK (F), then two tables: the bracketed
comparisons among all pairs of the five groups under three procedures, and
vehicle vs NM72 within each timepoint.

Each panel is one PowerPoint group, so it moves and resizes as a unit. Inside
the group the points, error bars and grid are a picture (PNG, with the SVG
embedded so PowerPoint 365 can Convert to Shape), and everything else is
native: every title, axis label, tick label, n= and p-value is a text box, and
every bracket and timepoint line is a line. slide_figures.py draws the panels
with the summary figure's own draw() call and records where each of those
pieces goes.

Only slide 1 has speaker notes: which test each bracket shows.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lxml import etree
from PIL import ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.opc.package import Part
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

import make_final_panels as fp
import slide_figures as sf

OUT = sf.OUTDIR / "NM72_spleen_flow.pptx"
W, H = 13.333, 7.5
FONT = "Arial"

INK = "0B0B0B"
TEXT2 = "52514E"
MUTED = "7A7973"
RULE = "E6E5E1"
CARD = "F3F3F1"
WHITE = "FFFFFF"

_FONTS = {False: "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
          True: "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"}

# Panel canvas (inches, as drawn) and the scale each slide shows it at.
SUMMARY = ((4.4, 3.5), 0.945)
SINGLE = ((5.9, 4.3), 1.4)
PAIR = ((5.0, 4.9), 1.22)


# ---------------------------------------------------------------------------
# Numbers, for the speaker notes and the table
# ---------------------------------------------------------------------------

class Numbers:
    def __init__(self):
        self.long = pd.read_csv(fp.LONG)
        self.res = {pop: fp._stats(self.long, pop) for _l, pop, _y, _b in fp.PANELS}
        self.within = {pop: fp._within(self.long, pop) for _l, pop, _y, _b in fp.PANELS}

    def values(self, pop: str, group: str) -> np.ndarray:
        d = self.long[(self.long.population == pop) & (self.long.group == group)]
        return d["value"].dropna().to_numpy(float)

    def mean(self, pop: str, group: str) -> float:
        return float(self.values(pop, group).mean())

    def n(self, pop: str, group: str) -> int:
        return int(self.values(pop, group).size)

    def p(self, pop: str, a: str, b: str, col: str = "dunnett_t3_p") -> float:
        return float(fp._p(self.res[pop], a, b, col))

    def w(self, pop: str, tp: str) -> dict:
        """The within-timepoint row: vehicle vs NM72 at tp."""
        return next(r for r in self.within[pop] if r["group_a"] == f"{tp} Vehicle")

    def change(self, pop: str, tp: str) -> float:
        v, d = self.mean(pop, f"{tp} Vehicle"), self.mean(pop, f"{tp} NM72")
        return 100 * (d - v) / v


# ---------------------------------------------------------------------------
# PowerPoint pieces
# ---------------------------------------------------------------------------

_font_cache: dict = {}


def text_width(s: str, size: float, bold: bool = False) -> float:
    """Inches, measured in Liberation Sans, which has Arial's widths."""
    key = (bold, size)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(_FONTS[bold], size=max(1, round(size * 10)))
    plain = s.replace("^{", "").replace("}", "")
    return _font_cache[key].getlength(plain) / 10 / 72


def rgb(hexstr: str) -> RGBColor:
    return RGBColor.from_string(hexstr)


def _runs(paragraph, text: str, size: float, color: str, bold: bool) -> None:
    """Fill a paragraph; ^{...} is a superscript."""
    import re
    for piece in re.split(r"(\^\{[^}]*\})", text):
        if not piece:
            continue
        sup = piece.startswith("^{")
        run = paragraph.add_run()
        run.text = piece[2:-1] if sup else piece
        f = run.font
        f.name = FONT
        f.size = Pt(size)
        f.bold = bold
        f.color.rgb = rgb(color)
        if sup:
            f._rPr.set("baseline", "30000")


def _plain(shape) -> None:
    """Drop the theme style reference, so no theme outline or shadow applies."""
    style = shape._element.find(qn("p:style"))
    if style is not None:
        shape._element.remove(style)


def embed_svg(slide, pic, svg: Path) -> None:
    """Put the SVG behind the picture's PNG, the way PowerPoint stores an inserted SVG."""
    package = slide.part.package
    part = Part(package.next_image_partname("svg"), "image/svg+xml", package, svg.read_bytes())
    rid = slide.part.relate_to(part, RT.IMAGE)
    blip = pic._element.find(".//" + qn("a:blip"))
    ext_lst = blip.find(qn("a:extLst"))
    if ext_lst is None:
        ext_lst = etree.SubElement(blip, qn("a:extLst"))
    ext = etree.SubElement(ext_lst, qn("a:ext"))
    ext.set("uri", "{96DAC541-7B7A-43D3-8B79-37D633B846F1}")
    asvg = "http://schemas.microsoft.com/office/drawing/2016/SVG/main"
    etree.SubElement(ext, f"{{{asvg}}}svgBlip", nsmap={"asvg": asvg}).set(qn("r:embed"), rid)


def _size(pt: float) -> float:
    return round(pt * 2) / 2


def add_panel(slide, panel: sf.Panel, left: float, top: float, k: float, name: str,
              alt: str = "points, mean and SEM"):
    """One panel as one group: its picture, then native lines, then native text."""
    group = slide.shapes.add_group_shape()
    group.name = name
    shapes = group.shapes

    pic = shapes.add_picture(str(panel.png), Inches(left), Inches(top),
                             Inches(panel.width * k), Inches(panel.height * k))
    pic.name = f"{name} plot"
    pic._element.nvPicPr.cNvPr.set("descr", f"{name}: {alt}")
    embed_svg(slide, pic, panel.svg)

    for ln in panel.lines:
        pts = [(Emu(Inches(left + x * k)), Emu(Inches(top + y * k))) for x, y in ln.points]
        builder = shapes.build_freeform(pts[0][0], pts[0][1], scale=1.0)
        builder.add_line_segments(pts[1:], close=False)
        shp = builder.convert_to_shape()
        _plain(shp)
        shp.fill.background()
        shp.line.color.rgb = rgb(ln.color)
        shp.line.width = Pt(ln.width * k)
        shp.name = {"timepoint": "timepoint line"}.get(ln.kind, ln.kind)

    for t in panel.texts:
        size = _size(t.size * k)
        lines = t.text.split("\n")
        tw = max(text_width(line, size, t.bold) for line in lines)
        cx = left + k * (t.x0 + t.x1) / 2
        cy = top + k * (t.y0 + t.y1) / 2
        backed = t.kind in ("pvalue", "note", "gate_backed")
        if t.rotation:
            bw, bh = tw + 0.12, size * 1.2 / 72
            bx, by = cx - bw / 2, cy - bh / 2
        else:
            pad = 0.03 if backed else 0.12
            bw = tw + 2 * pad
            bh = len(lines) * size * 1.2 / 72
            bx = {"left": left + k * t.x0,
                  "right": left + k * t.x1 - bw,
                  "center": cx - bw / 2}[t.align]
            by = cy - bh / 2
        anchor = MSO_ANCHOR.MIDDLE
        if t.kind == "pvalue":
            # Sit the label on the same baseline matplotlib gave it, just above its
            # bracket. The white backing reaches up past the ascenders, to hide
            # gridlines, but not down into the bracket.
            bh = k * (t.y1 - t.y0) + 0.25 * size / 72
            by = top + k * t.y1 - bh - 0.12 * size / 72
            anchor = MSO_ANCHOR.BOTTOM
        box = shapes.add_textbox(Inches(bx), Inches(by), Inches(bw), Inches(bh))
        box.name = t.kind
        tf = box.text_frame
        tf.word_wrap = False
        tf.auto_size = MSO_AUTO_SIZE.NONE
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        tf.vertical_anchor = anchor
        for i, line in enumerate(lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = {"left": PP_ALIGN.LEFT, "right": PP_ALIGN.RIGHT,
                           "center": PP_ALIGN.CENTER}[t.align if not t.rotation else "center"]
            p.line_spacing = 1.0
            _runs(p, line, size, t.color, t.bold)
        if t.rotation:
            box.rotation = 360 - t.rotation
        if backed:
            box.fill.solid()
            box.fill.fore_color.rgb = rgb(WHITE)
            if t.kind == "gate_backed":     # lets the events show through a little
                clr = box.fill._xPr.find(qn("a:solidFill")).find(qn("a:srgbClr"))
                etree.SubElement(clr, qn("a:alpha")).set("val", "78000")
    return group


def _cell_borders(c, fill=None, bottom=None):
    """Rewrite a cell's tcPr children in schema order: lnL, lnR, lnT, lnB, fill."""
    tc_pr = c._tc.get_or_add_tcPr()
    for child in list(tc_pr):
        tc_pr.remove(child)
    for tag, spec in (("a:lnL", None), ("a:lnR", None), ("a:lnT", None), ("a:lnB", bottom)):
        ln = etree.SubElement(tc_pr, qn(tag))
        if spec:
            ln.set("w", str(int(spec[1] * 12700)))
            solid = etree.SubElement(ln, qn("a:solidFill"))
            etree.SubElement(solid, qn("a:srgbClr")).set("val", spec[0])
        else:
            ln.set("w", "0")
            etree.SubElement(ln, qn("a:noFill"))
    if fill:
        solid = etree.SubElement(tc_pr, qn("a:solidFill"))
        etree.SubElement(solid, qn("a:srgbClr")).set("val", fill)
    else:
        etree.SubElement(tc_pr, qn("a:noFill"))


# Table rows use the panel titles, so a readout reads the same on plot and table.
READOUTS = [("B-1a (IgM+)", "B-1a (IgM^{+})"), ("IgM-", "B-1a IgM^{−}"),
            ("Neutrophils", "Neutrophils"), ("Tregs", "Tregs"),
            ("CD25+CD127-", "CD25^{+}CD127^{−}"), ("MerTK Median (M1 Like)", "MerTK")]
HEAD_H, ROW_H = 0.46, 0.228


def table_height(n_body: int) -> float:
    return HEAD_H + ROW_H * n_body


def _fmt(pv: float) -> str:
    return "< 0.001" if pv < 0.001 else f"{pv:.3f}"


def _table(s, name: str, head: list[str], widths: list[float], aligns: str, n_body: int,
           top: float, shade: int):
    """A plain table, centred across the slide, its column `shade` tinted.

    Returns a cell writer, cell(i, j, text, ...), and a section(i, text) that
    writes a label across the columns left of the tinted one.
    """
    n_rows = n_body + 1
    left = (W - sum(widths)) / 2
    gf = s.shapes.add_table(n_rows, len(head), Inches(left), Inches(top), Inches(sum(widths)),
                            Inches(ROW_H * n_rows))
    gf.name = name
    tbl = gf.table
    tbl_pr = gf._element.graphic.graphicData.tbl.tblPr
    tbl_pr.set("firstRow", "1")
    tbl_pr.set("bandRow", "0")
    for el in tbl_pr.findall(qn("a:tableStyleId")):
        tbl_pr.remove(el)
    style_id = etree.SubElement(tbl_pr, qn("a:tableStyleId"))
    style_id.text = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"  # No Style, No Grid
    for j, wdt in enumerate(widths):
        tbl.columns[j].width = Inches(wdt)
    tbl.rows[0].height = Inches(HEAD_H)
    for i in range(1, n_rows):
        tbl.rows[i].height = Inches(ROW_H)

    def cell(i, j, text, *, bold=False, color=INK, size=10.5, bottom=(RULE, 0.5)):
        c = tbl.cell(i, j)
        c.margin_left = c.margin_right = Inches(0.06)
        c.margin_top = c.margin_bottom = Inches(0.02)
        c.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf = c.text_frame
        for k, para in enumerate(text.split("\n")):
            p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
            p.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER}[aligns[j]]
            _runs(p, para, size, color, bold)
            # An empty cell otherwise takes PowerPoint's 18 pt default and
            # stretches its row.
            p._p.get_or_add_endParaRPr().set("sz", str(round(size * 100)))
        _cell_borders(c, fill=CARD if j == shade else None, bottom=bottom)

    def section(i, text):
        cell(i, 0, text, bold=True, color=MUTED, size=10, bottom=None)
        for j in range(1, len(head)):
            cell(i, j, "", size=10, bottom=None)
        tbl.cell(i, 0).merge(tbl.cell(i, shade - 1))

    for j, h in enumerate(head):
        cell(0, j, h, bold=True, bottom=(INK, 1.0))
    return cell, section


def stats_table(s, N: Numbers, top: float) -> None:
    """The bracketed comparisons among all pairs of the five groups."""
    V6, N6, V24, N24, UN = "6 h Vehicle", "6 h NM72", "24 h Vehicle", "24 h NM72", "Uninjured"

    def means(pop, a, b):
        ma, mb = N.mean(pop, a), N.mean(pop, b)
        return f"{ma:,.0f} vs {mb:,.0f}" if ma > 100 else f"{ma:.2f} vs {mb:.2f}"

    rows = [("section", "Vehicle vs NM72, among all pairwise comparisons")]
    for pop, label in READOUTS:
        for tp in ("6 h", "24 h"):
            rows.append((label, f"{tp}: vehicle vs NM72", pop, f"{tp} Vehicle", f"{tp} NM72"))
    rows.append(("section", "Injury and recovery"))
    label = dict(READOUTS)
    for pop, a, b, text in [
            ("B-1a (IgM+)", UN, V24, "Uninjured vs 24 h vehicle"),
            ("Neutrophils", UN, V6, "Uninjured vs 6 h vehicle"),
            ("Neutrophils", UN, N6, "Uninjured vs 6 h NM72"),
            ("Neutrophils", V6, V24, "6 h vs 24 h, vehicle"),
            ("Neutrophils", N6, N24, "6 h vs 24 h, NM72")]:
        rows.append((label[pop], text, pop, a, b))

    head = ["Readout", "Comparison", "Means", "Welch t\n(uncorrected)", "Tukey",
            "Dunnett's T3\n(default)"]
    cell, section = _table(s, "Statistics table", head, [1.75, 2.85, 2.1, 1.8, 1.8, 1.8],
                           "lllccc", len(rows), top, shade=5)
    for i, row in enumerate(rows, start=1):
        if row[0] == "section":
            section(i, row[1])
            continue
        label, text, pop, a, b = row
        ps = [N.p(pop, a, b, col) for col in ("welch_p", "tukey_p", "dunnett_t3_p")]
        cell(i, 0, label)
        cell(i, 1, text, color=TEXT2)
        cell(i, 2, means(pop, a, b), color=TEXT2)
        for j, pv in enumerate(ps, start=3):
            cell(i, j, _fmt(pv), bold=pv < 0.05, color=INK if pv < 0.05 else TEXT2)


WITHIN_ROWS = 1 + 2 * len(READOUTS)


def within_table(s, N: Numbers, top: float) -> None:
    """Vehicle vs NM72 within each timepoint: the drug brackets on every panel."""
    head = ["Readout", "Timepoint", "Vehicle\nmean (n)", "NM72\nmean (n)", "Change",
            "Welch t\n(uncorrected)", "Welch t\n+ Šidák", "Two-way ANOVA\n+ Šidák (default)"]
    cell, section = _table(s, "Within-timepoint table", head,
                           [1.75, 1.05, 1.45, 1.45, 1.0, 1.55, 1.55, 1.95], "llcccccc",
                           WITHIN_ROWS, top, shade=7)

    def mean_n(pop, g):
        m = N.mean(pop, g)
        return (f"{m:,.0f}" if m > 100 else f"{m:.2f}") + f" ({N.n(pop, g)})"

    section(1, "Vehicle vs NM72 within each timepoint")
    i = 2
    for pop, label in READOUTS:
        for tp in ("6 h", "24 h"):
            r = N.w(pop, tp)
            # One rule per readout, under its 24 h row, so the two timepoints read as a pair.
            rule = (RULE, 0.5) if tp == "24 h" else None
            ch = N.change(pop, tp)
            cell(i, 0, label if tp == "6 h" else "", bottom=rule)
            cell(i, 1, tp, color=TEXT2, bottom=rule)
            cell(i, 2, mean_n(pop, f"{tp} Vehicle"), color=TEXT2, bottom=rule)
            cell(i, 3, mean_n(pop, f"{tp} NM72"), color=TEXT2, bottom=rule)
            cell(i, 4, f"{'+' if ch >= 0 else '−'}{abs(ch):.0f}%", color=TEXT2, bottom=rule)
            for j, col in enumerate(("welch_p", "welch_sidak_p", "anova_sidak_p"), start=5):
                pv = r[col]
                cell(i, j, _fmt(pv), bold=pv < 0.05, color=INK if pv < 0.05 else TEXT2,
                     bottom=rule)
            i += 1


# ---------------------------------------------------------------------------
# The deck
# ---------------------------------------------------------------------------

def build() -> Presentation:
    N = Numbers()
    long, res = N.long, N.res
    names = {L: f"{L}: {sf.markup(lab).replace('^{', '').replace('}', '')}"
             for L, (lab, _p, _y, _b) in sf.BY_LETTER.items()}

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    blank = prs.slide_layouts[6]

    def slide(notes: str = ""):
        s = prs.slides.add_slide(blank)
        if notes:
            s.notes_slide.notes_text_frame.text = notes.strip()
        return s

    def panels(s, letters, size, k, gap):
        pw, ph = size[0] * k, size[1] * k
        cols = 3 if len(letters) == 6 else len(letters)
        rows = (len(letters) + cols - 1) // cols
        x0 = (W - cols * pw - (cols - 1) * gap) / 2
        y0 = (H - rows * ph - (rows - 1) * gap) / 2
        for i, L in enumerate(letters):
            p = sf.render_panel(L, size, "dunnett-t3", res, long)
            r, c = divmod(i, cols)
            add_panel(s, p, x0 + c * (pw + gap), y0 + r * (ph + gap), k, names[L])

    # One paragraph per line, as the notes pane shows them.
    s = slide("Mean ± SEM, each dot one animal. Vehicle vs NM72 within each timepoint: "
              f"{fp.WITHIN_METHOD}. Other brackets: Welch ANOVA + Dunnett's T3 across all five "
              "groups. Brackets show adjusted p, grey where p ≥ 0.05.\n"
              "A-B: B cell panel. C-F: T cell / myeloid panel.")
    panels(s, "ABCDEF", *SUMMARY, gap=0.0)
    panels(slide(), "C", *SINGLE, gap=0.0)
    panels(slide(), "AB", *PAIR, gap=0.15)
    panels(slide(), "DE", *PAIR, gap=0.15)
    panels(slide(), "F", *SINGLE, gap=0.0)
    stats_table(slide(), N, top=(H - table_height(19)) / 2)
    within_table(slide(), N, top=(H - table_height(WITHIN_ROWS)) / 2)
    return prs


def main() -> int:
    prs = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(f"Saved: {OUT} ({len(prs.slides)} slides)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
