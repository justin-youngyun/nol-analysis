#!/usr/bin/env python
"""
The slide deck: the summary panels, and the statistics behind them. Plots only.

    python3 make_slides.py        # -> outputs/slides/NM72_spleen_flow.pptx

Slides: all six panels (A-F), then neutrophils (C), B-1a with its IgM- control
(A-B), Tregs with CD25+CD127- (D-E), MerTK (F), and the table of every planned
comparison under three procedures.

Each panel is one PowerPoint group, so it moves and resizes as a unit. Inside
the group the points, error bars and grid are a picture (PNG, with the SVG
embedded so PowerPoint 365 can Convert to Shape), and everything else is
native: every title, axis label, tick label, n=, p-value and note is a text
box, and every bracket and timepoint line is a line. slide_figures.py draws
the panels with the summary figure's own draw() call and records where each
of those pieces goes.

Speaker notes carry what used to be on the slides, for anyone who asks.
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
from scipy import stats

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

def fmt_p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


class Numbers:
    def __init__(self):
        self.long = pd.read_csv(fp.LONG)
        self.res = {pop: fp._stats(self.long, pop) for _l, pop, _y, _b in fp.PANELS}

    def values(self, pop: str, group: str) -> np.ndarray:
        d = self.long[(self.long.population == pop) & (self.long.group == group)]
        return d["value"].dropna().to_numpy(float)

    def mean(self, pop: str, group: str) -> float:
        return float(self.values(pop, group).mean())

    def n(self, pop: str, group: str) -> int:
        return int(self.values(pop, group).size)

    def p(self, pop: str, a: str, b: str, col: str = "dunnett_t3_p") -> float:
        return float(fp._p(self.res[pop], a, b, col))

    def change(self, pop: str, tp: str) -> float:
        v, d = self.mean(pop, f"{tp} Vehicle"), self.mean(pop, f"{tp} NM72")
        return 100 * (d - v) / v

    def welch_ci(self, pop: str, tp: str) -> tuple[float, float]:
        a, b = self.values(pop, f"{tp} Vehicle"), self.values(pop, f"{tp} NM72")
        va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
        df = (va + vb) ** 2 / (va ** 2 / (a.size - 1) + vb ** 2 / (b.size - 1))
        half = stats.t.ppf(0.975, df) * np.sqrt(va + vb)
        diff = b.mean() - a.mean()
        return 100 * (diff - half) / a.mean(), 100 * (diff + half) / a.mean()

    def leave_one_out(self, pop: str, tp: str) -> tuple[float, float]:
        d = self.long[self.long.population == pop]
        v = d[d.group == f"{tp} Vehicle"].dropna(subset=["value"])
        t = d[d.group == f"{tp} NM72"].dropna(subset=["value"])
        ps = [stats.ttest_ind(v[v.animal != a]["value"], t[t.animal != a]["value"],
                              equal_var=False).pvalue
              for a in list(v.animal) + list(t.animal)]
        return min(ps), max(ps)


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


def add_panel(slide, panel: sf.Panel, left: float, top: float, k: float, name: str):
    """One panel as one group: its picture, then native lines, then native text."""
    group = slide.shapes.add_group_shape()
    group.name = name
    shapes = group.shapes

    pic = shapes.add_picture(str(panel.png), Inches(left), Inches(top),
                             Inches(panel.width * k), Inches(panel.height * k))
    pic.name = f"{name} plot"
    pic._element.nvPicPr.cNvPr.set("descr", f"{name}: points, mean and SEM")
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
        shp.name = "bracket" if len(pts) == 4 else "timepoint line"

    for t in panel.texts:
        size = _size(t.size * k)
        tw = text_width(t.text, size, t.bold)
        cx = left + k * (t.x0 + t.x1) / 2
        cy = top + k * (t.y0 + t.y1) / 2
        backed = t.kind in ("pvalue", "note")
        if t.rotation:
            bw, bh = tw + 0.12, size * 1.2 / 72
            bx, by = cx - bw / 2, cy - bh / 2
        else:
            pad = 0.03 if backed else 0.12
            bw = tw + 2 * pad
            bh = size * 1.2 / 72
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
        p = tf.paragraphs[0]
        p.alignment = {"left": PP_ALIGN.LEFT, "right": PP_ALIGN.RIGHT,
                       "center": PP_ALIGN.CENTER}[t.align if not t.rotation else "center"]
        p.line_spacing = 1.0
        _runs(p, t.text, size, t.color, t.bold)
        if t.rotation:
            box.rotation = 360 - t.rotation
        if backed:
            box.fill.solid()
            box.fill.fore_color.rgb = rgb(WHITE)
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


def stats_table(s, N: Numbers, top: float) -> None:
    V6, N6, V24, N24, UN = "6 h Vehicle", "6 h NM72", "24 h Vehicle", "24 h NM72", "Uninjured"
    readouts = [("B-1a (IgM+)", "B-1a"), ("IgM-", "IgM^{−}"), ("Neutrophils", "Neutrophils"),
                ("Tregs", "Tregs"), ("CD25+CD127-", "CD25^{+}CD127^{−}"),
                ("MerTK Median (M1 Like)", "MerTK")]

    def means(pop, a, b):
        ma, mb = N.mean(pop, a), N.mean(pop, b)
        return f"{ma:,.0f} vs {mb:,.0f}" if ma > 100 else f"{ma:.2f} vs {mb:.2f}"

    rows = [("section", "Vehicle vs NM72")]
    for pop, label in readouts:
        for tp in ("6 h", "24 h"):
            rows.append((label, f"{tp}: vehicle vs NM72", pop, f"{tp} Vehicle", f"{tp} NM72"))
    rows.append(("section", "Injury and recovery"))
    for pop, label, a, b, text in [
            ("B-1a (IgM+)", "B-1a", UN, V24, "Uninjured vs 24 h vehicle"),
            ("Neutrophils", "Neutrophils", UN, V6, "Uninjured vs 6 h vehicle"),
            ("Neutrophils", "Neutrophils", UN, N6, "Uninjured vs 6 h NM72"),
            ("Neutrophils", "Neutrophils", V6, V24, "6 h vs 24 h, vehicle"),
            ("Neutrophils", "Neutrophils", N6, N24, "6 h vs 24 h, NM72")]:
        rows.append((label, text, pop, a, b))

    head = ["Readout", "Comparison", "Means", "Welch t\n(uncorrected)", "Tukey",
            "Dunnett's T3\n(default)"]
    widths = [1.75, 2.85, 2.1, 1.8, 1.8, 1.8]
    n_rows = len(rows) + 1
    row_h = 0.228
    left = (W - sum(widths)) / 2
    gf = s.shapes.add_table(n_rows, len(head), Inches(left), Inches(top), Inches(sum(widths)),
                            Inches(row_h * n_rows))
    gf.name = "Statistics table"
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
    tbl.rows[0].height = Inches(0.46)
    for i in range(1, n_rows):
        tbl.rows[i].height = Inches(row_h)

    def cell(i, j, text, *, bold=False, color=INK, size=10.5, align="l", fill=None,
             bottom=None):
        c = tbl.cell(i, j)
        c.margin_left = c.margin_right = Inches(0.06)
        c.margin_top = c.margin_bottom = Inches(0.02)
        c.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf = c.text_frame
        for k, para in enumerate(text.split("\n")):
            p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
            p.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER}[align]
            _runs(p, para, size, color, bold)
        _cell_borders(c, fill=fill, bottom=bottom)

    for j, h in enumerate(head):
        cell(0, j, h, bold=True, align="l" if j < 3 else "c",
             fill=CARD if j == 5 else None, bottom=(INK, 1.0))
    for i, row in enumerate(rows, start=1):
        if row[0] == "section":
            cell(i, 0, row[1], bold=True, color=MUTED, size=10)
            for j in range(1, len(head)):
                cell(i, j, "", size=10, fill=CARD if j == 5 else None)
            tbl.cell(i, 0).merge(tbl.cell(i, 4))
            continue
        label, text, pop, a, b = row
        ps = [N.p(pop, a, b, col) for col in ("welch_p", "tukey_p", "dunnett_t3_p")]
        cell(i, 0, label, bottom=(RULE, 0.5))
        cell(i, 1, text, color=TEXT2, bottom=(RULE, 0.5))
        cell(i, 2, means(pop, a, b), color=TEXT2, bottom=(RULE, 0.5))
        for j, pv in enumerate(ps, start=3):
            txt = "< 0.001" if pv < 0.001 else f"{pv:.3f}"
            cell(i, j, txt, bold=pv < 0.05, color=INK if pv < 0.05 else TEXT2, align="c",
                 fill=CARD if j == 5 else None, bottom=(RULE, 0.5))


# ---------------------------------------------------------------------------
# The deck
# ---------------------------------------------------------------------------

def build() -> Presentation:
    N = Numbers()
    long, res = N.long, N.res
    V6, N6, V24, N24, UN = "6 h Vehicle", "6 h NM72", "24 h Vehicle", "24 h NM72", "Uninjured"
    NEU, B1A, TREG, CD25, MER = "Neutrophils", "B-1a (IgM+)", "Tregs", "CD25+CD127-", \
        "MerTK Median (M1 Like)"
    names = {L: f"{L}: {sf.markup(lab).replace('^{', '').replace('}', '')}"
             for L, (lab, _p, _y, _b) in sf.BY_LETTER.items()}

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    blank = prs.slide_layouts[6]

    def slide(notes: str):
        s = prs.slides.add_slide(blank)
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

    t3 = "Mean ± SEM, each dot one animal. Welch ANOVA + Dunnett's T3 across all five groups; " \
         "brackets show adjusted p, grey where p ≥ 0.05."
    lo, hi = N.leave_one_out(B1A, "24 h")
    mlo, mhi = N.welch_ci(MER, "24 h")

    s = slide(f"""
{t3}
A-B: B cell panel. C-F: T cell / myeloid panel. 489 and 497 excluded on live-leukocyte yield.
CD3 and CD4 frequencies are left out: they track sample viability (Spearman 0.74 for CD4),
which differs between the 6 h groups. The readouts shown do not (|rho| < 0.42).
""")
    panels(s, "ABCDEF", *SUMMARY, gap=0.0)

    s = slide(f"""
Neutrophils rise {N.mean(NEU, V6) / N.mean(NEU, UN):.1f}x uninjured at 6 h in vehicle
({N.mean(NEU, V6):.2f}% vs {N.mean(NEU, UN):.2f}% of live leukocytes) and are back to
baseline by 24 h. The 6 h -> 24 h fall is {fmt_p(N.p(NEU, V6, V24))} in vehicle after
correction, {fmt_p(N.p(NEU, N6, N24))} in the NM72 arm.
NM72 vs vehicle at 6 h: {N.change(NEU, '6 h'):+.0f}%, {fmt_p(N.p(NEU, V6, N6, 'welch_p'))}
uncorrected, {N.p(NEU, V6, N6):.3f} corrected.
Uninjured is n = {N.n(NEU, UN)}, so the rise itself does not survive correction.
Animal 483's CD3 gate was tightened after first analysis (its neutrophils 3.87% -> 5.67%),
which moved the uncorrected 6 h drug p from 0.26 to 0.048.
""")
    panels(s, "C", *SINGLE, gap=0.0)

    s = slide(f"""
B-1a at 24 h: vehicle {N.mean(B1A, V24):.2f}% of B cells, NM72 {N.mean(B1A, N24):.2f}%,
uninjured {N.mean(B1A, UN):.2f}%. Vehicle vs NM72 at 24 h: {fmt_p(N.p(B1A, V24, N24, 'welch_p'))}
uncorrected, {N.p(B1A, V24, N24):.3f} corrected. The IgM-negative fraction of the same gate
does not move in any group, so the B-1a shift is not a gate artifact.
Vehicle at 24 h is n = {N.n(B1A, V24)} (489 excluded); dropping any one animal moves the
uncorrected p between {lo:.3f} and {hi:.3f}.
""")
    panels(s, "AB", *PAIR, gap=0.15)

    s = slide(f"""
Tregs, NM72 vs vehicle: {N.change(TREG, '6 h'):+.0f}% at 6 h
({fmt_p(N.p(TREG, V6, N6, 'welch_p'))} uncorrected), {N.change(TREG, '24 h'):+.0f}% at 24 h
({fmt_p(N.p(TREG, V24, N24, 'welch_p'))}). CD25+CD127- moves the same way. No comparison
reaches corrected p < 0.05. Uninjured Tregs are n = {N.n(TREG, UN)} (497 excluded, 498 below
the 20-event floor), so there is no baseline.
""")
    panels(s, "DE", *PAIR, gap=0.15)

    s = slide(f"""
MerTK median fluorescence on M1-like red pulp macrophages. NM72 vs vehicle at 24 h:
{N.change(MER, '24 h'):+.1f}%, 95% CI {mlo:+.0f}% to {mhi:+.0f}%, so an effect larger than
about 20% is unlikely. At 6 h the vehicle animals spread widely and the interval is wide.
""")
    panels(s, "F", *SINGLE, gap=0.0)

    s = slide("""
Bold: p < 0.05. Dunnett's T3 is the default because it does not assume equal SDs
(Brown-Forsythe p = 0.048 for MerTK). Tukey pools the SD, which makes comparisons against
the n = 2 uninjured group look stronger. All three agree that no vehicle-vs-NM72
comparison survives correction. All 56 pairwise p-values: final/posthoc_all_methods.csv.
""")
    stats_table(s, N, top=(H - (0.46 + 0.228 * 19)) / 2)
    return prs


def main() -> int:
    prs = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(f"Saved: {OUT} ({len(prs.slides)} slides)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
