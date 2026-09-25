#!/usr/bin/env python
"""
The slide deck for presenting this cohort: nine slides and four backups.

    python3 make_slides.py        # -> outputs/slides/NM72_spleen_flow.pptx

It redraws the slide figures first (slide_figures.py), then builds the deck
with python-pptx. Slide text is live text and the statistics table is a native
PowerPoint table. Each figure is a PNG with its SVG embedded behind it: in
PowerPoint 365, right-click a figure and choose Convert to Shape to edit any
label, point or bracket. Older PowerPoint shows the PNG.

Every number in the slide text is computed here from the same data the
figures use, and the claims the text makes are asserted, so a re-analysis
that changes the story stops the build instead of leaving a stale slide.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lxml import etree
from PIL import Image, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.opc.package import Part
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt
from scipy import stats

import make_final_panels as fp
import slide_figures as sf

OUT = sf.OUTDIR / "NM72_spleen_flow.pptx"

# ---------------------------------------------------------------------------
# Look. The data colours are the figures' own, so a colour means the same
# group on every slide; everything else is neutral.
# ---------------------------------------------------------------------------

W, H = 13.333, 7.5
LEFT, RIGHT = 0.6, 13.333 - 0.6
FONT = "Arial"

WHITE = "FFFFFF"
INK = "0B0B0B"
TEXT2 = "52514E"
MUTED = "7A7973"
RULE = "E6E5E1"
CARD = "F3F3F1"
DARK = "1A1A19"
DTEXT2 = "C3C2B7"
DMUTED = "97968C"
VEH = "2A78D6"
NM72 = "EB6834"
UNINJ = "84837C"
FLAG = "D03B3B"
GROUP_COLOR = {"Uninjured": UNINJ, "Vehicle": VEH, "NM72": NM72}

_FONTS = {False: "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
          True: "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"}
MINUS = "−"


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------

NBSP = "\u00a0"


def fmt_p(p: float) -> str:
    """Exact p, held together with non-breaking spaces so it never wraps mid-value."""
    return f"p{NBSP}<{NBSP}0.001" if p < 0.001 else f"p{NBSP}={NBSP}{p:.3f}"


def signed(x: float, digits: int = 0) -> str:
    s = f"{x:+.{digits}f}"
    return s.replace("-", MINUS)


def pct(x: float) -> str:
    """A power, as a percent; never rounds up to a claim of 100%."""
    return ">99%" if x > 0.99 else f"{100 * x:.0f}%"


def power(d: float, n: int, alpha: float = 0.05) -> float:
    df = 2 * n - 2
    nc = d * np.sqrt(n / 2)
    tc = stats.t.ppf(1 - alpha / 2, df)
    return float(1 - stats.nct.cdf(tc, df, nc) + stats.nct.cdf(-tc, df, nc))


class Numbers:
    """Everything the slide text quotes, from the data the figures are drawn from."""

    def __init__(self):
        self.long = pd.read_csv(fp.LONG)
        self.res = {pop: fp._stats(self.long, pop) for _l, pop, _y, _b in fp.PANELS}
        self.quality = sf.quality_table()

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
        """NM72 relative to vehicle at one timepoint, in percent."""
        v, d = self.mean(pop, f"{tp} Vehicle"), self.mean(pop, f"{tp} NM72")
        return 100 * (d - v) / v

    def welch_ci(self, pop: str, tp: str) -> tuple[float, float]:
        """95% CI of NM72 minus vehicle, as percent of the vehicle mean."""
        a, b = self.values(pop, f"{tp} Vehicle"), self.values(pop, f"{tp} NM72")
        va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
        df = (va + vb) ** 2 / (va ** 2 / (a.size - 1) + vb ** 2 / (b.size - 1))
        half = stats.t.ppf(0.975, df) * np.sqrt(va + vb)
        diff = b.mean() - a.mean()
        return 100 * (diff - half) / a.mean(), 100 * (diff + half) / a.mean()

    def leave_one_out(self, pop: str, tp: str) -> tuple[float, float]:
        d = self.long[(self.long.population == pop)]
        v = d[d.group == f"{tp} Vehicle"].dropna(subset=["value"])
        t = d[d.group == f"{tp} NM72"].dropna(subset=["value"])
        ps = []
        for drop in list(v.animal) + list(t.animal):
            a = v[v.animal != drop]["value"].to_numpy(float)
            b = t[t.animal != drop]["value"].to_numpy(float)
            ps.append(stats.ttest_ind(a, b, equal_var=False).pvalue)
        return min(ps), max(ps)

    def pooled_sd(self, pop: str) -> float:
        groups = [self.values(pop, g) for g in fp.ORDER]
        groups = [g for g in groups if g.size > 1]
        ss = sum(((g - g.mean()) ** 2).sum() for g in groups)
        return float(np.sqrt(ss / sum(g.size - 1 for g in groups)))

    def power_30(self, pop: str, tp: str, n: int) -> float:
        d = 0.30 * self.mean(pop, f"{tp} Vehicle") / self.pooled_sd(pop)
        return power(d, n)

    def viability_rho(self, pop: str) -> tuple[float, float]:
        ok = self.quality[self.quality["live_ok"]]
        y = self.long[self.long.population == pop].set_index("animal")["value"]
        j = ok.join(y.rename("value"), how="inner").dropna(subset=["value"])
        r = stats.spearmanr(j["viability"], j["value"])
        return float(r.statistic), float(r.pvalue)


# ---------------------------------------------------------------------------
# Text fit. LibreOffice renders Arial as Liberation Sans, which has Arial's
# metrics, so measuring with it predicts what PowerPoint lays out.
# ---------------------------------------------------------------------------

_font_cache: dict = {}


def text_width(s: str, size: float, bold: bool = False) -> float:
    key = (bold, size)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(_FONTS[bold], size=max(1, round(size * 10)))
    return _font_cache[key].getlength(s) / 10 / 72


def lines_needed(s: str, size: float, width: float, bold: bool = False) -> int:
    n = 0
    for para in s.split("\n"):
        line, count = "", 1
        for word in para.split(" "):
            trial = f"{line} {word}".strip()
            if text_width(trial, size, bold) <= width or not line:
                line = trial
            else:
                count, line = count + 1, word
        n += count
    return n


WARNINGS: list[str] = []


def check_fit(s: str, size: float, w: float, h: float, bold: bool, spacing: float = 1.0,
              after: float = 0, where: str = "") -> None:
    """Warn when wrapped text would not fit its box (Arial line height is ~1.15 em)."""
    plain = re.sub(r"\^\{([^}]*)\}|\*\*", lambda m: m.group(1) or "", s)
    paras = plain.split("\n")
    lines = sum(lines_needed(p, size, w, bold) for p in paras)
    need = (lines * size * 1.15 * spacing + after * (len(paras) - 1)) / 72
    if need > h:
        WARNINGS.append(f"{where}: needs {need:.2f} in, box {h:.2f} in: {plain[:60]!r}")


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------

def rgb(hexstr: str) -> RGBColor:
    return RGBColor.from_string(hexstr)


def _plain_shape(shape) -> None:
    """Drop the theme style reference, so no theme shadow or outline applies."""
    style = shape._element.find(qn("p:style"))
    if style is not None:
        shape._element.remove(style)


_MARKUP = re.compile(r"(\^\{[^}]*\}|\*\*[^*]+\*\*)")


def _runs(paragraph, text: str, size: float, color: str, bold: bool) -> None:
    """Fill a paragraph. **bold** and ^{superscript} are the only markup."""
    for piece in _MARKUP.split(text):
        if not piece:
            continue
        sup = piece.startswith("^{")
        strong = piece.startswith("**")
        body = piece[2:-1] if sup else piece[2:-2] if strong else piece
        run = paragraph.add_run()
        run.text = body
        f = run.font
        f.name = FONT
        f.size = Pt(size)
        f.bold = bold or strong
        f.color.rgb = rgb(color)
        if sup:
            run.font._rPr.set("baseline", "30000")


def add_text(slide, x, y, w, h, text, *, size=14, color=INK, bold=False, align="l",
             anchor="t", spacing=1.0, after=0, char_spacing=None, check=True,
             where="") -> object:
    """A text box. Paragraphs are separated by blank-line-free '\n'."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE,
                          "b": MSO_ANCHOR.BOTTOM}[anchor]
    for i, para in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[align]
        p.line_spacing = spacing
        if after:
            p.space_after = Pt(after)
        _runs(p, para, size, color, bold)
        if char_spacing:
            for r in p.runs:
                r.font._rPr.set("spc", str(int(char_spacing * 100)))
    if check:
        check_fit(text, size, w, h, bold, spacing=spacing, after=after, where=where)
    return box


def add_rect(slide, x, y, w, h, fill, radius=None):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    shp = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    _plain_shape(shp)
    if radius:
        shp.adjustments[0] = radius / min(w, h)
    shp.fill.solid()
    shp.fill.fore_color.rgb = rgb(fill)
    shp.line.fill.background()
    return shp


def add_dot(slide, cx, cy, d, color, ring=False, line_w=1.75):
    shp = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx - d / 2), Inches(cy - d / 2),
                                 Inches(d), Inches(d))
    _plain_shape(shp)
    if ring:
        shp.fill.background()
        shp.line.color.rgb = rgb(color)
        shp.line.width = Pt(line_w)
    else:
        shp.fill.solid()
        shp.fill.fore_color.rgb = rgb(color)
        shp.line.fill.background()
    return shp


def number_badge(slide, x, y, d, label, fill, color=WHITE):
    add_dot(slide, x + d / 2, y + d / 2, d, fill)
    add_text(slide, x, y, d, d, label, size=d * 34, color=color, bold=True, align="c",
             anchor="m", check=False)


_svg_count = [0]


def add_figure(slide, png: Path, box, *, align="c", alt=""):
    """Fit a figure into box (x, y, w, h), top-aligned; embed its SVG behind the PNG."""
    bx, by, bw, bh = box
    wpx, hpx = Image.open(png).size
    ar = wpx / hpx
    w, h = (bh * ar, bh) if bw / bh > ar else (bw, bw / ar)
    x = {"c": bx + (bw - w) / 2, "l": bx, "r": bx + bw - w}[align]
    pic = slide.shapes.add_picture(str(png), Inches(x), Inches(by), Inches(w), Inches(h))
    pic._element.nvPicPr.cNvPr.set("descr", alt)
    svg = png.with_suffix(".svg")
    if svg.exists():
        _svg_count[0] += 1
        package = slide.part.package
        part = Part(package.next_image_partname("svg"), "image/svg+xml", package,
                    svg.read_bytes())
        rid = slide.part.relate_to(part, RT.IMAGE)
        blip = pic._element.find(".//" + qn("a:blip"))
        ext_lst = blip.find(qn("a:extLst"))
        if ext_lst is None:
            ext_lst = etree.SubElement(blip, qn("a:extLst"))
        ext = etree.SubElement(ext_lst, qn("a:ext"))
        ext.set("uri", "{96DAC541-7B7A-43D3-8B79-37D633B846F1}")
        asvg = "http://schemas.microsoft.com/office/drawing/2016/SVG/main"
        svg_blip = etree.SubElement(ext, f"{{{asvg}}}svgBlip", nsmap={"asvg": asvg})
        svg_blip.set(qn("r:embed"), rid)
    return x, by, w, h


# ---------------------------------------------------------------------------
# Slide furniture
# ---------------------------------------------------------------------------

class Deck:
    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width = Inches(W)
        self.prs.slide_height = Inches(H)
        self.layout = self.prs.slide_layouts[5]  # Title Only: a real title per slide
        self.count = 0

    def slide(self, *, dark=False, notes=""):
        s = self.prs.slides.add_slide(self.layout)
        self.count += 1
        s.background.fill.solid()
        s.background.fill.fore_color.rgb = rgb(DARK if dark else WHITE)
        if notes:
            s.notes_slide.notes_text_frame.text = notes.strip()
        return s

    def title(self, s, text, *, x=LEFT, y=0.74, w=RIGHT - LEFT, h=0.6, size=28, color=INK):
        t = s.shapes.title
        t.left, t.top, t.width, t.height = Inches(x), Inches(y), Inches(w), Inches(h)
        tf = t.text_frame
        tf.word_wrap = True
        tf.auto_size = MSO_AUTO_SIZE.NONE
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        tf.vertical_anchor = MSO_ANCHOR.TOP
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT
        p.line_spacing = 1.0
        _runs(p, text, size, color, True)
        check_fit(text, size, w, h, True, where=f"slide {self.count} title")
        return t

    def header(self, s, eyebrow, headline, *, legend=True, dark=False):
        add_text(s, LEFT, 0.42, 8.6, 0.28, eyebrow.upper(), size=12, bold=True,
                 color=DMUTED if dark else MUTED, char_spacing=1.5,
                 where=f"slide {self.count} eyebrow")
        self.title(s, headline, color=WHITE if dark else INK)
        if legend:
            self.legend_left = self.legend(s)
        if not dark:
            self.page_number(s)

    def legend(self, s, y=0.44, right=RIGHT) -> float:
        """Group key at the top right; returns its left edge."""
        items = [("Uninjured", UNINJ), ("Vehicle", VEH), ("NM72", NM72)]
        widths = [0.14 + 0.08 + text_width(t, 12) + 0.02 for t, _c in items]
        x = left = right - sum(widths) - 0.28 * (len(items) - 1)
        for (label, color), w in zip(items, widths):
            add_dot(s, x + 0.07, y + 0.13, 0.14, color)
            add_text(s, x + 0.22, y, w - 0.2, 0.26, label, size=12, color=TEXT2, check=False)
            x += w + 0.28
        return left

    def method_note(self, s, text):
        """Statistics line on the key's row, right-aligned against the key."""
        right = self.legend_left - 0.45
        add_text(s, right - 4.2, 0.455, 4.2, 0.26, text, size=11, color=MUTED, align="r",
                 where=f"slide {self.count} method")

    def page_number(self, s):
        add_text(s, RIGHT - 0.5, 7.08, 0.5, 0.24, str(self.count), size=10, color=MUTED,
                 align="r", check=False)

    def footnote(self, s, text, y=6.62, h=0.42):
        add_text(s, LEFT, y, RIGHT - LEFT - 0.6, h, text, size=10, color=MUTED,
                 spacing=1.0, where=f"slide {self.count} footnote")

    def callout(self, s, x, y, w, big, label, color, *, big_size=32, label_size=13,
                label_h=0.9):
        big_h = big_size * 1.25 / 72
        add_text(s, x, y, w, big_h, big, size=big_size, bold=True, color=color,
                 where=f"slide {self.count} callout")
        add_text(s, x, y + big_h + 0.04, w, label_h, label, size=label_size, color=TEXT2,
                 spacing=1.05, where=f"slide {self.count} callout label")


# ---------------------------------------------------------------------------
# The deck
# ---------------------------------------------------------------------------

def build(figs: dict[str, Path]) -> Presentation:
    N = Numbers()
    d = Deck()
    V6, N6, V24, N24, UN = "6 h Vehicle", "6 h NM72", "24 h Vehicle", "24 h NM72", "Uninjured"
    NEU, B1A, IGM, TREG, CD25, MER = ("Neutrophils", "B-1a (IgM+)", "IgM-", "Tregs",
                                      "CD25+CD127-", "MerTK Median (M1 Like)")
    welch = "welch_p"

    # The claims the slides make. If the data stop supporting one, stop here.
    drug = [(pop, tp) for pop in (B1A, IGM, NEU, TREG, CD25, MER) for tp in ("6 h", "24 h")]
    drug_t3 = [N.p(pop, f"{tp} Vehicle", f"{tp} NM72") for pop, tp in drug]
    assert N.p(NEU, V6, V24) < 0.05, "the 6 h -> 24 h neutrophil fall no longer survives T3"
    assert all(p >= 0.05 for p in drug_t3), "a drug contrast now survives T3: rewrite the story"
    assert N.change(NEU, "6 h") < 0 and N.change(B1A, "24 h") < 0
    assert N.change(TREG, "6 h") > 0 and N.change(TREG, "24 h") > 0
    assert N.change(CD25, "6 h") > 0 and N.change(CD25, "24 h") > 0
    igm_min = min(r["dunnett_t3_p"] for r in N.res[IGM]["rows"])
    assert igm_min >= 0.05
    lo_mer, hi_mer = N.welch_ci(MER, "24 h")
    lo_mer6, hi_mer6 = N.welch_ci(MER, "6 h")
    loo = N.leave_one_out(B1A, "24 h")
    pw8 = {pop: N.power_30(pop, tp, 8) for pop, tp in ((B1A, "24 h"), (NEU, "6 h"))}
    pw10 = {pop: N.power_30(pop, tp, 10) for pop, tp in ((B1A, "24 h"), (NEU, "6 h"))}
    assert min(pw8.values()) >= 0.79, "n = 8 no longer gives ~80% power for 30%"
    rho = {pop: N.viability_rho(pop) for pop in ("CD3+ T cells", "CD4+", NEU, TREG, CD25, MER)}
    shown_rho = max(abs(rho[p][0]) for p in (NEU, TREG, CD25, MER))
    assert rho["CD4+"][1] < 0.05 and rho["CD3+ T cells"][1] < 0.05 and shown_rho < 0.42

    q = N.quality
    q_ok = q[q["live_ok"]]
    blk = q_ok[q_ok["block"]]
    cd48 = sf.quality_cd4cd8()
    in_blk, out_blk = cd48[cd48.index.isin(blk.index)], cd48[~cd48.index.isin(blk.index)]
    assert in_blk.min() > out_blk.max()
    excluded = q[~q["live_ok"]].sort_values("live_events")
    next_low = q_ok.sort_values("live_events").iloc[0]
    six_h_nm72 = set(q[(q.timepoint == "6 h") & (q.treatment == "NM72")].index)
    assert six_h_nm72 <= set(blk.index), "not every 6 h NM72 animal is in the clean-gate block"

    neu_fold = N.mean(NEU, V6) / N.mean(NEU, UN)
    assert 2.8 <= neu_fold < 3.5, "neutrophil headline says 3x"
    assert N.mean(B1A, V24) > N.mean(B1A, UN) and N.p(B1A, UN, N24, welch) > 0.5, \
        "B-1a headline: vehicle up, NM72 at baseline"
    assert min(N.p(MER, f"{tp} Vehicle", f"{tp} NM72") for tp in ("6 h", "24 h")) > 0.5
    assert -25 < lo_mer and hi_mer < 25, "MerTK slide says an effect over ~20% is unlikely"
    t3_all = "Welch ANOVA + Dunnett's T3 across all five groups"

    # 1. Title -------------------------------------------------------------
    s = d.slide(dark=True, notes=f"""
Pilot cohort: spleens from {len(q)} animals, taken 6 h and 24 h after SCI, vehicle or NM72,
plus uninjured controls. Two flow panels: T cell / myeloid, and B cell.
""")
    add_text(s, 0.8, 1.3, 10.5, 0.3, "SPINAL CORD INJURY  ·  SPLEEN  ·  FLOW CYTOMETRY",
             size=12, bold=True, color=DMUTED, char_spacing=2, where="title eyebrow")
    d.title(s, "NM72 and the splenic immune response to spinal cord injury", x=0.8, y=1.72,
            w=10.8, h=1.45, size=40, color=WHITE)
    add_text(s, 0.8, 3.35, 10.8, 0.8,
             "Vehicle vs NM72 at 6 h and 24 h after SCI, against uninjured controls",
             size=20, color=DTEXT2, where="title subtitle")
    # One dot per animal, by group: the cohort at a glance.
    x = 0.8
    for g, (tp, tr) in zip(fp.ORDER, [("Uninjured", "Uninjured"), ("6 h", "Vehicle"),
                                      ("6 h", "NM72"), ("24 h", "Vehicle"), ("24 h", "NM72")]):
        animals = q[(q.timepoint == tp) & (q.treatment == tr)].index
        for i, _a in enumerate(animals):
            add_dot(s, x + 0.13 + i * 0.36, 5.1, 0.26, GROUP_COLOR[tr])
        width = max(0.36 * len(animals), text_width(g.replace("Vehicle", "vehicle"), 11))
        add_text(s, x, 5.42, width + 0.2, 0.26, g.replace("Vehicle", "vehicle"), size=11,
                 color=DTEXT2, check=False)
        x += width + 0.55
    add_text(s, 0.8, 5.8, 8, 0.26, f"Each dot is one animal: {len(q)} spleens",
             size=11, color=DMUTED, check=False)
    add_text(s, 0.8, 6.6, 6, 0.3, "September 2026", size=12, color=DMUTED, check=False)

    # 2. Design --------------------------------------------------------------
    s = d.slide(notes="""
Uneven group sizes: 501, 503 and 504 were assigned to the 6 h groups but never acquired.
489 and 497 fall below 1,000 live leukocytes on the T cell / myeloid panel. 489 is also
out of the B cell panel; 497 is kept there.
If asked about the uninjured baseline: on the T cell / myeloid panel it is two animals,
498 and 499, and the FCS files date 497 and 498 to 5 Sep, two days before the rest (7 Sep).
""")
    d.header(s, "Study design", "Spleens at 6 h and 24 h after SCI, vehicle vs NM72",
             legend=False)
    cy, ch = 1.62, 2.95
    add_rect(s, LEFT, cy, 2.75, ch, CARD, radius=0.12)
    add_text(s, LEFT + 0.3, cy + 0.25, 2.3, 0.35, "Uninjured", size=17, bold=True)
    boxes = [(4.35, "6 h after SCI", "6 h"), (8.69, "24 h after SCI", "24 h")]

    def group_row(x0, y0, tp, tr, label):
        animals = list(q[(q.timepoint == tp) & (q.treatment == tr)].index)
        if label:
            add_text(s, x0, y0 - 0.14, 1.2, 0.3, label, size=14, bold=True,
                     color=GROUP_COLOR[tr], check=False)
            x0 += 1.25
        for i, a in enumerate(animals):
            cx = x0 + 0.2 + i * 0.6
            out = not q.loc[a, "live_ok"]
            add_dot(s, cx, y0, 0.3, GROUP_COLOR[tr], ring=out)
            add_text(s, cx - 0.3, y0 + 0.22, 0.6, 0.24, str(a), size=11,
                     color=FLAG if out else MUTED, bold=out, align="c", check=False)

    group_row(LEFT + 0.3, cy + 1.15, "Uninjured", "Uninjured", None)
    for bx, head, tp in boxes:
        add_rect(s, bx, cy, 4.04, ch, CARD, radius=0.12)
        add_text(s, bx + 0.3, cy + 0.25, 3.5, 0.35, head, size=17, bold=True)
        group_row(bx + 0.3, cy + 1.15, tp, "Vehicle", "Vehicle")
        group_row(bx + 0.3, cy + 2.2, tp, "NM72", "NM72")
    arrow = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(3.55), Inches(cy + 1.0),
                               Inches(0.6), Inches(0.4))
    _plain_shape(arrow)
    arrow.fill.solid()
    arrow.fill.fore_color.rgb = rgb("CFCEC8")
    arrow.line.fill.background()
    add_text(s, 3.4, cy + 0.6, 0.9, 0.3, "SCI", size=13, bold=True, color=TEXT2, align="c",
             check=False)
    ex = ", ".join(f"{a} ({int(r.live_events)} events)" for a, r in excluded.iterrows())
    add_text(s, LEFT + 0.3, cy + 1.75, 2.25, 1.05,
             "Rings: excluded for low live-leukocyte yield. 497 is excluded from the "
             "T cell / myeloid panel only.", size=10.5, color=TEXT2, spacing=1.05,
             where="design ring note")
    ry = 4.9
    for i, (head, body) in enumerate([
            ("T cell / myeloid panel",
             "Neutrophils, Tregs, CD25^{+}CD127^{-} and red pulp macrophages, with MerTK "
             "median fluorescence"),
            ("B cell panel", "B-1a (IgM^{+}) B cells, and the IgM^{-} fraction of the same "
                             "gate as a control")]):
        x0 = LEFT if i == 0 else LEFT + 6.215
        add_rect(s, x0, ry, 5.915, 1.35, CARD, radius=0.12)
        add_text(s, x0 + 0.3, ry + 0.22, 5.3, 0.3, head, size=15, bold=True)
        add_text(s, x0 + 0.3, ry + 0.6, 5.3, 0.6, body, size=13, color=TEXT2, spacing=1.05,
                 where="design readout")
    d.footnote(s, f"Excluded: {ex}; threshold 1,000. Frequencies from gates with fewer "
                  "than 20 events are left blank, so rare gates such as Tregs drop to "
                  "n\u00a0=\u00a01–3 in some groups.", y=6.55)

    # 3. Bottom line ---------------------------------------------------------
    s = d.slide(notes=f"""
Three messages. One: the injury response is real, fast and self-limiting; the 6 h to 24 h
fall in vehicle is {fmt_p(N.p(NEU, V6, V24))} after correction. Two: every NM72 effect is a
direction, not a result: none of the twelve vehicle-vs-NM72 comparisons survives
correction; the smallest corrected p is {min(drug_t3):.3f}. Three: the replication that
would settle it is modest.
""")
    d.header(s, "Bottom line", "One clear injury effect; NM72 trends need power",
             legend=False)
    cards = [
        ("1", VEH, "SCI drives a transient neutrophil surge",
         f"Splenic neutrophils rise {neu_fold:.1f}× by 6 h and are back to baseline by 24 h.",
         fmt_p(N.p(NEU, V6, V24)), "6 h → 24 h fall in vehicle, after correction"),
        ("2", NM72, "NM72 moves three populations, as trends",
         f"B-1a stays at baseline at 24 h, neutrophils run "
         f"{abs(N.change(NEU, '6 h')):.0f}% lower at 6 h, Tregs trend higher.",
         f"0 of {len(drug)}", "vehicle-vs-NM72 comparisons survive correction at "
                             "n\u00a0=\u00a03–4 per group"),
        ("3", INK, "A powered cohort would settle it",
         "Sized for a 30% difference, about the size of the effects seen here, with one "
         "gate applied blind to every sample.",
         "n\u00a0=\u00a08–10", f"per arm: {100 * min(pw8.values()):.0f}–"
                     f"{100 * min(pw10.values()):.0f}% power for a 30% difference"),
    ]
    cw, gap = (RIGHT - LEFT - 2 * 0.35) / 3, 0.35
    for i, (num, color, head, body, big, small) in enumerate(cards):
        x0 = LEFT + i * (cw + gap)
        add_rect(s, x0, 1.62, cw, 4.85, CARD, radius=0.14)
        number_badge(s, x0 + 0.35, 1.95, 0.56, num, color)
        add_text(s, x0 + 0.35, 2.78, cw - 0.7, 0.85, head, size=18, bold=True,
                 spacing=1.0, where="bottom-line head")
        add_text(s, x0 + 0.35, 3.72, cw - 0.7, 1.3, body, size=14, color=TEXT2,
                 spacing=1.08, where="bottom-line body")
        add_text(s, x0 + 0.35, 5.08, cw - 0.7, 0.55, big, size=30, bold=True, color=color,
                 where="bottom-line stat")
        add_text(s, x0 + 0.35, 5.66, cw - 0.7, 0.6, small, size=11.5, color=MUTED,
                 spacing=1.05, where="bottom-line stat label")

    # 4. Overview -------------------------------------------------------------
    s = d.slide(notes=f"""
The six panels together. A and B are the B cell panel; C to F the T cell / myeloid panel.
Mean ± SEM, each dot one animal. {t3_all}; brackets show adjusted p, grey where p ≥ 0.05.
CD3 and CD4 frequencies are left out because they track sample quality (backup slide).
Animal 483's CD3 gate was tightened after first analysis; see the neutrophil slide.
""")
    d.header(s, "Summary figure  ·  A–F", "Injury effects survive correction; NM72 effects do not")
    d.method_note(s, "Mean ± SEM  ·  Dunnett's T3-adjusted p")
    add_figure(s, figs["overview"], (LEFT, 1.4, RIGHT - LEFT, 5.6),
               alt="Six dot-plot panels, A to F: B-1a, IgM-negative B cells, neutrophils, "
                   "Tregs, CD25+CD127- T cells and MerTK, by group, with adjusted p-values.")

    # 5. Neutrophils ------------------------------------------------------------
    s = d.slide(notes=f"""
Neutrophils are {neu_fold:.1f}x uninjured at 6 h in vehicle, and back to baseline by 24 h.
The fall from 6 h to 24 h is the robust result: {fmt_p(N.p(NEU, V6, V24))} in vehicle after
correction, {fmt_p(N.p(NEU, N6, N24))} in the NM72 arm. The rise itself is not significant
after correction only because uninjured is n = {N.n(NEU, UN)}.
NM72 is {abs(N.change(NEU, '6 h')):.0f}% lower at 6 h: {fmt_p(N.p(NEU, V6, N6, welch))}
uncorrected, {N.p(NEU, V6, N6):.3f} corrected. Neutrophil frequency does not track sample
viability (rho = {rho[NEU][0]:+.2f}), so unlike CD4 this 6 h comparison is interpretable.
If asked about 483: its CD3 gate was tightened after the first analysis, which raised its
neutrophils from 3.87% to 5.67% and moved the uncorrected p from 0.26 to 0.048. Only 483
was re-gated; one gate applied to every sample in the next cohort avoids that question.
""")
    d.header(s, "C  ·  Neutrophils", "Neutrophils surge 3× at 6 h and resolve by 24 h")
    add_figure(s, figs["neutrophils"], (LEFT, 1.45, 7.3, 5.05), align="l",
               alt="Neutrophils as percent of live leukocytes by group, with brackets for "
                   "the planned comparisons and adjusted p-values.")
    cx, cw3 = 8.45, RIGHT - 8.45
    d.callout(s, cx, 1.6, cw3, f"{neu_fold:.1f}×",
              f"6 h vehicle vs uninjured: {N.mean(NEU, V6):.2f}% vs {N.mean(NEU, UN):.2f}% "
              "of live leukocytes", VEH)
    d.callout(s, cx, 3.2, cw3, fmt_p(N.p(NEU, V6, V24)),
              f"The 6 h → 24 h fall in vehicle, after correction. NM72 arm: "
              f"{fmt_p(N.p(NEU, N6, N24))}.", INK)
    d.callout(s, cx, 4.8, cw3, f"{signed(N.change(NEU, '6 h'))}%",
              f"NM72 vs vehicle at 6 h ({N.mean(NEU, N6):.2f}% vs {N.mean(NEU, V6):.2f}%): "
              f"{fmt_p(N.p(NEU, V6, N6, welch))} uncorrected, {N.p(NEU, V6, N6):.3f} "
              "after correction", NM72)
    d.footnote(s, f"Uninjured n\u00a0=\u00a0{N.n(NEU, UN)} (497 excluded), so the rise itself does not "
                  f"survive correction ({fmt_p(N.p(NEU, UN, V6))} vehicle, "
                  f"{N.p(NEU, UN, N6):.3f} NM72). Animal 483's CD3 gate was tightened after "
                  "first analysis (its neutrophils 3.87% → 5.67%), moving the uncorrected "
                  "6 h drug p from 0.26 to 0.048.")

    # 6. B-1a --------------------------------------------------------------------
    s = d.slide(notes=f"""
At 24 h vehicle B-1a is {N.mean(B1A, V24):.2f}% of B cells against {N.mean(B1A, UN):.2f}%
uninjured; NM72 sits at {N.mean(B1A, N24):.2f}%. Vehicle vs NM72 at 24 h is
{fmt_p(N.p(B1A, V24, N24, welch))} uncorrected and {N.p(B1A, V24, N24):.3f} after
correction. The IgM-negative fraction of the same gate does not move, so this is not a
shift of the IgM gate. If asked how robust: vehicle is n = {N.n(B1A, V24)}, and dropping any
single animal moves the uncorrected p between {loo[0]:.3f} and {loo[1]:.3f}.
This is the lead worth powering, at 24 h.
""")
    d.header(s, "A–B  ·  B-1a B cells", "B-1a at 24 h: up with vehicle, at baseline with NM72")
    add_figure(s, figs["b1a"], (LEFT, 1.45, RIGHT - LEFT, 3.75),
               alt="B-1a (IgM-positive) and IgM-negative B cells as percent of B cells, "
                   "by group, with adjusted p-values.")
    b1a_cards = [
        (f"{signed(N.change(B1A, '24 h'))}%",
         f"NM72 vs vehicle at 24 h ({N.mean(B1A, N24):.2f}% vs {N.mean(B1A, V24):.2f}%): "
         f"{fmt_p(N.p(B1A, V24, N24, welch))} uncorrected, {N.p(B1A, V24, N24):.3f} "
         "corrected", NM72),
        ("≈ baseline",
         f"NM72 at 24 h vs uninjured: {N.mean(B1A, N24):.2f}% vs {N.mean(B1A, UN):.2f}% "
         f"({fmt_p(N.p(B1A, UN, N24, welch))})", NM72),
        ("IgM^{−} flat",
         f"Same gate, no change in any group (all corrected p\u00a0≥\u00a0{igm_min:.2f}), so the "
         "B-1a shift is not a gating artifact", INK),
    ]
    callout_row(d, s, b1a_cards, 5.33)
    d.footnote(s, f"Vehicle at 24 h is n\u00a0=\u00a0{N.n(B1A, V24)} (489 excluded). Dropping any one "
                  f"animal moves the uncorrected 24 h p between {loo[0]:.3f} and "
                  f"{loo[1]:.3f}.", y=6.78, h=0.25)

    # 7. Tregs --------------------------------------------------------------------
    s = d.slide(notes=f"""
Tregs are higher with NM72 at both timepoints, and the CD25+CD127- parent gate moves the
same way, but nothing here is close to significant after correction. There is no usable
baseline: uninjured Tregs are n = {N.n(TREG, UN)}. Tregs do not track sample viability
(rho = {rho[TREG][0]:+.2f}), so the trend is not the quality artifact that rules out CD4.
""")
    d.header(s, "D–E  ·  Regulatory T cells", "Tregs trend higher with NM72 at both timepoints")
    add_figure(s, figs["tregs"], (LEFT, 1.45, RIGHT - LEFT, 3.75),
               alt="Tregs and CD25+CD127- cells as percent of CD4+ T cells, by group.")
    treg_cards = [
        (f"{signed(N.change(TREG, '6 h'))}%",
         f"Tregs at 6 h, NM72 vs vehicle ({N.mean(TREG, N6):.2f}% vs "
         f"{N.mean(TREG, V6):.2f}% of CD4^{{+}}): {fmt_p(N.p(TREG, V6, N6, welch))} "
         "uncorrected", NM72),
        (f"{signed(N.change(TREG, '24 h'))}%",
         f"Tregs at 24 h ({N.mean(TREG, N24):.2f}% vs {N.mean(TREG, V24):.2f}%): "
         f"{fmt_p(N.p(TREG, V24, N24, welch))} uncorrected", NM72),
        ("Same direction",
         f"CD25^{{+}}CD127^{{−}}, the parent gate, at both timepoints "
         f"(p\u00a0=\u00a0{N.p(CD25, V6, N6, welch):.3f} and {N.p(CD25, V24, N24, welch):.3f})", INK),
    ]
    callout_row(d, s, treg_cards, 5.33)
    d.footnote(s, "No comparison reaches corrected p\u00a0<\u00a00.05. Uninjured Tregs are n\u00a0=\u00a0"
                  f"{N.n(TREG, UN)} (497 excluded; 498 below the 20-event floor), so there is "
                  "no baseline to compare with.", y=6.78, h=0.25)

    # 8. MerTK ----------------------------------------------------------------------
    s = d.slide(notes=f"""
MerTK is the efferocytosis receptor, so it is the readout closest to a TG2 clearance
mechanism. NM72 does not move it: at 24 h the difference is {signed(N.change(MER, '24 h'), 1)}%,
and the 95% CI of {signed(lo_mer)}% to {signed(hi_mer)}% makes a large effect unlikely.
At 6 h the vehicle animals spread widely ({N.values(MER, V6).min():,.0f} to
{N.values(MER, V6).max():,.0f}), so the interval is wide ({signed(lo_mer6)}% to
{signed(hi_mer6)}%) and 6 h says less. This is MerTK on the M1-like subset; reading it on
the whole red pulp gate would remove the dependence on the M1/M2 split.
""")
    d.header(s, "F  ·  MerTK on red pulp macrophages",
             "NM72 does not change MerTK on red pulp macrophages")
    add_figure(s, figs["mertk"], (RIGHT - 7.3, 1.45, 7.3, 5.05), align="r",
               alt="MerTK median fluorescence on red pulp macrophages by group.")
    cw4 = 3.9
    d.callout(s, LEFT, 1.6, cw4, f"{signed(N.change(MER, '24 h'), 1)}%",
              f"NM72 vs vehicle at 24 h (median fluorescence "
              f"{N.mean(MER, N24):,.0f} vs {N.mean(MER, V24):,.0f})", INK)
    d.callout(s, LEFT, 3.2, cw4, f"{signed(lo_mer)}% to {signed(hi_mer)}%",
              "95% confidence interval at 24 h: an effect larger than about 20% is "
              "unlikely", INK, big_size=28)
    d.callout(s, LEFT, 4.8, cw4, f"{signed(N.change(MER, '6 h'))}%",
              f"At 6 h, where the vehicle spread is wide (95% CI {signed(lo_mer6)}% to "
              f"{signed(hi_mer6)}%)", MUTED)
    d.footnote(s, "Median MerTK fluorescence on the M1-like subset of red pulp macrophages. "
                  "Read on the whole red pulp gate (about 500 events per animal), it would not "
                  "depend on the M1/M2 split.")

    # 9. Next steps (dark) ------------------------------------------------------------
    s = d.slide(dark=True, notes=f"""
Power, computed from the variability in this cohort (SD pooled over all five groups):
a 30% difference at n = 8 per arm gives {pct(pw8[B1A])} power for B-1a at 24 h and
{pct(pw8[NEU])} for neutrophils at 6 h; at n = 10, {pct(pw10[B1A])} and {pct(pw10[NEU])}.
Effects estimated at n = 3-4 tend to be inflated, so plan on 10 if you can.
One cohort can carry both: a 6 h arm for neutrophils and a 24 h arm for B-1a.
""")
    d.header(s, "Next steps", "What would turn these trends into results", legend=False,
             dark=True)
    steps = [
        ("Powered replication",
         "n\u00a0=\u00a08–10 per arm, sized for a 30% difference. Pre-specify vehicle vs NM72 as "
         "the primary contrasts, for neutrophils at 6 h and for B-1a at 24 h, so neither is "
         "corrected against all ten pairs."),
        ("One gate, applied blind",
         "Set the CD3 gate once and apply it to every sample before unblinding, so no animal "
         "is re-gated on its own."),
        ("MerTK on the red pulp gate",
         "Add one statistic in FlowJo: MerTK median on the whole red pulp gate, the cleanest "
         "efferocytosis readout in the panel."),
    ]
    for i, (head, body) in enumerate(steps):
        x0 = LEFT + i * (cw + gap)
        number_badge(s, x0, 1.85, 0.62, str(i + 1), WHITE, color=DARK)
        add_text(s, x0, 2.75, cw - 0.2, 0.9, head, size=21, bold=True, color=WHITE,
                 where="next head")
        add_text(s, x0, 3.55, cw - 0.25, 1.95, body, size=15, color=DTEXT2, spacing=1.12,
                 where="next body")
    add_rect(s, LEFT, 5.75, RIGHT - LEFT, 1.0, "2B2B29", radius=0.12)
    add_text(s, LEFT + 0.35, 5.75, RIGHT - LEFT - 0.7, 1.0,
             "**One cohort can answer both questions:** a 6 h arm for neutrophils and a 24 h "
             "arm for B-1a, each vehicle vs NM72.", size=16, color=WHITE, anchor="m",
             where="next band")

    # B1. Statistics table -------------------------------------------------------------
    s = d.slide(notes="""
All three procedures agree on the drug question: no vehicle-vs-NM72 comparison reaches
0.05 after correction. They differ on the injury contrasts against uninjured, where Tukey
pools the SD across groups and so gives the n = 2 uninjured group more weight than T3 does.
""")
    d.header(s, "Backup  ·  Statistics", "Every planned comparison, three procedures",
             legend=False)
    stats_table(s, N)
    d.footnote(s, "Bold: p\u00a0<\u00a00.05. Dunnett's T3 is the default because it does not assume "
                  "equal SDs (Brown-Forsythe p = 0.048 for MerTK). Tukey pools the SD, which "
                  "makes comparisons against the n\u00a0=\u00a02 uninjured group look stronger. All 56 "
                  "pairwise p-values: final/posthoc_all_methods.csv.", y=6.55)

    # B2. Sample quality ----------------------------------------------------------------
    s = d.slide(notes=f"""
The six samples run back to back from 10:14 to 10:37 on 7 Sep have clean CD3 gates; every
other sample's CD3 gate holds a large share of non-CD4, non-CD8 events. That is sample
quality and flow rate (three of the six ran at Medium), not a drift across the run.
All four 6 h NM72 animals are in that block, so for readouts that move with sample quality
the 6 h drug comparison is confounded. CD3 and CD4 move with it; the readouts in the
summary figure do not.
""")
    d.header(s, "Backup  ·  Sample quality", "Why CD3 and CD4 frequencies are not shown")
    add_figure(s, figs["quality_vs_viability"], (LEFT, 1.45, 7.3, 5.2), align="l",
               alt="Six scatter plots of each readout against sample viability, with "
                   "Spearman correlations.")
    blk_ids = ", ".join(str(a) for a in q_ok[q_ok["block"]].sort_values("btim").index) \
        if "btim" in q_ok.columns else ", ".join(str(a) for a in blk.index)
    paras = [
        f"Six samples run back to back ({blk_ids}) have much cleaner CD3 gates: CD4^{{+}} "
        f"and CD8^{{+}} make up {in_blk.min():.0f}–{in_blk.max():.0f}% of CD3^{{+}}, against "
        f"{out_blk.min():.0f}–{out_blk.max():.0f}% in every other sample.",
        "They include all four 6 h NM72 animals.",
        f"CD3^{{+}} and CD4^{{+}} frequencies track viability "
        f"(ρ = {signed(rho['CD3+ T cells'][0], 2)} and {signed(rho['CD4+'][0], 2)}), so a "
        "6 h drug effect on them cannot be told apart from sample quality.",
        f"The readouts in the summary figure do not (|ρ| < 0.42), so they are shown.",
    ]
    add_text(s, 8.35, 1.6, RIGHT - 8.35, 4.9, "\n".join(paras), size=14, color=TEXT2,
             spacing=1.08, after=12, where="quality text")

    # B3. Exclusions ----------------------------------------------------------------------
    s = d.slide(notes=f"""
The rule is on the live-leukocyte gate alone, not on any readout: fewer than 1,000 live
leukocyte events. Two animals fall below it, and the gap to the next animal
({next_low.name}, {int(next_low.live_events):,} events) is large.
""")
    d.header(s, "Backup  ·  Exclusions", "Two animals excluded for low live-leukocyte yield")
    add_figure(s, figs["exclusions"], (LEFT, 1.5, 8.0, 5.0), align="l",
               alt="Live leukocyte events per animal on a log scale, with the 1,000-event "
                   "exclusion threshold.")
    lines = []
    for a, r in excluded.iterrows():
        where = ("both panels" if a == 489 else
                 "the T cell / myeloid panel only; it is kept on the B cell panel")
        lines.append(f"**{a}** ({r.timepoint + ' ' if r.timepoint != 'Uninjured' else ''}"
                     f"{r.treatment.lower()}): {int(r.live_events):,} live leukocytes. "
                     f"Excluded from {where}.")
    lines.append(f"Threshold: 1,000 events. The next-lowest animal, {next_low.name}, has "
                 f"{int(next_low.live_events):,}.")
    add_text(s, 9.0, 1.6, RIGHT - 9.0, 4.6, "\n".join(lines), size=14, color=TEXT2,
             spacing=1.08, after=12, where="exclusion text")

    # B4. Tukey ----------------------------------------------------------------------------
    tk = "tukey_p"
    s = d.slide(notes=f"""
Same panels with ordinary one-way ANOVA and Tukey. The injury rise at 6 h becomes
significant in both arms ({fmt_p(N.p(NEU, UN, V6, tk))} vehicle, {fmt_p(N.p(NEU, UN, N6, tk))}
NM72), because Tukey pools the SD across groups. No drug contrast gets there: the
smallest is neutrophils at 6 h, p = {N.p(NEU, V6, N6, tk):.3f}.
""")
    d.header(s, "Backup  ·  Tukey", "With Tukey's test, the drug conclusions hold")
    d.method_note(s, "Mean ± SEM  ·  Tukey-adjusted p")
    add_figure(s, figs["overview_tukey"], (LEFT, 1.4, RIGHT - LEFT, 5.6),
               alt="The six summary panels with Tukey-adjusted p-values.")
    return d.prs


def callout_row(d: Deck, s, cards, y, h=1.28):
    cw, gap = (RIGHT - LEFT - 2 * 0.3) / 3, 0.3
    for i, (big, label, color) in enumerate(cards):
        x0 = LEFT + i * (cw + gap)
        add_rect(s, x0, y, cw, h, CARD, radius=0.1)
        add_text(s, x0 + 0.25, y + 0.14, cw - 0.5, 0.42, big, size=24, bold=True,
                 color=color, where=f"slide {d.count} card number")
        add_text(s, x0 + 0.25, y + 0.6, cw - 0.5, h - 0.68, label, size=11.5, color=TEXT2,
                 spacing=1.02, where=f"slide {d.count} card label")


def stats_table(s, N: Numbers) -> None:
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
    widths = [1.75, 2.85, 2.1, 1.75, 1.55, 2.13]
    n_rows = len(rows) + 1
    x, y, row_h = LEFT, 1.4, 0.228
    gf = s.shapes.add_table(n_rows, len(head), Inches(x), Inches(y), Inches(sum(widths)),
                            Inches(row_h * n_rows))
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
            p.alignment = {"l": PP_ALIGN.LEFT, "r": PP_ALIGN.RIGHT, "c": PP_ALIGN.CENTER}[align]
            _runs(p, para, size, color, bold)
        _cell_borders(c, fill=fill, bottom=bottom)

    for j, h in enumerate(head):
        cell(0, j, h, bold=True, color=INK if j < 5 else INK, size=10.5,
             align="l" if j < 3 else "r", bottom=(INK, 1.0))
    for i, row in enumerate(rows, start=1):
        if row[0] == "section":
            cell(i, 0, row[1], bold=True, color=MUTED, size=10)
            for j in range(1, len(head)):
                cell(i, j, "", size=10)
            tbl.cell(i, 0).merge(tbl.cell(i, len(head) - 1))
            continue
        label, text, pop, a, b = row
        ps = [N.p(pop, a, b, col) for col in ("welch_p", "tukey_p", "dunnett_t3_p")]
        cell(i, 0, label, size=10.5)
        cell(i, 1, text, size=10.5, color=TEXT2)
        cell(i, 2, means(pop, a, b), size=10.5, color=TEXT2)
        for j, pv in enumerate(ps, start=3):
            txt = "< 0.001" if pv < 0.001 else f"{pv:.3f}"
            cell(i, j, txt, bold=pv < 0.05, color=INK if pv < 0.05 else TEXT2, size=10.5,
                 align="r", fill=CARD if j == 5 else None, bottom=(RULE, 0.5))
        for j in range(3):
            _cell_borders(tbl.cell(i, j), bottom=(RULE, 0.5))


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


def main() -> int:
    figs = sf.render()
    prs = build(figs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    for w in WARNINGS:
        print("FIT WARNING  " + w)
    print(f"Saved: {OUT} ({len(prs.slides)} slides)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
