#!/usr/bin/env python
"""
Re-apply a FlowJo workspace's gating to one FCS file, outside FlowJo.

Reads the sample's compensation matrix, per-channel display transforms and gate
tree from the .wsp, applies them to the events in the .fcs, and checks every
population's count against the count FlowJo saved in the workspace. The
drawing code (gating_slides.py) only runs on a sample that reproduces.

    python3 flowjo_gating.py WORKSPACE.wsp SAMPLE.fcs

The biexponential transform is FlowJo's own (the published "biex" algorithm,
as reimplemented in cytolib), not logicle, so axes and polygon gates sit
where FlowJo puts them.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.optimize import brentq

NS = {
    "gating": "http://www.isac-net.org/std/Gating-ML/v2.0/gating",
    "transforms": "http://www.isac-net.org/std/Gating-ML/v2.0/transformations",
    "data-type": "http://www.isac-net.org/std/Gating-ML/v2.0/datatypes",
}
G, T, D = (f"{{{NS[k]}}}" for k in ("gating", "transforms", "data-type"))


# ---------------------------------------------------------------------------
# FCS
# ---------------------------------------------------------------------------

def read_fcs(path: Path) -> tuple[dict, np.ndarray, list[str]]:
    """FCS 3.x list-mode float data: (keywords, events x parameters, $PnN names)."""
    raw = Path(path).read_bytes()
    head = raw[:58].decode("ascii")
    t0, t1, d0, d1 = (int(head[i:i + 8]) for i in (10, 18, 26, 34))
    text = raw[t0:t1 + 1].decode("latin-1")
    delim = text[0]
    body = text[1:].replace(delim * 2, "\0")          # a doubled delimiter is a literal one
    parts = [p.replace("\0", delim) for p in body.split(delim)]
    kv = dict(zip(parts[0::2], parts[1::2]))
    if not d0 or not d1:
        d0, d1 = int(kv["$BEGINDATA"]), int(kv["$ENDDATA"])
    npar, tot = int(kv["$PAR"]), int(kv["$TOT"])
    if kv.get("$DATATYPE") != "F" or kv.get("$MODE", "L") != "L":
        raise ValueError("only list-mode float FCS files are supported")
    order = "<" if kv["$BYTEORD"].strip().startswith("1,2") else ">"
    data = np.frombuffer(raw[d0:d1 + 1], dtype=order + "f4", count=npar * tot)
    names = [kv[f"$P{i}N"] for i in range(1, npar + 1)]
    return kv, data.reshape(tot, npar).astype(float), names


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

@dataclass
class Transform:
    """Data value -> display channel in [0, length], FlowJo's way."""
    kind: str
    params: dict
    length: int = 256
    _x: np.ndarray | None = None      # data values of the channel table
    _y: np.ndarray | None = None      # channels

    def __post_init__(self):
        if self.kind == "biex":
            self.length = int(float(self.params.get("length", 256)))
            self._x, self._y = _biex_table(
                self.length, float(self.params["maxRange"]), float(self.params["pos"]),
                float(self.params["neg"]), float(self.params["width"]))
        elif self.kind == "linear":
            self.length = 256

    def __call__(self, v, clip: bool = True):
        """Channel for each value. clip=False extends the scale past both ends,
        which gate tests need: a vertex may sit off the displayed range."""
        v = np.asarray(v, float)
        if self.kind == "biex":
            c = np.interp(v, self._x, self._y)
            if not clip:
                lo_slope = (self._y[1] - self._y[0]) / (self._x[1] - self._x[0])
                hi_slope = (self._y[-1] - self._y[-2]) / (self._x[-1] - self._x[-2])
                c = np.where(v < self._x[0], self._y[0] + (v - self._x[0]) * lo_slope, c)
                c = np.where(v > self._x[-1], self._y[-1] + (v - self._x[-1]) * hi_slope, c)
            return c
        lo, hi = float(self.params.get("minRange", 0)), float(self.params["maxRange"])
        c = (v - lo) / (hi - lo) * self.length
        return np.clip(c, 0, self.length) if clip else c

    def inverse(self, c):
        c = np.asarray(c, float)
        if self.kind == "biex":
            return np.interp(c, self._y, self._x)
        lo, hi = float(self.params.get("minRange", 0)), float(self.params["maxRange"])
        return lo + c / self.length * (hi - lo)

    @property
    def data_range(self) -> tuple[float, float]:
        return float(self.inverse(0)), float(self.inverse(self.length))


def _biex_table(length: int, max_value: float, pos: float, neg: float, width_basis: float):
    ln10 = np.log(10.0)
    decades = pos
    width = np.log10(-width_basis)
    decades -= width / 2
    extra = max(neg, 0.0) + width / 2
    zero = int(extra * length / (extra + decades))
    zero = min(zero, length // 2)
    if zero > 0:
        decades = extra * length / zero
    width /= 2 * decades
    positive_range = ln10 * decades
    minimum = max_value / np.exp(positive_range)
    if width == 0:
        negative_range = positive_range
    else:
        fb = -2 * np.log(positive_range) + width * positive_range
        negative_range = brentq(lambda d: 2 * np.log(d) + width * d + fb, 1e-12, positive_range)
    n = length + 1
    j = np.arange(n, dtype=float)
    positive = np.exp(j / n * positive_range)
    negative = np.exp(-j / n * negative_range)
    negative *= np.exp((positive_range + negative_range) * (width + extra / decades))
    s = positive[zero] - negative[zero]
    positive[zero:] = minimum * (positive[zero:] - negative[zero:] - s)
    for k in range(zero):
        positive[k] = -positive[2 * zero - k]
    return positive, j * (length / (n - 1))


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------

@dataclass
class Gate:
    kind: str                         # rect, polygon
    dims: list[str]
    lo: list[float | None] = field(default_factory=list)   # rect bounds, data units
    hi: list[float | None] = field(default_factory=list)
    vertices: list[tuple[float, ...]] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


@dataclass
class Population:
    path: str
    name: str
    count: int
    negated: bool
    gate: Gate | None
    children: list["Population"] = field(default_factory=list)
    mask: np.ndarray | None = None


@dataclass
class Sample:
    name: str
    sample_id: str
    events: dict[str, np.ndarray]
    transforms: dict[str, Transform]
    markers: dict[str, str]           # channel -> marker, as the workspace names it
    root: Population
    populations: dict[str, Population]


def _parse_gate(gate_el) -> Gate | None:
    if gate_el is None or not len(gate_el):
        return None
    g = gate_el[0]
    kind = g.tag.split("}")[-1]
    dims = [d.find("data-type:fcs-dimension", NS).get(D + "name")
            for d in g.findall("gating:dimension", NS)]
    if kind == "RectangleGate":
        lo, hi = [], []
        for d in g.findall("gating:dimension", NS):
            lo.append(float(d.get(G + "min")) if d.get(G + "min") is not None else None)
            hi.append(float(d.get(G + "max")) if d.get(G + "max") is not None else None)
        extra = {k: v for k, v in g.attrib.items()}
        ratio = g.find("gating:dimension", NS).get("yRatio")
        if ratio is not None:
            extra["yRatio"] = float(ratio)
        return Gate("rect", dims, lo, hi, extra=extra)
    if kind == "PolygonGate":
        verts = [tuple(float(c.get(D + "value")) for c in v.findall("gating:coordinate", NS))
                 for v in g.findall("gating:vertex", NS)]
        return Gate("polygon", dims, vertices=verts, extra=dict(g.attrib))
    raise ValueError(f"unsupported gate type {kind}")


def _inside(gate: Gate, ev: dict[str, np.ndarray], tf: dict[str, Transform]) -> np.ndarray:
    if gate.kind == "rect":
        m = np.ones(len(next(iter(ev.values()))), bool)
        for dim, lo, hi in zip(gate.dims, gate.lo, gate.hi):
            v = ev[dim]
            if lo is not None:
                m &= v >= lo
            if hi is not None:
                m &= v < hi
        return m
    # FlowJo draws polygon edges straight on the display scale, so test there.
    pts = np.column_stack([tf[d](ev[d], clip=False) for d in gate.dims])
    poly = np.column_stack([tf[d](np.array([v[i] for v in gate.vertices]), clip=False)
                            for i, d in enumerate(gate.dims)])
    return MplPath(poly).contains_points(pts)


def load(wsp: Path, fcs: Path) -> Sample:
    root = ET.parse(wsp).getroot()
    kv, data, names = read_fcs(fcs)
    fname = Path(fcs).name
    target = kv.get("$FIL", fname)

    sample_el = None
    for s in root.iter("Sample"):
        ds = s.find("DataSet")
        node = s.find("SampleNode")
        if ds is None or node is None:
            continue
        if node.get("name") in (target, fname) or ds.get("uri", "").endswith("/" + target):
            sample_el = s
            break
    if sample_el is None:
        raise SystemExit(f"{target} is not a sample in {wsp}")
    node = sample_el.find("SampleNode")

    events = {n: data[:, i] for i, n in enumerate(names)}
    # Compensation, as the workspace defines it: comp = raw . inverse(spillover).
    sm = sample_el.find("transforms:spilloverMatrix", NS)
    if sm is not None:
        params = [p.get(D + "name") for p in sm.find("data-type:parameters", NS)]
        prefix = sm.get("prefix", "Comp-")
        M = np.eye(len(params))
        for i, row in enumerate(sm.findall("transforms:spillover", NS)):
            for c in row.findall("transforms:coefficient", NS):
                M[i, params.index(c.get(D + "parameter"))] = float(c.get(T + "value"))
        comp = np.column_stack([events[p] for p in params]) @ np.linalg.inv(M)
        for i, p in enumerate(params):
            events[prefix + p] = comp[:, i]

    transforms = {}
    for el in sample_el.find("Transformations"):
        kind = el.tag.split("}")[-1]
        par = el.find("data-type:parameter", NS).get(D + "name")
        attrs = {k.split("}")[-1]: v for k, v in el.attrib.items()}
        transforms[par] = Transform(kind, attrs)

    keywords = {k.get("name"): k.get("value") for k in sample_el.iter("Keyword")}
    markers = {}
    i = 1
    while f"$P{i}N" in keywords:
        ch, mk = keywords[f"$P{i}N"], keywords.get(f"$P{i}S", "")
        markers[ch] = markers["Comp-" + ch] = mk
        i += 1

    pops: dict[str, Population] = {}
    top = Population("", node.get("name"), int(node.get("count")), False, None)
    top.mask = np.ones(len(data), bool)

    def walk(el, parent: Population):
        for sub in el.findall("Subpopulations"):
            for child in sub:
                if child.tag not in ("Population", "NotNode"):
                    continue
                name = child.get("name")
                path = f"{parent.path}/{name}" if parent.path else name
                gate = _parse_gate(child.find("Gate"))
                pop = Population(path, name, int(child.get("count")),
                                 child.tag == "NotNode", gate)
                inside = _inside(gate, events, transforms)
                pop.mask = parent.mask & (~inside if pop.negated else inside)
                parent.children.append(pop)
                pops[path] = pop
                walk(child, pop)

    walk(node, top)
    return Sample(target, node.get("sampleID"), events, transforms, markers, top, pops)


def check_counts(sample: Sample) -> list[tuple[str, int, int]]:
    """(path, FlowJo count, count here) for every population."""
    return [(p.path, p.count, int(p.mask.sum())) for p in sample.populations.values()]


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 2:
        print(__doc__)
        return 2
    s = load(Path(argv[0]), Path(argv[1]))
    worst = 0.0
    for path, fj, here in check_counts(s):
        rel = abs(here - fj) / max(fj, 1)
        worst = max(worst, rel)
        print(f"{fj:8d} {here:8d} {'ok' if here == fj else f'{here - fj:+d}':>6}  {path}")
    print(f"largest relative difference: {worst:.4%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
