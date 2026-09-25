#!/usr/bin/env python
"""
Multiple-comparison procedures for the five-group SCI design, matching the
options GraphPad Prism offers so any number here can be reproduced there.

  tukey        ordinary one-way ANOVA + Tukey HSD. Pools the variance, so it
               assumes every group has the same SD.
  games_howell Welch ANOVA + Games-Howell. Per-pair Welch SE and df, studentized
               range reference. No equal-SD assumption; known to run liberal
               below n~6 per group.
  dunnett_t3   Welch ANOVA + Dunnett's T3. Per-pair Welch SE and df, studentized
               maximum modulus reference. No equal-SD assumption, and the one
               Prism recommends after Welch ANOVA when n < 50 per group.

All three compare every pair of groups, as Prism's "compare all pairs" does.
Groups with fewer than two animals carry no variance estimate, so they are
dropped from the family and named in the result rather than silently.

within_rows() is the other family: a few planned comparisons inside a
two-factor design (vehicle vs NM72 at each timepoint), as Prism's two-way
ANOVA does when asked to compare cell means within each row. It uses the
pooled SD of the design's cells and Sidak's correction for the number of
comparisons, and also reports Welch's t for each, with and without Sidak.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy import integrate, stats


def _welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
    t = (b.mean() - a.mean()) / np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (a.size - 1) + vb ** 2 / (b.size - 1))
    return float(t), float(df)


def _smm_cdf(c: float, m: int, df: float) -> float:
    """P(max of m |t| <= c): studentized maximum modulus with df.

    Conditional on the shared scale S = sqrt(chi2_df / df), the m moduli are
    independent, so the CDF is E_S[(2*Phi(c*S) - 1)^m].
    """
    def integrand(s):
        return (2 * stats.norm.cdf(c * s) - 1) ** m * stats.chi2.pdf(df * s * s, df) * 2 * df * s
    val, _ = integrate.quad(integrand, 0, np.inf, limit=200)
    return float(min(max(val, 0.0), 1.0))


def pairwise(groups: dict[str, np.ndarray]) -> dict:
    """All pairwise comparisons under the three procedures.

    groups: name -> 1-d array. Returns {"dropped": [...], "rows": [...]}.
    """
    clean = {k: np.asarray(v, float)[~np.isnan(np.asarray(v, float))] for k, v in groups.items()}
    dropped = [k for k, v in clean.items() if v.size < 2]
    g = {k: v for k, v in clean.items() if v.size >= 2}
    names = list(g)
    k = len(names)
    rows = []
    if k < 2:
        return {"dropped": dropped, "rows": rows}

    tk = stats.tukey_hsd(*[g[n] for n in names])
    m = k * (k - 1) // 2
    for i, j in itertools.combinations(range(k), 2):
        a, b = g[names[i]], g[names[j]]
        t, df = _welch(a, b)
        welch_p = float(2 * stats.t.sf(abs(t), df))
        gh = float(stats.studentized_range.sf(abs(t) * np.sqrt(2), k, df))
        t3 = 1.0 - _smm_cdf(abs(t), m, df)
        rows.append({"group_a": names[i], "group_b": names[j],
                     "n_a": a.size, "n_b": b.size,
                     "mean_a": float(a.mean()), "mean_b": float(b.mean()),
                     "welch_p": welch_p,
                     "tukey_p": float(tk.pvalue[i, j]),
                     "games_howell_p": min(gh, 1.0),
                     "dunnett_t3_p": min(max(t3, welch_p), 1.0)})
    return {"dropped": dropped, "rows": rows}


METHODS = {
    "none": ("welch_p", "Welch's t-test, uncorrected"),
    "tukey": ("tukey_p", "One-way ANOVA + Tukey (assumes equal SDs)"),
    "games-howell": ("games_howell_p", "Welch ANOVA + Games-Howell"),
    "dunnett-t3": ("dunnett_t3_p", "Welch ANOVA + Dunnett's T3"),
}


def within_rows(cells: dict[str, np.ndarray], pairs: list[tuple[str, str]]) -> list[dict]:
    """Planned comparisons within the rows of a two-factor design.

    cells holds every cell of the design (for the SCI cohort, the four injured
    groups), pairs the comparisons to make. The pooled-SD test uses the full
    factorial model's residual mean square, df = N - number of cells, so it is
    Prism's two-way ANOVA with Sidak's multiple comparisons within rows.
    """
    cells = {k: np.asarray(v, float)[~np.isnan(np.asarray(v, float))] for k, v in cells.items()}
    cells = {k: v for k, v in cells.items() if v.size}
    df = sum(v.size for v in cells.values()) - len(cells)
    mse = sum(((v - v.mean()) ** 2).sum() for v in cells.values()) / df
    m = len(pairs)
    rows = []
    for a, b in pairs:
        x, y = cells.get(a), cells.get(b)
        if x is None or y is None or x.size < 2 or y.size < 2:
            continue
        t = (y.mean() - x.mean()) / np.sqrt(mse * (1 / x.size + 1 / y.size))
        p = float(2 * stats.t.sf(abs(t), df))
        wt, wdf = _welch(x, y)
        wp = float(2 * stats.t.sf(abs(wt), wdf))
        rows.append({"group_a": a, "group_b": b, "n_a": x.size, "n_b": y.size,
                     "mean_a": float(x.mean()), "mean_b": float(y.mean()),
                     "anova_t": float(t), "anova_df": df, "anova_p": p,
                     "anova_sidak_p": 1 - (1 - p) ** m,
                     "welch_p": wp, "welch_sidak_p": 1 - (1 - wp) ** m})
    return rows
