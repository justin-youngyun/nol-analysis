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
