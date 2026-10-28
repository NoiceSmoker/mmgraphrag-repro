"""Bootstrap confidence intervals.

Neither paper reports variance. Every comparison in this repository is
therefore accompanied by a (paired) percentile bootstrap interval computed
over evaluation questions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CI:
    mean: float
    low: float
    high: float

    def as_dict(self, prefix: str) -> dict[str, float]:
        return {prefix: self.mean, f"{prefix}_ci_low": self.low, f"{prefix}_ci_high": self.high}


def bootstrap_mean(values, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> CI:
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return CI(float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    return CI(float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def paired_bootstrap_diff(a, b, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[CI, float]:
    """CI of mean(a - b) over paired samples and the one-sided bootstrap
    probability that the difference is <= 0 (a rough p-value for a > b)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired arrays must match: {a.shape} vs {b.shape}")
    d = a - b
    if d.size == 0:
        return CI(float("nan"), float("nan"), float("nan")), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    ci = CI(float(d.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))
    p_le_zero = float((means <= 0).mean())
    return ci, p_le_zero
