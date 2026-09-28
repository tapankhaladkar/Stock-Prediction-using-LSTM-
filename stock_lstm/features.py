"""Turn a price series into stationary, leak-free model inputs."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def log_returns(prices: pd.Series) -> pd.Series:
    """Daily log return r[t] = log(P[t] / P[t-1]); indexed by the date of P[t]."""
    return np.log(prices).diff().dropna().rename("log_return")


@dataclass(frozen=True)
class Scaler:
    """Standardiser. Fit it on the training slice only: fitting on the full series lets
    test-period statistics leak into training."""

    mean: float
    std: float

    @classmethod
    def fit(cls, x) -> "Scaler":
        x = np.asarray(x, dtype=float)
        if x.size < 2:
            raise ValueError("need at least 2 values to fit a scaler")
        return cls(mean=float(x.mean()), std=max(float(x.std()), 1e-12))

    def transform(self, x):
        return (np.asarray(x, dtype=float) - self.mean) / self.std

    def inverse(self, z):
        return np.asarray(z, dtype=float) * self.std + self.mean


def make_windows(x: np.ndarray, window: int, start: int, stop: int):
    """Supervised samples for targets t in [start, stop).

    Input is x[t-window : t] (strictly before t) and the target is x[t]. Windows may reach
    back before ``start`` (that is just history), but never touch x[t] or later.
    Returns X of shape (n, window, 1), y of shape (n,), and the target positions t.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be 1-D")
    if window < 1 or start < window:
        raise ValueError(f"start ({start}) must be >= window ({window}) so every window is complete")
    if stop > len(x) or stop <= start:
        raise ValueError(f"need start < stop <= len(x); got start={start}, stop={stop}, len={len(x)}")
    t = np.arange(start, stop)
    idx = t[:, None] + np.arange(-window, 0)
    return x[idx][..., None], x[t], t
