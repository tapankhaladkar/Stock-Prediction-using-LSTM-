"""Forecast metrics. Everything here works on prices in dollars, never on scaled values."""
from __future__ import annotations

import numpy as np


def _pair(y_true, y_pred):
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    # Strict shapes: (n,) vs (n, 1) would silently broadcast to (n, n) in some code paths.
    if a.ndim != 1 or a.shape != b.shape:
        raise ValueError(f"expected two 1-D arrays of equal length, got {a.shape} and {b.shape}")
    if np.isnan(a).any() or np.isnan(b).any():
        raise ValueError("NaN in metric inputs; align the series before scoring")
    return a, b


def rmse(y_true, y_pred) -> float:
    a, b = _pair(y_true, y_pred)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(y_true, y_pred) -> float:
    a, b = _pair(y_true, y_pred)
    return float(np.mean(np.abs(a - b)))


def mape(y_true, y_pred) -> float:
    """Mean absolute percentage error, in percent."""
    a, b = _pair(y_true, y_pred)
    if (a <= 0).any():
        raise ValueError("MAPE needs strictly positive actual values")
    return float(np.mean(np.abs(a - b) / a) * 100)


def directional_accuracy(actual, pred, prev) -> float:
    """Percent of days where the predicted move (vs. yesterday's close) has the right sign.

    Days with no actual move are skipped. Returns NaN for a model that never calls a
    direction (e.g. persistence predicts "no change"), instead of a misleading 0%.
    """
    a, p = _pair(actual, pred)
    _, v = _pair(actual, prev)
    pred_move = np.sign(p - v)
    if not pred_move.any():
        return float("nan")
    true_move = np.sign(a - v)
    keep = true_move != 0
    return float(np.mean(pred_move[keep] == true_move[keep]) * 100)


def up_day_rate(actual, prev) -> float:
    """Percent of days that closed up. The bar directional accuracy has to clear:
    always guessing "up" scores this without any model."""
    a, v = _pair(actual, prev)
    keep = a != v
    return float(np.mean(a[keep] > v[keep]) * 100)


def score(actual, pred, prev) -> dict:
    """RMSE / MAE (dollars), MAPE and directional accuracy (percent) for one model."""
    return {
        "RMSE": rmse(actual, pred),
        "MAE": mae(actual, pred),
        "MAPE_%": mape(actual, pred),
        "DirAcc_%": directional_accuracy(actual, pred, prev),
    }


def _block_resample_indices(n: int, block: int, n_boot: int, seed: int) -> np.ndarray:
    """Moving-block bootstrap: (n_boot, n) indices made of consecutive runs of ``block`` units."""
    block = max(1, min(block, n))
    rng = np.random.default_rng(seed)
    n_blocks = -(-n // block)
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    return (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]


def bootstrap_mse_ratio(se_model, se_base, block: int = 10, n_boot: int = 2000, seed: int = 0,
                        level: float = 0.95) -> tuple[float, float]:
    """Moving-block bootstrap CI for sqrt(mean(se_model) / mean(se_base)).

    ``se_*`` are squared errors per unit: days for one-step forecasts, or whole forecast origins
    for multi-day forecasts. Neighbouring units are correlated (overlapping windows, volatility
    clusters), hence blocks of consecutive units instead of single ones.
    """
    m = np.asarray(se_model, dtype=float)
    b = np.asarray(se_base, dtype=float)
    if m.ndim != 1 or m.shape != b.shape or m.size == 0:
        raise ValueError(f"expected two non-empty 1-D arrays of equal length, got {m.shape} and {b.shape}")
    idx = _block_resample_indices(len(m), block, n_boot, seed)
    ratios = np.sqrt(m[idx].mean(axis=1) / b[idx].mean(axis=1))
    lo, hi = np.quantile(ratios, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def bootstrap_rmse_ratio(actual, pred_model, pred_base, block: int = 10, n_boot: int = 2000,
                         seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """Moving-block bootstrap CI for RMSE(model) / RMSE(baseline).

    A ratio whose interval includes 1.0 means the data cannot tell the two apart;
    daily errors are autocorrelated, hence blocks instead of single days.
    """
    a, m = _pair(actual, pred_model)
    _, b = _pair(actual, pred_base)
    return bootstrap_mse_ratio((a - m) ** 2, (a - b) ** 2, block=block, n_boot=n_boot, seed=seed, level=level)


def bootstrap_mean_ci(values, block: int = 3, n_boot: int = 2000, seed: int = 0,
                      level: float = 0.95) -> tuple[float, float]:
    """Moving-block bootstrap CI for the mean of ``values`` (one number per unit, e.g. per origin)."""
    v = np.asarray(values, dtype=float)
    if v.ndim != 1 or v.size == 0:
        raise ValueError("values must be a non-empty 1-D array")
    idx = _block_resample_indices(len(v), block, n_boot, seed)
    means = v[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)
