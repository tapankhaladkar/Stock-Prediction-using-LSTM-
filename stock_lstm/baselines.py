"""Naive benchmarks an LSTM has to beat before it can claim any skill.

Every function returns a Series aligned to ``prices.index`` where the value at date t is
a one-step-ahead prediction of the close at t made using only closes before t (NaN where
that is not possible). Persistence ("tomorrow = today") is famously hard to beat for
daily stock prices.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def persistence(prices: pd.Series) -> pd.Series:
    return prices.shift(1).rename("Persistence")


def drift(prices: pd.Series) -> pd.Series:
    """Persistence plus the average daily log return seen so far ("random walk with drift").

    The prediction for day t is ``P[t-1] * exp(mean of the log returns strictly before t)``.
    Stocks rise on average, so this is a fairer bar than persistence: a model that only
    learned "prices go up a bit" would otherwise look skilful in a rising market.
    """
    rets = np.log(prices).diff()                     # rets[t] = log(P[t] / P[t-1]); NaN on row 0
    mu = rets.expanding().mean().shift(1)            # mean of returns up to t-1: never sees day t
    return (prices.shift(1) * np.exp(mu)).rename("Drift")


def moving_average(prices: pd.Series, window: int) -> pd.Series:
    # shift(1): the average of the `window` closes strictly before t.
    return prices.shift(1).rolling(window).mean().rename(f"MA({window})")


def arima_one_step(prices: pd.Series, n_train: int, order: tuple[int, int, int] = (1, 1, 1)) -> pd.Series:
    """ARIMA on log prices, fit once on ``prices[:n_train]``, then rolled forward one step
    at a time over the rest without refitting (parameters never see future data).
    """
    from statsmodels.tsa.arima.model import ARIMA

    if not 10 < n_train < len(prices):
        raise ValueError("n_train must leave at least one out-of-sample day and enough history to fit")
    log_p = np.log(prices.to_numpy(dtype=float))
    with warnings.catch_warnings():
        # Near-unit-root log prices routinely make statsmodels fall back to zero start
        # parameters; that is expected and harmless. Other warnings still surface.
        warnings.filterwarnings("ignore", message="Non-stationary starting autoregressive parameters")
        warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters")
        fitted = ARIMA(log_p[:n_train], order=order).fit()
    rolled = fitted.append(log_p[n_train:], refit=False)
    one_step = rolled.predict(start=n_train, end=len(prices) - 1, dynamic=False)
    out = pd.Series(np.nan, index=prices.index, name=f"ARIMA{order}")
    out.iloc[n_train:] = np.exp(one_step)
    return out
