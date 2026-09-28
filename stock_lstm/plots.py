"""Plots with titles, labelled axes, dates and legends."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from .forecast import Forecast


def plot_adjusted_vs_raw(raw: pd.DataFrame, ticker: str = "AAPL"):
    """Traded close vs adjusted close: the gap is what a split does to an unadjusted series."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(raw["date"], raw["close"], label="close (as traded)", alpha=0.7)
    ax.plot(raw["date"], raw["adjClose"], label="adjClose (used for modelling)")
    ax.set(title=f"{ticker}: raw vs split/dividend-adjusted close", xlabel="Date", ylabel="Price (USD)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_walk_forward(prices: pd.Series, predictions: pd.DataFrame, models=("LSTM", "Persistence")):
    """Actual vs predicted closes over the walk-forward test days, with fold boundaries."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    days = predictions.index
    ax.plot(days, prices.loc[days], color="black", lw=1.4, label="actual")
    for m in models:
        ax.plot(days, predictions[m], lw=1, alpha=0.85, label=m)
    for _, chunk in predictions.groupby("fold"):
        ax.axvline(chunk.index[0], color="grey", ls=":", lw=0.8)
    ax.set(title="One-step-ahead predictions on walk-forward test days (dotted = new fold)",
           xlabel="Date", ylabel="Close (USD)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_ratio_ci(table: pd.DataFrame, reference: str = "Persistence"):
    """RMSE relative to the reference model, with 95% bootstrap intervals. Below 1 is better."""
    t = table.drop(index=reference)
    ratio = t[f"RMSE_vs_{reference}"].to_numpy()
    err = [ratio - t["ratio_CI95_lo"].to_numpy(), t["ratio_CI95_hi"].to_numpy() - ratio]
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(t) + 1.6))
    ax.errorbar(ratio, range(len(t)), xerr=err, fmt="o", capsize=4)
    ax.axvline(1.0, color="red", ls="--", label=f"same as {reference}")
    ax.set_yticks(range(len(t)), t.index)
    ax.set(title=f"RMSE vs {reference} (95% CI; left of the line = better)", xlabel="RMSE ratio")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_forecast(history: pd.Series, fc: Forecast, scored: pd.DataFrame | None = None,
                  ticker: str = "AAPL", tail: int = 120):
    """Recent history, the forecast fan, and (if given) the closes that actually happened."""
    h = history.iloc[-tail:]
    if scored is not None:
        dates = scored.index
    else:
        dates = pd.bdate_range(fc.origin_date + pd.offsets.BDay(1), periods=fc.horizon)
    n = len(dates)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(h.index, h, color="black", lw=1.2, label="history")
    for level, alpha in ((0.95, 0.15), (0.80, 0.30)):
        lo, hi = fc.band(level)
        ax.fill_between(dates, lo[:n], hi[:n], color="tab:blue", alpha=alpha, label=f"{int(level * 100)}% band")
    ax.plot(dates, fc.point[:n], color="tab:blue", lw=1.8, label="forecast (noise-free path)")
    ax.axhline(fc.origin_price, color="grey", ls="--", lw=0.9, label="flat at last close")
    if scored is not None:
        ax.plot(dates, scored["actual"], color="tab:red", lw=1.8, label="actual")
    ax.axvline(fc.origin_date, color="grey", ls=":", lw=0.8)
    ax.set(title=f"{ticker}: {fc.horizon}-day forecast from {fc.origin_date:%Y-%m-%d}"
                 + (" vs what happened" if scored is not None else " (dates ignore exchange holidays)"),
           xlabel="Date", ylabel="Close (USD, adjusted)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig
