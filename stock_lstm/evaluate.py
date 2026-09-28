"""Score several one-step-ahead forecasts on the same days, in dollars."""
from __future__ import annotations

import pandas as pd

from .metrics import bootstrap_rmse_ratio, score, up_day_rate


def compare(prices: pd.Series, predictions: dict[str, pd.Series], reference: str = "Persistence",
            ci: bool = True) -> pd.DataFrame:
    """One row per model, scored on the days where *every* model has a prediction.

    ``RMSE_vs_<reference>`` < 1 means better than the reference; with ``ci`` it also gives a
    95% block-bootstrap interval, because a point ratio of 0.98 on ~500 days is noise.
    """
    preds = pd.DataFrame(predictions)
    days = preds.dropna().index.intersection(prices.index[1:])
    if len(days) == 0:
        raise ValueError("no common days with predictions from every model")
    actual = prices.loc[days].to_numpy()
    prev = prices.shift(1).loc[days].to_numpy()

    rows = {name: score(actual, preds.loc[days, name].to_numpy(), prev) for name in preds.columns}
    table = pd.DataFrame(rows).T
    if reference in preds.columns:
        ref = preds.loc[days, reference].to_numpy()
        table[f"RMSE_vs_{reference}"] = table["RMSE"] / table.loc[reference, "RMSE"]
        if ci:
            bounds = {name: bootstrap_rmse_ratio(actual, preds.loc[days, name].to_numpy(), ref)
                      if name != reference else (1.0, 1.0) for name in preds.columns}
            table["ratio_CI95_lo"] = [bounds[n][0] for n in table.index]
            table["ratio_CI95_hi"] = [bounds[n][1] for n in table.index]
    table.attrs["n_days"] = len(days)
    table.attrs["up_day_rate_%"] = up_day_rate(actual, prev)
    return table
