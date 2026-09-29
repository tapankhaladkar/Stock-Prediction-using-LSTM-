"""Rolling-origin backtest of the multi-day forecast.

One 30-day forecast scored against one realised path says almost nothing: it is a single,
strongly autocorrelated sample. Here a 30-day forecast is started from many origins spread
across the walk-forward test days, and each is scored against what actually followed.

Two rules keep it honest:

* An origin uses the model of *its own fold*, i.e. one trained before the fold's test block,
  so the model has never seen the days being forecast (it is up to ``test_size`` days stale,
  like a model retrained twice a year).
* The fan at an origin is built only from one-step errors already observed by that day
  (residuals from the fold's validation slice up to the origin), never from later ones, and the
  drift benchmark only from returns up to the origin.

Consecutive origins overlap when ``step < horizon``, so every interval below is a moving-block
bootstrap over *origins*, not over days.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import log_returns
from .forecast import make_forecast, score_forecast
from .metrics import bootstrap_mean_ci, bootstrap_mse_ratio
from .walkforward import Fold


@dataclass(frozen=True)
class RollingConfig:
    horizon: int = 30        # trading days per forecast
    step: int = 21           # trading days between origins (~monthly)
    n_paths: int = 600       # simulated paths per origin, split across the fold's seed models
    block: int = 3           # bootstrap block length, in origins (3 x 21 days covers one horizon)
    seed: int = 0


def origin_positions(fold: Fold, n_returns: int, cfg: RollingConfig) -> list[int]:
    """Positions ``p`` in the returns array of the last day known at each origin of ``fold``.

    The first forecast day is ``p + 1``. The earliest origin is ``val_end - 1``, so the first
    forecast day is the first test day; and ``p + horizon`` must exist, so every forecast is
    scored on its full horizon.
    """
    return [p for p in range(fold.val_end - 1, fold.test_end - 1, cfg.step)
            if p + cfg.horizon <= n_returns - 1]


@dataclass
class RollingResult:
    origins: pd.DataFrame          # one row per origin
    config: RollingConfig

    def summary(self, n_boot: int = 2000) -> tuple[pd.DataFrame, dict]:
        """(error table, coverage) pooled over all origins.

        Table rows are Forecast / Drift / Flat; RMSE is ``sqrt(mean of per-origin MSE)``, and the
        ratios (below 1 = better) carry 95% block-bootstrap intervals over origins. ``coverage`` is
        the share of realised closes inside the 80% / 95% bands, again with intervals.
        """
        o, cfg = self.origins, self.config
        mse = {name: o[f"RMSE_{col}"].to_numpy() ** 2
               for name, col in (("Forecast", "forecast"), ("Drift", "drift"), ("Flat", "flat"))}
        mae = {"Forecast": o["MAE_forecast"], "Drift": o["MAE_drift"], "Flat": o["MAE_flat"]}

        def ratio(a: str, b: str):
            point = float(np.sqrt(mse[a].mean() / mse[b].mean()))
            lo, hi = bootstrap_mse_ratio(mse[a], mse[b], block=cfg.block, n_boot=n_boot, seed=cfg.seed)
            return point, lo, hi

        rows = {}
        for name in ("Forecast", "Drift", "Flat"):
            row = {"RMSE": float(np.sqrt(mse[name].mean())), "MAE": float(mae[name].mean())}
            for base in ("Flat", "Drift"):
                key = f"vs_{base}"
                if name == base:
                    row[key], row[key + "_lo"], row[key + "_hi"] = 1.0, 1.0, 1.0
                else:
                    row[key], row[key + "_lo"], row[key + "_hi"] = ratio(name, base)
            rows[name] = row
        table = pd.DataFrame(rows).T
        table.attrs.update(
            n_origins=len(o),
            wins_vs_flat=float((o["RMSE_forecast"] < o["RMSE_flat"]).mean() * 100),
            wins_vs_drift=float((o["RMSE_forecast"] < o["RMSE_drift"]).mean() * 100),
        )

        coverage = {}
        for level in (80, 95):
            per_origin = o[f"coverage_{level}_%"].to_numpy()
            lo, hi = bootstrap_mean_ci(per_origin, block=cfg.block, n_boot=n_boot, seed=cfg.seed)
            coverage[level] = {"mean_%": float(per_origin.mean()), "lo_%": lo, "hi_%": hi}
        return table, coverage


def rolling_origin_backtest(prices: pd.Series, folds: list[Fold], fold_models: dict[int, list],
                            cfg: RollingConfig = RollingConfig(), progress=print) -> RollingResult:
    """Score a ``cfg.horizon``-day forecast from many origins inside each fold's test block.

    ``fold_models[k]`` are the fitted models (one per seed) of fold ``k``, as kept by
    ``run_walk_forward``. Nothing is retrained.
    """
    rets = log_returns(prices)
    r, dates = rets.to_numpy(), rets.index
    n = len(r)

    rows = []
    for f in folds:
        models = fold_models[f.index]
        origins = origin_positions(f, n, cfg)
        if not origins:
            continue
        # One-step errors the fold's models made from train_end onward; an origin may use only
        # those up to its own day (r[p] is known at origin p, and its prediction used r before p).
        pred = np.mean([m.predict_returns(r, f.train_end, f.test_end) for m in models], axis=0)
        errors = r[f.train_end:f.test_end] - pred
        for p in origins:
            fc = make_forecast(
                models, r[:p + 1], origin_price=float(prices.loc[dates[p]]), origin_date=dates[p],
                residuals=errors[:p + 1 - f.train_end], horizon=cfg.horizon,
                n_paths=cfg.n_paths, seed=cfg.seed + p,
            )
            _, s = score_forecast(fc, prices.loc[dates[p + 1:p + 1 + cfg.horizon]])
            lo80, hi80 = fc.band(0.80)
            rows.append({"fold": f.index, "origin_date": dates[p], "origin_price": fc.origin_price, **s,
                         # how wide the fan was at the end of the horizon, as % of the origin price
                         "band80_width_%": float((hi80[-1] - lo80[-1]) / fc.origin_price * 100)})
        if progress:
            progress(f"fold {f.index}: {len(origins)} origins")

    if not rows:
        raise ValueError("no forecast origins fit inside the folds; use a longer history or a shorter horizon")
    return RollingResult(pd.DataFrame(rows), cfg)
