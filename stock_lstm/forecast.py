"""Multi-day forecast as a fan of simulated paths, plus a backtest against realised prices.

A single recursive line ("predict tomorrow, feed it back, repeat") looks precise but is not:
errors compound and the model's own outputs quickly become its inputs. Instead we simulate
many paths, adding a resampled out-of-sample residual to each predicted return before it is
fed back, and report the median path with 80% / 95% bands. The residuals come from the
walk-forward test days, i.e. from errors the model actually made on data it had not seen.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model import FittedModel


@dataclass(frozen=True)
class Forecast:
    origin_date: pd.Timestamp
    origin_price: float
    point: np.ndarray        # (h,) prices from the noise-free recursion, averaged over models
    paths: np.ndarray        # (n_paths, h) simulated prices
    drift_path: np.ndarray | None = None   # (h,) constant-drift benchmark: origin price grown at the mean past return

    @property
    def horizon(self) -> int:
        return self.point.shape[0]

    def band(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        q = (1 - level) / 2
        return np.quantile(self.paths, q, axis=0), np.quantile(self.paths, 1 - q, axis=0)

    @property
    def median(self) -> np.ndarray:
        return np.median(self.paths, axis=0)


def residuals_from_predictions(actual: pd.Series, predicted: pd.Series) -> np.ndarray:
    """One-step log-return residuals log(actual / predicted), from out-of-sample predictions."""
    days = predicted.index
    return np.log(actual.loc[days].to_numpy() / predicted.to_numpy())


def _recurse(model: FittedModel, last_window: np.ndarray, horizon: int, eps: np.ndarray) -> np.ndarray:
    """Cumulative log return for each row of ``eps`` (n_paths, horizon), feeding each sampled
    return back into that path's window."""
    n = eps.shape[0]
    windows = np.tile(np.asarray(last_window, dtype=float), (n, 1))
    cum = np.zeros((n, horizon))
    running = np.zeros(n)
    for h in range(horizon):
        r_next = model.predict_windows(windows) + eps[:, h]
        running = running + r_next
        cum[:, h] = running
        windows = np.concatenate([windows[:, 1:], r_next[:, None]], axis=1)
    return cum


def make_forecast(models: list[FittedModel], returns: np.ndarray, origin_price: float,
                  origin_date: pd.Timestamp, residuals: np.ndarray, horizon: int = 30,
                  n_paths: int = 1000, seed: int = 0) -> Forecast:
    """Forecast ``horizon`` days after the end of ``returns``.

    The forecast always starts from exactly the last ``window`` returns; nothing is padded.
    ``n_paths`` is split across ``models`` (e.g. one per seed) so the fan reflects model
    variability as well as day-to-day noise.
    """
    if not models:
        raise ValueError("need at least one fitted model")
    returns = np.asarray(returns, dtype=float)
    centred = np.asarray(residuals, dtype=float)
    centred = centred - centred.mean()               # noise only; drift stays the model's own
    rng = np.random.default_rng(seed)

    points, paths = [], []
    per_model = max(1, n_paths // len(models))
    for m in models:
        if len(returns) < m.window:
            raise ValueError(f"need at least {m.window} returns, have {len(returns)}")
        last = returns[-m.window:]
        zero = np.zeros((1, horizon))
        points.append(origin_price * np.exp(_recurse(m, last, horizon, zero)[0]))
        eps = rng.choice(centred, size=(per_model, horizon), replace=True)
        paths.append(origin_price * np.exp(_recurse(m, last, horizon, eps)))
    # Benchmark that needs no model: keep growing at the average log return seen up to the origin.
    drift_path = origin_price * np.exp(returns.mean() * np.arange(1, horizon + 1))
    return Forecast(origin_date=pd.Timestamp(origin_date), origin_price=float(origin_price),
                    point=np.mean(points, axis=0), paths=np.concatenate(paths), drift_path=drift_path)


def _ratio(model: float, base: float) -> float:
    """model / base, where a base that is exactly perfect gives inf (or nan if the model is too)."""
    return model / base if base > 0 else (float("nan") if model == 0 else float("inf"))


def score_forecast(fc: Forecast, actual: pd.Series) -> tuple[pd.DataFrame, dict]:
    """Compare a forecast with the closes that actually followed, day by day.

    ``actual`` holds closes *after* the origin, in order; trading days are matched by
    position (1st close after origin = horizon day 1). Two model-free benchmarks: "flat at
    the origin close" and, when the forecast carries one, constant drift at the mean return
    seen before the origin. This is one path, so treat the result as an illustration.
    """
    n = min(len(actual), fc.horizon)
    if n == 0:
        raise ValueError("no realised prices after the forecast origin to score against")
    act = actual.iloc[:n].to_numpy(dtype=float)
    lo80, hi80 = (b[:n] for b in fc.band(0.80))
    lo95, hi95 = (b[:n] for b in fc.band(0.95))
    table = pd.DataFrame({
        "actual": act, "forecast": fc.point[:n], "flat_baseline": fc.origin_price,
        "lo80": lo80, "hi80": hi80, "lo95": lo95, "hi95": hi95,
    }, index=actual.index[:n])
    if fc.drift_path is not None:
        table.insert(3, "drift_baseline", fc.drift_path[:n])
    table.index.name = "date"

    rmse = lambda err: float(np.sqrt(np.mean(err ** 2)))
    mae = lambda err: float(np.mean(np.abs(err)))
    err_f, err_b = act - table["forecast"].to_numpy(), act - fc.origin_price
    summary = {
        "days_scored": n,
        "RMSE_forecast": rmse(err_f),
        "RMSE_flat": rmse(err_b),
        "MAE_forecast": mae(err_f),
        "MAE_flat": mae(err_b),
        "coverage_80_%": float(np.mean((act >= lo80) & (act <= hi80)) * 100),
        "coverage_95_%": float(np.mean((act >= lo95) & (act <= hi95)) * 100),
        "actual_move_%": float((act[-1] / fc.origin_price - 1) * 100),
        "forecast_move_%": float((fc.point[n - 1] / fc.origin_price - 1) * 100),
    }
    summary["RMSE_ratio_vs_flat"] = _ratio(summary["RMSE_forecast"], summary["RMSE_flat"])
    if fc.drift_path is not None:
        err_d = act - table["drift_baseline"].to_numpy()
        summary["RMSE_drift"] = rmse(err_d)
        summary["MAE_drift"] = mae(err_d)
        summary["drift_move_%"] = float((fc.drift_path[n - 1] / fc.origin_price - 1) * 100)
        summary["RMSE_ratio_vs_drift"] = _ratio(summary["RMSE_forecast"], summary["RMSE_drift"])
    return table, summary
