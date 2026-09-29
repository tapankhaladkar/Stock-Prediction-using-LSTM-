"""Walk-forward (expanding window) evaluation of the LSTM against naive baselines.

Layout of each fold, in strict time order, over the array of daily log returns::

    [ ...........train........... | ..val.. | ..test.. ]

The network trains on ``train``, stops early on ``val``, and is scored on ``test``, which it
has never influenced. Fold k+1 slides the boundaries forward by ``test_size`` and retrains
from scratch, so the score covers several market regimes instead of one lucky split.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import baselines
from .data import validate_prices
from .evaluate import compare
from .features import log_returns
from .model import ModelConfig, fit_model


@dataclass(frozen=True)
class WalkForwardConfig:
    n_folds: int = 5
    test_size: int = 126      # ~6 months of trading days per fold
    val_size: int = 126
    n_seeds: int = 3
    ma_windows: tuple[int, ...] = (5, 20)
    arima_order: tuple[int, int, int] = (1, 1, 1)


@dataclass(frozen=True)
class Fold:
    index: int
    train_end: int   # train = returns[:train_end]
    val_end: int     # val   = returns[train_end:val_end]
    test_end: int    # test  = returns[val_end:test_end]


def make_folds(n_returns: int, wf: WalkForwardConfig, window: int, min_train_windows: int = 250) -> list[Fold]:
    """Contiguous test blocks that exactly tile the last ``n_folds * test_size`` returns."""
    first_test = n_returns - wf.n_folds * wf.test_size
    folds = []
    for k in range(wf.n_folds):
        val_end = first_test + k * wf.test_size
        folds.append(Fold(k, train_end=val_end - wf.val_size, val_end=val_end, test_end=val_end + wf.test_size))
    if folds[0].train_end - window < min_train_windows:
        raise ValueError(
            f"only {folds[0].train_end - window} training windows in the first fold (< {min_train_windows}); "
            "use a longer history or fewer/smaller folds"
        )
    return folds


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame                    # one row per test day; columns = models + "fold"
    seed_predictions: dict[int, pd.Series]       # LSTM prediction per seed (LSTM column is their mean)
    fold_info: list[dict] = field(default_factory=list)
    folds: list[Fold] = field(default_factory=list)
    fold_models: dict[int, list] = field(default_factory=dict)   # fold index -> fitted models, one per seed

    def summary(self, prices: pd.Series, **kw) -> pd.DataFrame:
        models = [c for c in self.predictions.columns if c != "fold"]
        return compare(prices, {m: self.predictions[m] for m in models}, **kw)

    def per_fold_rmse(self, prices: pd.Series) -> pd.DataFrame:
        rows = {}
        for k, chunk in self.predictions.groupby("fold"):
            models = [c for c in chunk.columns if c != "fold"]
            rows[k] = compare(prices, {m: chunk[m] for m in models}, ci=False)["RMSE"]
        out = pd.DataFrame(rows).T
        out.index.name = "fold"
        return out

    def seed_rmse(self, prices: pd.Series) -> pd.Series:
        """Dollar RMSE of each individual seed (the spread shows run-to-run noise)."""
        days = self.predictions.index
        actual = prices.loc[days].to_numpy()
        return pd.Series({s: float(np.sqrt(np.mean((actual - p.loc[days].to_numpy()) ** 2)))
                          for s, p in self.seed_predictions.items()}, name="RMSE")


def run_walk_forward(prices: pd.Series, model_cfg: ModelConfig = ModelConfig(),
                     wf: WalkForwardConfig = WalkForwardConfig(), progress=print) -> WalkForwardResult:
    validate_prices(prices)
    rets = log_returns(prices)
    r = rets.to_numpy()
    dates = rets.index
    prev_price = prices.shift(1).loc[dates].to_numpy()
    folds = make_folds(len(r), wf, model_cfg.window)

    static = {"Persistence": baselines.persistence(prices), "Drift": baselines.drift(prices)}
    for w in wf.ma_windows:
        static[f"MA({w})"] = baselines.moving_average(prices, w)

    frames, seed_parts, info = [], {s: [] for s in range(wf.n_seeds)}, []
    fold_models: dict[int, list] = {}
    for f in folds:
        test = slice(f.val_end, f.test_end)
        test_dates = dates[test]
        per_seed = []
        fold_models[f.index] = []
        for seed in range(wf.n_seeds):
            m = fit_model(r, f.train_end, f.val_end, model_cfg, seed=seed)
            fold_models[f.index].append(m)
            pred_r = m.predict_returns(r, f.val_end, f.test_end)
            price_pred = pd.Series(prev_price[test] * np.exp(pred_r), index=test_dates)
            per_seed.append(price_pred)
            seed_parts[seed].append(price_pred)
            info.append({"fold": f.index, "seed": seed, "epochs": m.epochs_run, "best_val_loss": m.best_val_loss,
                         "train_end": dates[f.train_end - 1], "test_start": test_dates[0], "test_end": test_dates[-1]})
            if progress:
                progress(f"fold {f.index} seed {seed}: {m.epochs_run} epochs, best val loss {m.best_val_loss:.4f}")
        frame = pd.DataFrame({"LSTM": pd.concat(per_seed, axis=1).mean(axis=1)})
        for name, series in static.items():
            frame[name] = series.loc[test_dates]
        # ARIMA is fit on everything before the test block; prices position of the first test day is val_end + 1
        arima = baselines.arima_one_step(prices, n_train=f.val_end + 1, order=wf.arima_order)
        frame[arima.name] = arima.loc[test_dates]
        frame["fold"] = f.index
        frames.append(frame)

    preds = pd.concat(frames)
    if preds.isna().any().any():  # never drop test days silently
        raise RuntimeError("unexpected NaN in walk-forward predictions:\n" + str(preds.isna().sum()))
    return WalkForwardResult(
        predictions=preds,
        seed_predictions={s: pd.concat(p) for s, p in seed_parts.items()},
        fold_info=info,
        folds=folds,
        fold_models=fold_models,
    )
