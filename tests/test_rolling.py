import numpy as np
import pandas as pd
import pytest

from stock_lstm import plots
from stock_lstm.features import log_returns
from stock_lstm.rolling import RollingConfig, RollingResult, origin_positions, rolling_origin_backtest
from stock_lstm.walkforward import Fold


class Stub:
    """A fitted model with known behaviour: predicts fn(window). Needs no training."""

    def __init__(self, window, fn):
        self.window, self.fn = window, fn

    def predict_windows(self, w):
        return self.fn(np.asarray(w))

    def predict_returns(self, returns, start, stop):
        w = np.stack([returns[t - self.window:t] for t in range(start, stop)])
        return self.fn(w)


def const(c):
    return lambda w: np.full(len(w), float(c))


def half_of_last(w):
    return 0.5 * w[:, -1]


def make_prices(n_returns, mu=0.0, sigma=0.01, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.normal(mu, sigma, n_returns)
    idx = pd.bdate_range("2015-01-01", periods=n_returns + 1)
    return pd.Series(100 * np.exp(np.concatenate([[0.0], np.cumsum(r)])), index=idx)


CFG = RollingConfig(horizon=30, step=21, n_paths=200, block=3, seed=1)


# ---- which origins exist ------------------------------------------------------------------------

def test_origins_start_at_the_last_validation_day_and_fit_their_full_horizon():
    f = Fold(0, train_end=300, val_end=400, test_end=520)
    ps = origin_positions(f, n_returns=560, cfg=CFG)
    assert ps[0] == f.val_end - 1                     # first forecast day is the first test day
    assert all(b - a == CFG.step for a, b in zip(ps, ps[1:]))
    assert all(f.val_end - 1 <= p < f.test_end - 1 for p in ps)
    assert all(p + CFG.horizon <= 560 - 1 for p in ps)


def test_no_origin_is_created_when_the_horizon_would_run_past_the_data():
    last = Fold(1, train_end=400, val_end=520, test_end=640)
    assert origin_positions(last, n_returns=560, cfg=CFG) == [519]      # 540 + 30 > 559
    assert origin_positions(last, n_returns=540, cfg=CFG) == []


def test_no_forecast_day_is_ever_inside_the_data_a_model_trained_or_stopped_on():
    """Every forecast day is >= val_end of the fold whose model makes it."""
    prices = make_prices(700)
    folds = [Fold(0, 250, 350, 470), Fold(1, 370, 470, 590)]
    models = {0: [Stub(5, const(0))], 1: [Stub(5, const(0))]}
    res = rolling_origin_backtest(prices, folds, models, CFG, progress=None)
    dates = log_returns(prices).index
    for _, row in res.origins.iterrows():
        f = folds[int(row["fold"])]
        first_forecast_day = dates.get_loc(row["origin_date"]) + 1
        assert first_forecast_day >= f.val_end


# ---- each origin uses its own fold's model ------------------------------------------------------

def test_each_origin_uses_the_model_of_its_own_fold():
    prices = make_prices(700)
    folds = [Fold(0, 250, 350, 470), Fold(1, 370, 470, 590)]
    c = {0: 0.002, 1: -0.003}
    models = {k: [Stub(5, const(c[k]))] for k in c}
    res = rolling_origin_backtest(prices, folds, models, CFG, progress=None)
    for _, row in res.origins.iterrows():
        k = int(row["fold"])
        expected = (np.exp(c[k] * CFG.horizon) - 1) * 100     # noise-free path grows at exactly c_k per day
        assert row["forecast_move_%"] == pytest.approx(expected)
    assert set(res.origins["fold"]) == {0, 1}


# ---- causality ----------------------------------------------------------------------------------

def test_a_forecast_never_depends_on_prices_after_its_origin():
    prices = make_prices(700, seed=3)
    folds = [Fold(0, 250, 350, 470), Fold(1, 370, 470, 590)]
    models = {k: [Stub(5, half_of_last), Stub(5, const(0.0005))] for k in (0, 1)}
    base = rolling_origin_backtest(prices, folds, models, CFG, progress=None).origins

    tp = 520                                                  # tamper with every price from here on
    tampered = prices.copy()
    tampered.iloc[tp:] *= 1.25
    alt = rolling_origin_backtest(tampered, folds, models, CFG, progress=None).origins

    cutoff = prices.index[tp]
    early = base["origin_date"] < cutoff
    assert early.any() and (~early).any()                     # both sides exist, so this is not vacuous
    # band80_width_% comes from the residuals: it catches errors observed *after* the origin leaking in
    cols = ["origin_price", "forecast_move_%", "drift_move_%", "band80_width_%"]
    pd.testing.assert_frame_equal(base.loc[early, cols], alt.loc[early, cols])
    # Origins after the tampering do see it (their mean-return benchmark includes the jump)...
    assert not np.allclose(base.loc[~early, "drift_move_%"], alt.loc[~early, "drift_move_%"])
    assert not np.allclose(base.loc[~early, "band80_width_%"], alt.loc[~early, "band80_width_%"])
    # ...and what happened after the early origins changed too, so the scoring sees the tampering:
    assert not np.allclose(base.loc[early, "actual_move_%"], alt.loc[early, "actual_move_%"])


# ---- calibration --------------------------------------------------------------------------------

def test_bands_cover_about_80_and_95_percent_across_many_origins_when_the_noise_model_is_right():
    """iid returns, a model that predicts zero: the fan (built from errors seen so far) should
    contain the realised close on ~80% / ~95% of forecast days across ~270 origins."""
    n = 9000
    prices = make_prices(n, sigma=0.01, seed=9)
    fold = Fold(0, train_end=500, val_end=800, test_end=n - 30)
    cfg = RollingConfig(horizon=30, step=30, n_paths=800, block=3, seed=2)
    res = rolling_origin_backtest(prices, [fold], {0: [Stub(5, const(0.0))]}, cfg, progress=None)
    assert len(res.origins) > 250
    _, cov = res.summary(n_boot=300)
    assert cov[80]["mean_%"] == pytest.approx(80, abs=4)
    assert cov[95]["mean_%"] == pytest.approx(95, abs=3)
    for level in (80, 95):
        assert cov[level]["lo_%"] <= cov[level]["mean_%"] <= cov[level]["hi_%"]


# ---- summary ------------------------------------------------------------------------------------

def _origins(rmse_f, rmse_flat, rmse_drift, cov80=None, cov95=None):
    n = len(rmse_f)
    return pd.DataFrame({
        "fold": 0, "origin_date": pd.bdate_range("2024-01-01", periods=n), "origin_price": 100.0,
        "RMSE_forecast": rmse_f, "RMSE_flat": rmse_flat, "RMSE_drift": rmse_drift,
        "MAE_forecast": rmse_f, "MAE_flat": rmse_flat, "MAE_drift": rmse_drift,
        "coverage_80_%": cov80 if cov80 is not None else [80.0] * n,
        "coverage_95_%": cov95 if cov95 is not None else [95.0] * n,
    })


def test_summary_pools_mse_across_origins_and_reports_ratios_wins_and_coverage():
    o = _origins(rmse_f=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], rmse_flat=[2.0] * 6, rmse_drift=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    table, cov = RollingResult(o, CFG).summary(n_boot=200)
    assert table.loc["Forecast", "RMSE"] == pytest.approx(np.sqrt(np.mean(np.arange(1, 7) ** 2)))
    assert table.loc["Flat", "RMSE"] == pytest.approx(2.0)
    assert table.loc["Forecast", "vs_Drift"] == pytest.approx(1.0)                # identical to drift
    assert table.loc["Forecast", "vs_Flat"] == pytest.approx(np.sqrt(np.mean(np.arange(1, 7) ** 2)) / 2.0)
    assert table.loc["Flat", "vs_Flat"] == 1.0
    assert table.attrs["n_origins"] == 6 and table.attrs["wins_vs_flat"] == pytest.approx(100 / 6)
    assert table.attrs["wins_vs_drift"] == 0.0                                      # ties are not wins
    assert cov[80]["mean_%"] == 80.0 and cov[95]["mean_%"] == 95.0


def test_backtest_refuses_to_run_when_no_origin_fits():
    prices = make_prices(300)
    with pytest.raises(ValueError, match="no forecast origins"):
        rolling_origin_backtest(prices, [Fold(0, 100, 200, 250)], {0: [Stub(5, const(0))]},
                                RollingConfig(horizon=120, step=21), progress=None)


def test_rolling_plot_renders():
    o = _origins(rmse_f=[1.0, 2.0, 1.5, 2.5], rmse_flat=[1.5, 1.8, 1.6, 2.0], rmse_drift=[1.2, 2.1, 1.4, 2.4],
                 cov80=[75.0, 85.0, 80.0, 90.0], cov95=[93.0, 97.0, 95.0, 100.0])
    assert plots.plot_rolling_backtest(RollingResult(o, CFG)) is not None


@pytest.mark.slow
def test_walk_forward_keeps_its_fold_models_and_the_rolling_backtest_runs_on_real_networks():
    from stock_lstm.walkforward import WalkForwardConfig, run_walk_forward
    from tests.test_walkforward import TINY, synthetic_prices

    px = synthetic_prices(n=900)
    wf = WalkForwardConfig(n_folds=2, test_size=60, val_size=60, n_seeds=2, ma_windows=(5,))
    res = run_walk_forward(px, TINY, wf, progress=None)
    assert set(res.fold_models) == {0, 1} and all(len(v) == 2 for v in res.fold_models.values())

    roll = rolling_origin_backtest(px, res.folds, res.fold_models,
                                   RollingConfig(horizon=10, step=15, n_paths=40), progress=None)
    assert len(roll.origins) >= 4 and roll.origins["fold"].nunique() == 2
    table, cov = roll.summary(n_boot=100)
    assert np.isfinite(table[["RMSE", "vs_Flat", "vs_Drift"]].to_numpy()).all()
    assert set(cov) == {80, 95}
