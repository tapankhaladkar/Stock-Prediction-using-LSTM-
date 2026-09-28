import numpy as np
import pandas as pd
import pytest

from stock_lstm import forecast as fcm
from stock_lstm import plots
from stock_lstm.evaluate import compare
from stock_lstm.model import ModelConfig, fit_model
from stock_lstm.features import log_returns

ORIGIN = pd.Timestamp("2025-07-18")


class Stub:
    """Stands in for FittedModel with known arithmetic; records every batch it is asked about."""

    def __init__(self, window, fn):
        self.window, self.fn, self.calls = window, fn, []

    def predict_windows(self, w):
        self.calls.append(np.array(w, copy=True))
        return self.fn(w)


def half_of_last(w):
    return 0.5 * w[:, -1]


def zeros(w):
    return np.zeros(len(w))


def test_forecast_starts_from_exactly_the_last_window_and_feeds_predictions_back():
    r = np.random.default_rng(0).normal(0, 0.01, 200)
    stub = Stub(window=10, fn=half_of_last)
    fcm.make_forecast([stub], r, 100.0, ORIGIN, residuals=np.zeros(5), horizon=3, n_paths=4)
    first, second = stub.calls[0], stub.calls[1]          # calls[0] is the noise-free point path
    assert first.shape == (1, 10)
    np.testing.assert_array_equal(first[0], r[-10:])       # the old code passed 98 values + 2 zero pads
    # Recursion: window 2 = window 1 shifted left, with the prediction appended
    np.testing.assert_allclose(second[0, :-1], r[-9:])
    assert second[0, -1] == pytest.approx(0.5 * r[-1])


def test_recursion_is_correct_and_output_is_in_dollars():
    r = np.full(50, 0.0)
    r[-1] = 0.02
    stub = Stub(window=5, fn=half_of_last)
    fc = fcm.make_forecast([stub], r, origin_price=200.0, origin_date=ORIGIN,
                           residuals=np.zeros(3), horizon=4, n_paths=10)
    h = np.arange(1, 5)
    expected = 200.0 * np.exp(0.02 * (1 - 0.5 ** h))       # cum return = r_last * (1 - 0.5^h)
    np.testing.assert_allclose(fc.point, expected)
    np.testing.assert_allclose(fc.paths, np.tile(expected, (10, 1)))   # zero noise -> all paths equal
    assert 190 < fc.point[0] < 215                            # dollars near the last close, not scaled 0..1


def test_bands_widen_with_horizon_and_bracket_the_median():
    rng = np.random.default_rng(1)
    resid = rng.normal(0, 0.01, 2000)
    fc = fcm.make_forecast([Stub(20, zeros)], np.zeros(100), 100.0, ORIGIN, resid,
                           horizon=30, n_paths=4000, seed=5)
    lo, hi = fc.band(0.95)
    width = hi - lo
    assert (lo < fc.median).all() and (fc.median < hi).all()
    assert (np.diff(width) > -0.3).all() and width[-1] > 3 * width[0]       # roughly sqrt(30) ~ 5.5x
    assert width[-1] / width[0] == pytest.approx(np.sqrt(30), rel=0.25)
    lo80, hi80 = fc.band(0.80)
    assert ((hi80 - lo80) < width).all()                                    # 80% band is inside the 95% band


def test_residual_mean_does_not_smuggle_in_a_drift():
    resid = np.random.default_rng(2).normal(0.01, 0.005, 1000)             # biased residuals
    fc = fcm.make_forecast([Stub(5, zeros)], np.zeros(50), 100.0, ORIGIN, resid, horizon=20, n_paths=3000)
    assert fc.median[-1] == pytest.approx(100.0, rel=0.01)                  # would be ~122 uncentred


def test_same_seed_same_fan_and_paths_are_split_across_models():
    resid = np.random.default_rng(3).normal(0, 0.01, 500)
    kw = dict(returns=np.zeros(60), origin_price=100.0, origin_date=ORIGIN, residuals=resid, horizon=5, n_paths=100)
    a, b = Stub(5, zeros), Stub(5, zeros)
    f1 = fcm.make_forecast([a, b], seed=9, **kw)
    f2 = fcm.make_forecast([Stub(5, zeros), Stub(5, zeros)], seed=9, **kw)
    np.testing.assert_array_equal(f1.paths, f2.paths)
    assert f1.paths.shape == (100, 5)
    # one predict call per horizon step per recursion (noise-free point path + simulation)
    assert len(a.calls) == len(b.calls) == 2 * 5
    assert max(len(c) for c in a.calls) == max(len(c) for c in b.calls) == 50     # 100 paths split 50/50


def test_forecast_input_validation():
    kw = dict(origin_price=100.0, origin_date=ORIGIN, residuals=np.zeros(3))
    with pytest.raises(ValueError, match="at least one"):
        fcm.make_forecast([], np.zeros(50), **kw)
    with pytest.raises(ValueError, match="at least 20 returns"):
        fcm.make_forecast([Stub(20, zeros)], np.zeros(19), **kw)


def test_residuals_from_predictions_is_log_ratio():
    idx = pd.bdate_range("2025-01-01", periods=3)
    actual = pd.Series([100.0, 110.0, 121.0], index=idx)
    pred = pd.Series([100.0, 100.0], index=idx[1:])
    np.testing.assert_allclose(fcm.residuals_from_predictions(actual, pred), np.log([1.10, 1.21]))


# ---- scoring against what actually happened ---------------------------------------------------

def _fc(point, spread=1.0, n=400):
    point = np.asarray(point, dtype=float)
    rng = np.random.default_rng(0)
    paths = point + rng.normal(0, spread, size=(n, len(point)))
    return fcm.Forecast(ORIGIN, origin_price=100.0, point=point, paths=paths)


def _actual(values):
    return pd.Series(values, index=pd.bdate_range("2025-07-21", periods=len(values)), dtype=float)


def test_score_perfect_forecast():
    point = [101.0, 102.0, 103.0]
    table, s = fcm.score_forecast(_fc(point), _actual(point))
    assert s["RMSE_forecast"] == 0 and s["days_scored"] == 3
    assert s["coverage_80_%"] == 100 and s["coverage_95_%"] == 100
    assert list(table.columns[:3]) == ["actual", "forecast", "flat_baseline"]


def test_score_far_off_actuals_have_zero_coverage_and_flat_baseline_is_exact():
    table, s = fcm.score_forecast(_fc([101.0, 102.0]), _actual([150.0, 150.0]))
    assert s["coverage_80_%"] == 0 and s["coverage_95_%"] == 0
    assert s["RMSE_flat"] == pytest.approx(50.0)                      # |150 - 100| both days
    assert s["actual_move_%"] == pytest.approx(50.0)


def test_flat_baseline_wins_when_price_does_not_move_and_forecast_drifts():
    _, s = fcm.score_forecast(_fc([95.0, 90.0, 85.0]), _actual([100.0, 100.0, 100.0]))
    assert s["RMSE_flat"] == 0 and s["RMSE_ratio_vs_flat"] == float("inf")


def test_score_only_uses_days_that_actually_exist():
    _, s = fcm.score_forecast(_fc([101.0] * 30), _actual([101.0] * 7))
    assert s["days_scored"] == 7
    with pytest.raises(ValueError, match="no realised prices"):
        fcm.score_forecast(_fc([101.0] * 30), _actual([]))


# ---- with a real (tiny) trained model and the plots --------------------------------------------

def test_real_model_end_to_end_forecast_and_plots():
    rng = np.random.default_rng(0)
    px = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 700))), index=pd.bdate_range("2020-01-01", periods=700))
    r = log_returns(px).to_numpy()
    cfg = ModelConfig(window=10, units=(4, 4), dropout=0.0, epochs=2, patience=1)
    models = [fit_model(r, 500, 600, cfg, seed=s) for s in range(2)]
    fc = fcm.make_forecast(models, r, px.iloc[-1], px.index[-1], residuals=rng.normal(0, 0.01, 200),
                           horizon=10, n_paths=100)
    assert fc.paths.shape == (100, 10) and np.isfinite(fc.paths).all()
    assert abs(fc.point[0] / px.iloc[-1] - 1) < 0.05
    future = pd.Series(px.iloc[-1] * np.exp(np.cumsum(rng.normal(0, 0.01, 10))),
                       index=pd.bdate_range(px.index[-1] + pd.offsets.BDay(1), periods=10))
    table, _ = fcm.score_forecast(fc, future)
    assert plots.plot_forecast(px, fc) is not None
    assert plots.plot_forecast(px, fc, table) is not None


def test_comparison_plots_render():
    idx = pd.bdate_range("2024-01-01", periods=300)
    rng = np.random.default_rng(4)
    px = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))), index=idx)
    preds = pd.DataFrame({"LSTM": px.shift(1) * 1.001, "Persistence": px.shift(1), "fold": 0}).dropna().iloc[100:]
    table = compare(px, {c: preds[c] for c in ("LSTM", "Persistence")})
    assert plots.plot_ratio_ci(table) is not None
    assert plots.plot_walk_forward(px, preds) is not None
    raw = pd.DataFrame({"date": idx, "close": px.to_numpy() * 4, "adjClose": px.to_numpy()})
    assert plots.plot_adjusted_vs_raw(raw) is not None


def test_fan_bands_are_calibrated_when_the_noise_model_is_right():
    """If real daily errors follow the residual distribution the fan resamples, the terminal
    80% / 95% bands must contain the truth about 80% / 95% of the time."""
    sigma, horizon = 0.01, 30
    rng = np.random.default_rng(11)
    resid = rng.normal(0, sigma, 5000)
    fc = fcm.make_forecast([Stub(5, zeros)], np.zeros(50), 100.0, ORIGIN, resid,
                           horizon=horizon, n_paths=4000, seed=1)
    truth = 100.0 * np.exp(rng.normal(0, sigma, size=(3000, horizon)).sum(axis=1))
    for level in (0.80, 0.95):
        lo, hi = fc.band(level)
        covered = np.mean((truth >= lo[-1]) & (truth <= hi[-1]))
        assert covered == pytest.approx(level, abs=0.03)
