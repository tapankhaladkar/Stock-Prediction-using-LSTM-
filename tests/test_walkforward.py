import numpy as np
import pandas as pd
import pytest

from stock_lstm.features import log_returns
from stock_lstm.model import ModelConfig, fit_model
from stock_lstm.walkforward import WalkForwardConfig, make_folds, run_walk_forward

TINY = ModelConfig(window=10, units=(4, 4), dropout=0.0, epochs=3, patience=2, batch_size=64)


def synthetic_prices(n=900, kind="rw", phi=0.5, seed=0, sigma=0.012):
    rng = np.random.default_rng(seed)
    e = rng.normal(0, sigma, n)
    if kind == "ar1":
        r = np.zeros(n)
        for t in range(1, n):
            r[t] = phi * r[t - 1] + e[t]
    else:
        r = e
    return pd.Series(100 * np.exp(np.cumsum(r)), index=pd.bdate_range("2016-01-01", periods=n))


# ---- fold layout ---------------------------------------------------------------------

def test_folds_tile_the_end_of_the_series_in_strict_time_order():
    wf = WalkForwardConfig(n_folds=4, test_size=50, val_size=40)
    folds = make_folds(n_returns=1000, wf=wf, window=20, min_train_windows=100)
    assert folds[-1].test_end == 1000
    for prev, nxt in zip(folds, folds[1:]):
        assert nxt.val_end == prev.test_end                    # test blocks are contiguous, no gaps/overlap
    for f in folds:
        assert f.train_end < f.val_end < f.test_end
        assert f.val_end - f.train_end == 40 and f.test_end - f.val_end == 50
    assert sum(f.test_end - f.val_end for f in folds) == 4 * 50


def test_folds_refuse_to_train_on_too_little_data():
    with pytest.raises(ValueError, match="training windows"):
        make_folds(300, WalkForwardConfig(n_folds=3, test_size=60, val_size=60), window=60)


# ---- leakage guards on a single fit ---------------------------------------------------------

def test_scaler_and_weights_ignore_everything_after_val_end():
    r = log_returns(synthetic_prices()).to_numpy()
    train_end, val_end = 500, 600
    base = fit_model(r, train_end, val_end, TINY, seed=0)

    tampered = r.copy()
    tampered[val_end:] = 5.0                                   # absurd 'future'
    other = fit_model(tampered, train_end, val_end, TINY, seed=0)

    assert base.scaler == other.scaler                          # statistics: train slice only
    assert base.scaler.mean == pytest.approx(r[:train_end].mean())
    np.testing.assert_array_equal(base.predict_returns(r, 600, 650), other.predict_returns(r, 600, 650))
    # ...while the val slice does matter (it drives early stopping), so this isn't vacuous:
    assert base.best_val_loss == other.best_val_loss


def test_same_seed_reproduces_and_different_seed_differs():
    r = log_returns(synthetic_prices()).to_numpy()
    a = fit_model(r, 500, 600, TINY, seed=3).predict_returns(r, 600, 650)
    b = fit_model(r, 500, 600, TINY, seed=3).predict_returns(r, 600, 650)
    c = fit_model(r, 500, 600, TINY, seed=4).predict_returns(r, 600, 650)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_fit_model_validates_its_boundaries():
    r = np.random.default_rng(0).normal(0, 0.01, 300)
    for bad in [(5, 100), (200, 100), (100, 301)]:
        with pytest.raises(ValueError):
            fit_model(r, *bad, TINY, seed=0)


def test_predict_next_demands_exactly_one_window_the_old_98_vs_100_bug():
    r = log_returns(synthetic_prices()).to_numpy()
    m = fit_model(r, 500, 600, TINY, seed=0)
    for wrong in (r[-9:], r[-11:], r[-10:].reshape(1, -1), r[-10:].reshape(-1, 1)):
        with pytest.raises(ValueError, match="exactly 10"):
            m.predict_next(wrong)
    m.predict_next(r[-10:])


def test_predict_next_matches_batch_prediction_for_the_same_window():
    r = log_returns(synthetic_prices()).to_numpy()
    m = fit_model(r, 500, 600, TINY, seed=0)
    t = 620
    batch = m.predict_returns(r, t, t + 1)[0]
    single = m.predict_next(r[t - 10:t])
    assert single == pytest.approx(batch, abs=1e-6)


# ---- end to end -----------------------------------------------------------------------

WF = WalkForwardConfig(n_folds=2, test_size=60, val_size=60, n_seeds=1, ma_windows=(5,))


@pytest.mark.slow
def test_no_prediction_depends_on_the_future_end_to_end():
    """Tamper with prices from day d onward: every prediction through day d (all models, both
    folds) must be bit-identical, because each one only uses data strictly before its own day.
    Predictions after d must change, or the test proves nothing."""
    px = synthetic_prices(n=900)
    base = run_walk_forward(px, TINY, WF, progress=None).predictions

    d_pos = len(px) - 30                                       # inside the last test block
    tampered = px.copy()
    tampered.iloc[d_pos:] *= 1.25       # (x1.5 would trip the unadjusted-split guard, which is a good sign)
    alt = run_walk_forward(tampered, TINY, WF, progress=None).predictions

    d = px.index[d_pos]
    models = [c for c in base.columns if c != "fold"]
    assert {"LSTM", "Persistence", "Drift"} <= set(models)
    pd.testing.assert_frame_equal(base.loc[:d, models], alt.loc[:d, models])
    assert not np.allclose(base.loc[d:, "LSTM"].iloc[1:], alt.loc[d:, "LSTM"].iloc[1:])


# Controls need enough test days (3 x 120) for the bootstrap interval to mean something.
WF_CTRL = WalkForwardConfig(n_folds=3, test_size=120, val_size=120, n_seeds=1, ma_windows=(5,))
CTRL_CFG = ModelConfig(window=20, units=(16, 16), dropout=0.0, epochs=30, patience=5)


@pytest.mark.slow
def test_pipeline_finds_no_skill_on_a_random_walk():
    """Negative control: nothing to learn, so the LSTM must not look meaningfully better than
    'tomorrow = today'. (Observed ~1.004; a real edge would show ratio well below 1.)"""
    px = synthetic_prices(n=1600, kind="rw")
    tab = run_walk_forward(px, CTRL_CFG, WF_CTRL, progress=None).summary(px)
    assert 0.97 < tab.loc["LSTM", "RMSE_vs_Persistence"] < 1.08


@pytest.mark.slow
def test_pipeline_detects_real_skill_when_it_exists():
    """Positive control: returns with genuine autocorrelation (phi=0.5) are predictable, so the
    same pipeline must show the LSTM beating persistence. (Observed ~0.91; optimum ~0.87.)"""
    px = synthetic_prices(n=1600, kind="ar1", phi=0.5)
    tab = run_walk_forward(px, CTRL_CFG, WF_CTRL, progress=None).summary(px)
    assert tab.loc["LSTM", "RMSE_vs_Persistence"] < 0.97
    assert tab.loc["LSTM", "ratio_CI95_hi"] < 1.0
    assert tab.loc["LSTM", "DirAcc_%"] > 55


def test_predict_windows_matches_predict_returns_and_rejects_wrong_widths():
    r = log_returns(synthetic_prices()).to_numpy()
    m = fit_model(r, 500, 600, TINY, seed=0)
    from stock_lstm.features import make_windows
    X, _, _ = make_windows(r, 10, 620, 630)
    np.testing.assert_allclose(m.predict_windows(X[..., 0]), m.predict_returns(r, 620, 630), atol=1e-6)
    with pytest.raises(ValueError, match="exactly 10"):
        m.predict_windows(X[..., 0][:, :8])          # 8 columns instead of 10, the old 98-vs-100 slip
