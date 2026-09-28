import numpy as np
import pandas as pd
import pytest

from stock_lstm import baselines, evaluate, metrics


@pytest.fixture
def prices():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2022-01-03", periods=400)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, 400))), index=idx, name="px")


def test_persistence_is_yesterdays_close(prices):
    p = baselines.persistence(prices)
    assert np.isnan(p.iloc[0])
    pd.testing.assert_series_equal(p.iloc[1:], prices.shift(1).iloc[1:], check_names=False)


def test_moving_average_uses_only_prior_days(prices):
    ma = baselines.moving_average(prices, 5)
    t = 50
    assert ma.iloc[t] == pytest.approx(prices.iloc[t - 5:t].mean())
    assert ma.iloc[:5].isna().all() and ma.iloc[5:].notna().all()


@pytest.mark.parametrize("make", [
    lambda s: baselines.persistence(s),
    lambda s: baselines.moving_average(s, 10),
    lambda s: baselines.arima_one_step(s, n_train=250),
])
def test_baselines_are_causal_changing_the_future_never_changes_the_past(prices, make):
    """The prediction for day t must not depend on the close at t or later.

    Tampering with prices[t0:] must therefore leave predictions through t0 *inclusive*
    unchanged (the prediction for t0 may only use closes before t0)."""
    t0 = 300
    tampered = prices.copy()
    tampered.iloc[t0:] *= 1.5
    before, after = make(prices), make(tampered)
    pd.testing.assert_series_equal(before.iloc[:t0 + 1], after.iloc[:t0 + 1])


def test_arima_only_predicts_out_of_sample(prices):
    out = baselines.arima_one_step(prices, n_train=250)
    assert out.iloc[:250].isna().all() and out.iloc[250:].notna().all()
    # On a random walk ARIMA should be roughly as good as persistence, not wildly different.
    days = prices.index[250:]
    r = metrics.rmse(prices.loc[days], out.loc[days]) / metrics.rmse(prices.loc[days], prices.shift(1).loc[days])
    assert 0.9 < r < 1.1


def test_arima_rejects_bad_split(prices):
    with pytest.raises(ValueError):
        baselines.arima_one_step(prices, n_train=len(prices))


def test_compare_scores_models_on_common_days_and_ranks_vs_persistence(prices):
    preds = {
        "Persistence": baselines.persistence(prices),
        "MA(5)": baselines.moving_average(prices, 5),
        "Oracle": prices.copy(),  # cheats on purpose: sees today's close
    }
    table = evaluate.compare(prices, preds)
    assert table.attrs["n_days"] == len(prices) - 5   # limited by the MA warm-up
    assert table.loc["Oracle", "RMSE"] == 0
    assert table.loc["Persistence", "RMSE_vs_Persistence"] == 1.0
    assert table.loc["MA(5)", "RMSE_vs_Persistence"] > 1.0        # smoothing lags a random walk
    assert np.isnan(table.loc["Persistence", "DirAcc_%"])
    assert 0 < table.attrs["up_day_rate_%"] < 100


def test_compare_needs_overlap():
    idx = pd.bdate_range("2024-01-01", periods=5)
    s = pd.Series([1.0, 2, 3, 4, 5], index=idx)
    with pytest.raises(ValueError, match="no common days"):
        evaluate.compare(s, {"a": pd.Series([np.nan] * 5, index=idx)})
