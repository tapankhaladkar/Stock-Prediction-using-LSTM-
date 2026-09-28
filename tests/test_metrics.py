import numpy as np
import pytest

from stock_lstm import metrics


def test_perfect_predictions_score_zero_error():
    y = np.array([100.0, 101.5, 99.0, 103.0])
    assert metrics.rmse(y, y) == 0 and metrics.mae(y, y) == 0 and metrics.mape(y, y) == 0


def test_known_values():
    y, p = np.array([100.0, 200.0]), np.array([110.0, 190.0])
    assert metrics.rmse(y, p) == pytest.approx(10.0)
    assert metrics.mae(y, p) == pytest.approx(10.0)
    assert metrics.mape(y, p) == pytest.approx((10 / 100 + 10 / 200) / 2 * 100)


def test_the_original_scale_mismatch_bug_is_what_made_rmse_look_like_price():
    """Old notebook: y was scaled to [0,1] but predictions were converted to dollars.
    Even a perfect model then scored an 'RMSE' near the average price (hence 213 on ~$211)."""
    dollars = np.linspace(170, 260, 300)
    scaled = (dollars - dollars.min()) / (dollars.max() - dollars.min())
    buggy = metrics.rmse(scaled, dollars)      # what the notebook computed
    correct = metrics.rmse(dollars, dollars)   # same (perfect) model, consistent units
    assert correct == 0
    assert buggy == pytest.approx(dollars.mean(), rel=0.1)


def test_shape_mismatch_is_rejected_not_broadcast():
    with pytest.raises(ValueError):
        metrics.rmse(np.zeros(5), np.zeros((5, 1)))
    with pytest.raises(ValueError):
        metrics.rmse(np.zeros(5), np.zeros(4))


def test_nan_is_rejected():
    with pytest.raises(ValueError, match="NaN"):
        metrics.rmse([1.0, 2.0], [1.0, np.nan])


def test_mape_rejects_non_positive_actuals():
    with pytest.raises(ValueError):
        metrics.mape([0.0, 1.0], [1.0, 1.0])


def test_directional_accuracy():
    prev = np.array([100.0, 100.0, 100.0, 100.0])
    actual = np.array([101.0, 99.0, 102.0, 98.0])
    right = np.array([100.5, 99.5, 101.0, 97.0])     # 4/4 right
    half = np.array([100.5, 100.5, 99.0, 97.0])      # right, wrong, wrong, right
    assert metrics.directional_accuracy(actual, right, prev) == 100.0
    assert metrics.directional_accuracy(actual, half, prev) == 50.0


def test_directional_accuracy_is_nan_for_persistence_not_zero():
    prev = np.array([100.0, 101.0, 99.0])
    actual = np.array([101.0, 99.0, 100.0])
    assert np.isnan(metrics.directional_accuracy(actual, prev.copy(), prev))


def test_up_day_rate_skips_flat_days():
    prev = np.array([100.0, 100.0, 100.0, 100.0])
    actual = np.array([101.0, 101.0, 99.0, 100.0])   # up, up, down, flat
    assert metrics.up_day_rate(actual, prev) == pytest.approx(200 / 3)


def test_bootstrap_ratio_is_one_for_identical_models():
    rng = np.random.default_rng(1)
    a = 100 + rng.normal(0, 1, 300)
    p = a + rng.normal(0, 1, 300)
    lo, hi = metrics.bootstrap_rmse_ratio(a, p, p.copy())
    assert lo == pytest.approx(1.0) and hi == pytest.approx(1.0)


def test_bootstrap_ratio_detects_a_clearly_better_model_and_a_clearly_worse_one():
    rng = np.random.default_rng(2)
    a = 100 + rng.normal(0, 1, 500)
    good, bad = a + rng.normal(0, 0.3, 500), a + rng.normal(0, 1.0, 500)
    lo, hi = metrics.bootstrap_rmse_ratio(a, good, bad)
    assert hi < 1.0
    lo, hi = metrics.bootstrap_rmse_ratio(a, bad, good)
    assert lo > 1.0


def test_bootstrap_ratio_interval_is_calibrated_for_equal_skill_models():
    """Two equally skilled models: the 95% interval should contain 1.0 about 95% of the time.
    (A single-dataset 'must straddle 1' check would fail ~5% of the time by construction.)"""
    hits, trials = 0, 150
    for s in range(trials):
        r = np.random.default_rng(1000 + s)
        a = 100 + r.normal(0, 1, 500)
        m1, m2 = a + r.normal(0, 1, 500), a + r.normal(0, 1, 500)
        lo, hi = metrics.bootstrap_rmse_ratio(a, m1, m2, n_boot=300, seed=s)
        hits += lo < 1.0 < hi
    assert 0.85 <= hits / trials <= 1.0
