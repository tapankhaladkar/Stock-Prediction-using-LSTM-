import numpy as np
import pandas as pd
import pytest

from stock_lstm.features import Scaler, log_returns, make_windows


def test_log_returns_reconstruct_prices_and_drop_only_the_first_day():
    idx = pd.bdate_range("2024-01-01", periods=6)
    px = pd.Series([100.0, 102, 101, 105, 104, 108], index=idx)
    r = log_returns(px)
    assert list(r.index) == list(idx[1:])
    np.testing.assert_allclose(px.iloc[0] * np.exp(r.cumsum()), px.iloc[1:])


def test_scaler_roundtrip_and_uses_only_the_slice_it_was_fit_on():
    x = np.array([1.0, 2.0, 3.0, 4.0, 1000.0])
    s = Scaler.fit(x[:4])
    assert s.mean == pytest.approx(2.5)                 # the 1000 outlier never influenced it
    np.testing.assert_allclose(s.inverse(s.transform(x)), x)


def test_scaler_survives_constant_input_and_rejects_tiny_input():
    z = Scaler.fit(np.ones(10)).transform(np.ones(10))
    assert np.isfinite(z).all()
    with pytest.raises(ValueError):
        Scaler.fit([1.0])


def test_windows_are_aligned_and_never_contain_their_own_target():
    x = np.arange(100, dtype=float)                     # value == position, so leaks are obvious
    X, y, t = make_windows(x, window=5, start=20, stop=30)
    assert X.shape == (10, 5, 1) and y.shape == (10,)
    np.testing.assert_array_equal(t, np.arange(20, 30))
    for row, target, pos in zip(X[..., 0], y, t):
        np.testing.assert_array_equal(row, np.arange(pos - 5, pos))   # exactly the 5 days before t
        assert target == pos and row.max() < target                   # target is strictly later


def test_windows_may_reach_back_before_start_but_start_must_leave_room():
    x = np.arange(50, dtype=float)
    X, _, _ = make_windows(x, window=5, start=5, stop=8)
    assert X[0, :, 0].tolist() == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError, match="complete"):
        make_windows(x, window=5, start=4, stop=8)


@pytest.mark.parametrize("start,stop", [(10, 10), (10, 9), (10, 51)])
def test_window_bounds_are_validated(start, stop):
    with pytest.raises(ValueError):
        make_windows(np.arange(50, dtype=float), window=5, start=start, stop=stop)
