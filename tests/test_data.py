import numpy as np
import pandas as pd
import pytest

from stock_lstm import data
from tests.conftest import FakeSession, make_tiingo_rows

KEY = "test-key-not-real"


def test_load_prices_returns_continuous_adjusted_close(tmp_path, session):
    prices = data.load_prices("AAPL", "2020-07-20", "2020-12-31", cache_dir=tmp_path,
                              api_key=KEY, session=session)
    assert prices.index.is_monotonic_increasing
    # A 4:1 split must not show up as a crash: biggest daily move stays small.
    assert np.log(prices).diff().abs().max() < 0.1


def test_raw_close_is_rejected_as_unadjusted_split():
    """The original notebook modelled raw `close`; this is the guard that catches it."""
    rows = make_tiingo_rows()
    raw = pd.Series([r["close"] for r in rows],
                    index=pd.to_datetime([r["date"] for r in rows]).tz_localize(None))
    with pytest.raises(ValueError, match="unadjusted stock split"):
        data.validate_prices(raw)


def test_second_load_uses_cache_and_needs_no_network_or_key(tmp_path, session):
    first = data.load_prices("AAPL", "2020-07-20", "2020-12-31", cache_dir=tmp_path,
                             api_key=KEY, session=session)
    assert len(session.calls) == 1
    again = data.load_prices("AAPL", "2020-07-20", "2020-12-31", cache_dir=tmp_path,
                             session=FakeSession(status_code=500))  # would blow up if used
    pd.testing.assert_series_equal(first, again, check_freq=False)
    assert (tmp_path / "AAPL_2020-07-20_2020-12-31.csv").exists()


def test_refresh_forces_a_new_download(tmp_path, session):
    kw = dict(cache_dir=tmp_path, api_key=KEY, session=session)
    data.load_prices("AAPL", "2020-07-20", "2020-12-31", **kw)
    data.load_prices("AAPL", "2020-07-20", "2020-12-31", refresh=True, **kw)
    assert len(session.calls) == 2


def test_api_key_goes_in_header_not_url(tmp_path, session):
    data.load_prices("AAPL", "2020-07-20", "2020-12-31", cache_dir=tmp_path,
                     api_key=KEY, session=session)
    call = session.calls[0]
    assert call["headers"]["Authorization"] == f"Token {KEY}"
    assert KEY not in call["url"] and KEY not in str(call["params"])


def test_missing_key_gives_actionable_error():
    with pytest.raises(RuntimeError, match="TIINGO_API_KEY"):
        data.get_api_key()


def test_key_is_read_from_environment(monkeypatch):
    monkeypatch.setenv("TIINGO_API_KEY", "from-env")
    assert data.get_api_key() == "from-env"


@pytest.mark.parametrize("status,exc", [(401, PermissionError), (403, PermissionError), (404, ValueError)])
def test_http_errors_are_translated(status, exc):
    with pytest.raises(exc):
        data.fetch_tiingo("AAPL", "2020-01-01", "2020-02-01", api_key=KEY,
                          session=FakeSession(status_code=status))


def test_empty_response_raises():
    with pytest.raises(ValueError, match="no rows"):
        data.fetch_tiingo("AAPL", "2020-01-01", "2020-02-01", api_key=KEY,
                          session=FakeSession(rows=[]))


def _series(values, dates=None):
    idx = pd.DatetimeIndex(dates if dates is not None else pd.bdate_range("2024-01-01", periods=len(values)))
    return pd.Series(values, index=idx, dtype=float)


def test_validate_rejects_nan_nonpositive_and_duplicate_dates():
    with pytest.raises(ValueError, match="missing"):
        data.validate_prices(_series([1.0, np.nan, 1.1]))
    with pytest.raises(ValueError, match="non-positive"):
        data.validate_prices(_series([1.0, 0.0, 1.1]))
    dup = _series([1.0, 1.0, 1.1], dates=["2024-01-02", "2024-01-02", "2024-01-03"])
    with pytest.raises(ValueError, match="increasing"):
        data.validate_prices(dup)


def test_validate_accepts_a_realistic_volatile_series():
    data.validate_prices(_series([100, 88, 95, 80, 84]))  # -20% day is fine, not a split


def test_adjustment_events_finds_the_split():
    frame = pd.DataFrame(make_tiingo_rows())
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
    events = data.adjustment_events(frame)
    assert len(events) == 1 and events.iloc[0]["splitFactor"] == 4.0
