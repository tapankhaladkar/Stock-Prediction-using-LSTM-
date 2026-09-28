import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # no display needed for plot smoke tests


def make_tiingo_rows(n=120, split_at=30, factor=4.0, seed=0, start="2020-07-20"):
    """Tiingo-shaped rows for a stock with one ``factor``:1 split at row ``split_at``.

    ``close`` is the traded price (cliff at the split); ``adjClose`` is continuous.
    """
    rng = np.random.default_rng(seed)
    adj = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n)))
    raw = adj.copy()
    raw[:split_at] *= factor  # pre-split shares traded at factor x the adjusted price
    dates = pd.bdate_range(start, periods=n)
    rows = []
    for i, d in enumerate(dates):
        rows.append({
            "date": d.strftime("%Y-%m-%dT00:00:00.000Z"),
            "close": float(raw[i]), "adjClose": float(adj[i]),
            "volume": 1_000_000, "adjVolume": 1_000_000,
            "divCash": 0.0, "splitFactor": factor if i == split_at else 1.0,
        })
    return rows


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else []

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    """Stands in for ``requests``; records every call so tests can inspect them."""

    def __init__(self, rows=None, status_code=200):
        self.rows = rows if rows is not None else make_tiingo_rows()
        self.status_code = status_code
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": params, "headers": headers})
        return FakeResponse(self.status_code, self.rows)


@pytest.fixture
def session():
    return FakeSession()


@pytest.fixture(autouse=True)
def _no_ambient_key(monkeypatch, tmp_path):
    """No test may see a real key: drop the env var and run from an empty directory so a
    developer's own .env (git-ignored, in the repo root) is not picked up either."""
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)


def write_synthetic_tiingo_cache(path, start="2015-01-01", end="2025-09-30", split_date="2020-08-31", seed=42):
    """A Tiingo-shaped CSV of *fake* prices with a real 4:1 split, for exercising the notebook
    offline. Never a substitute for real data."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    adj = 25 * np.exp(np.cumsum(rng.normal(0.0006, 0.014, len(dates))))
    split_i = int(np.searchsorted(dates, pd.Timestamp(split_date)))
    raw = adj.copy()
    raw[:split_i] *= 4.0
    df = pd.DataFrame({"date": dates, "close": raw, "high": raw * 1.01, "low": raw * 0.99, "open": raw,
                       "volume": 1_000_000, "adjClose": adj, "adjHigh": adj * 1.01, "adjLow": adj * 0.99,
                       "adjOpen": adj, "adjVolume": 4_000_000, "divCash": 0.0, "splitFactor": 1.0})
    df.loc[split_i, "splitFactor"] = 4.0
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
