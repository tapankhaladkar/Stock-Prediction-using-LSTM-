"""Load split- and dividend-adjusted daily prices from Tiingo, cached on disk.

Why adjusted prices: Tiingo's ``close`` is the price as traded on the day. AAPL did a
4:1 split on 2020-08-31, so the raw series falls from ~$500 to ~$130 overnight and a
model sees a crash that never happened. ``adjClose`` is rewritten to be continuous
across splits and dividends, which is what a price-history model needs.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

TIINGO_URL = "https://api.tiingo.com/tiingo/daily/{ticker}/prices"
API_KEY_VAR = "TIINGO_API_KEY"

# A single-day move beyond this (in log terms, ~-33% / +49%) has never happened to AAPL
# outside of a stock split; the smallest common split (3:2) is log(1.5) = 0.405.
MAX_ABS_DAILY_LOG_RETURN = 0.4


def _read_dotenv(path: str | Path = ".env") -> dict[str, str]:
    """Minimal ``.env`` reader: ``NAME=value`` lines, optional ``export``, quotes and comments.

    Kept dependency-free on purpose. Values are returned, never logged or copied into
    ``os.environ``.
    """
    values: dict[str, str] = {}
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return values
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        name, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].strip()      # trailing comment on an unquoted value
        values[name.strip()] = value
    return values


def get_api_key(api_key: str | None = None) -> str:
    """Return the Tiingo key.

    Lookup order: the ``api_key`` argument, the ``TIINGO_API_KEY`` environment variable, a
    ``TIINGO_API_KEY=...`` line in ``.env`` in the current directory (git-ignored), then a
    Colab secret. An empty value counts as not set.
    """
    key = api_key or os.environ.get(API_KEY_VAR) or _read_dotenv().get(API_KEY_VAR)
    if not key:
        try:  # Colab: Secrets panel (key icon in the left sidebar)
            from google.colab import userdata  # type: ignore

            key = userdata.get(API_KEY_VAR)
        except Exception:
            key = None
    if not key:
        raise RuntimeError(
            f"No Tiingo API key found. Set the {API_KEY_VAR} environment variable, put "
            f"{API_KEY_VAR}=... in a .env file (see .env.example), or add it as a Colab secret. "
            "Free key: https://www.tiingo.com/account/api/token"
        )
    return key


def fetch_tiingo(ticker: str, start: str, end: str, api_key: str | None = None,
                 session: requests.Session | None = None) -> pd.DataFrame:
    """Download daily bars for ``ticker`` between ``start`` and ``end`` (inclusive).

    The key is sent as a header, never in the URL, so it cannot leak through logs or
    exception messages.
    """
    http = session or requests
    resp = http.get(
        TIINGO_URL.format(ticker=ticker.lower()),
        params={"startDate": start, "endDate": end, "format": "json", "resampleFreq": "daily"},
        headers={"Authorization": f"Token {get_api_key(api_key)}"},
        timeout=30,
    )
    if resp.status_code in (401, 403):
        raise PermissionError("Tiingo rejected the API key (HTTP %d). Check or rotate the key." % resp.status_code)
    if resp.status_code == 404:
        raise ValueError(f"Tiingo does not know ticker {ticker!r} (HTTP 404).")
    resp.raise_for_status()
    rows = resp.json()
    if not rows:
        raise ValueError(f"Tiingo returned no rows for {ticker} between {start} and {end}.")
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"], utc=True).dt.tz_localize(None).dt.normalize()
    return frame.sort_values("date").reset_index(drop=True)


def validate_prices(prices: pd.Series, max_abs_log_return: float = MAX_ABS_DAILY_LOG_RETURN) -> None:
    """Fail loudly on the data problems that silently wreck a price model."""
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("prices must be indexed by date")
    if not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
        raise ValueError("prices index must be strictly increasing (no duplicate dates)")
    if prices.isna().any():
        raise ValueError(f"{int(prices.isna().sum())} missing prices")
    if (prices <= 0).any():
        raise ValueError("non-positive prices")
    jumps = np.log(prices).diff().abs()
    bad = jumps[jumps > max_abs_log_return]
    if not bad.empty:
        when = ", ".join(d.strftime("%Y-%m-%d") for d in bad.index[:5])
        raise ValueError(
            f"{len(bad)} one-day move(s) larger than {max_abs_log_return:.2f} in log terms "
            f"(first: {when}). This looks like an unadjusted stock split; use adjClose."
        )


def cache_path(ticker: str, start: str, end: str, cache_dir: str | Path = "data") -> Path:
    return Path(cache_dir) / f"{ticker.upper()}_{start}_{end}.csv"


def load_prices(ticker: str, start: str, end: str, cache_dir: str | Path = "data",
                refresh: bool = False, api_key: str | None = None,
                session: requests.Session | None = None) -> pd.Series:
    """Adjusted close for ``ticker`` over a fixed ``[start, end]`` range.

    The raw Tiingo response is cached as ``{cache_dir}/{ticker}_{start}_{end}.csv`` so
    later runs need neither network nor key. Note: adjusted history is rewritten by
    Tiingo whenever a new dividend or split is paid, so re-downloading later can change
    old values slightly; keep the cached CSV if you need exact reproducibility.
    """
    path = cache_path(ticker, start, end, cache_dir)
    if path.exists() and not refresh:
        frame = pd.read_csv(path, parse_dates=["date"])
    else:
        frame = fetch_tiingo(ticker, start, end, api_key=api_key, session=session)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    prices = frame.set_index("date")["adjClose"].astype(float).rename(f"{ticker.upper()} adjClose")
    validate_prices(prices)
    return prices


def adjustment_events(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows of a raw Tiingo frame where a split happened (``splitFactor != 1``)."""
    return frame.loc[frame["splitFactor"] != 1.0, ["date", "close", "adjClose", "splitFactor"]]
