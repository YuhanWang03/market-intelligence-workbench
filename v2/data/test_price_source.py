"""Smoke tests for v2/data/price_source.py — Phase 4.5-mini.

The wider ``v2/data`` directory is production-only (FDClient + Price
dataclass live on the VPS, not in this repo). For sandbox testing of
the price_source module we stub ``v2.data`` + ``v2.data.models`` via
sys.modules and load the file directly through importlib — same
pattern the Phase 3 SEC bot-responder tests use to bypass v2.data
without touching the rest of the import chain.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import types
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Sandbox stubs for v2.data + v2.data.models — production-only modules
# ---------------------------------------------------------------------------

@dataclass
class _Price:
    """Sandbox-only Price stub. Field shape mirrors production's
    v2.data.models.Price: time (ISO str) / open / high / low / close /
    volume."""
    time: str
    open: float
    high: float
    low: float
    close: float
    volume: int


def _ensure_data_stubs() -> None:
    """Install v2.data + v2.data.models in sys.modules so price_source.py
    resolves ``from v2.data.models import Price`` cleanly. Called once
    at module load — pytest fixtures install fresh CachedFDClient stubs
    per-test."""
    if "v2.data" not in sys.modules or not hasattr(
        sys.modules.get("v2.data"), "CachedFDClient",
    ):
        v2_data = types.ModuleType("v2.data")
        v2_data.__path__ = []                            # mark as package
        v2_data.CachedFDClient = type("CachedFDClient", (), {})
        v2_data.FDClient = type("FDClient", (), {})
        sys.modules["v2.data"] = v2_data

    if "v2.data.models" not in sys.modules:
        v2_data_models = types.ModuleType("v2.data.models")
        v2_data_models.Price = _Price
        sys.modules["v2.data.models"] = v2_data_models


_ensure_data_stubs()


# Load price_source.py after the stubs are in place. importlib gets us
# a real module object pointing at our file on disk.
def _load_price_source():
    spec = importlib.util.spec_from_file_location(
        "v2.data.price_source",
        _REPO_ROOT / "v2" / "data" / "price_source.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v2.data.price_source"] = mod
    spec.loader.exec_module(mod)
    return mod


price_source = _load_price_source()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _FakeYFTicker:
    """Mocks yfinance.Ticker; records calls + returns canned DataFrame."""

    def __init__(self, symbol: str, *, history_return=None, raise_exc=None):
        self.symbol = symbol
        self.history_return = history_return
        self.raise_exc = raise_exc
        self.calls: list[dict] = []

    def history(self, *, start, end, auto_adjust):
        self.calls.append({"start": start, "end": end,
                           "auto_adjust": auto_adjust})
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.history_return


def _make_df(rows: list[dict]):
    """Build a pandas DataFrame with a DatetimeIndex from row dicts."""
    import pandas as pd
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("date")
    df.index = pd.to_datetime(df.index)
    return df


# ---------------------------------------------------------------------------
# YFinancePriceSource tests
# ---------------------------------------------------------------------------

def test_yfinance_price_source_returns_real_data():
    """Mocked yfinance.Ticker.history returns a known DataFrame;
    verify Price dataclass output schema matches FD shape (Stage 0
    audit pinned: date / open / high / low / close / volume)."""
    df = _make_df([
        {"date": "2026-06-03", "Open": 100.0, "High": 102.0,
         "Low":  99.5, "Close": 101.5, "Volume": 1_200_000},
        {"date": "2026-06-04", "Open": 101.5, "High": 103.0,
         "Low": 101.0, "Close": 102.8, "Volume": 1_500_000},
    ])
    fake_ticker = _FakeYFTicker("NVDA", history_return=df)
    src = price_source.YFinancePriceSource(
        ticker_factory=lambda s: fake_ticker,
    )

    prices = src.get_prices("NVDA",
                            date(2026, 6, 3), date(2026, 6, 4))
    assert len(prices) == 2
    # Sorted ascending by date
    assert prices[0].time < prices[1].time
    # Schema check — all 6 fields populated
    p0 = prices[0]
    assert p0.time == "2026-06-03"
    assert p0.open == 100.0
    assert p0.high == 102.0
    assert p0.low == 99.5
    assert p0.close == 101.5
    assert p0.volume == 1_200_000
    # End-exclusive bump: yfinance got end=2026-06-05
    assert fake_ticker.calls[0]["end"] == "2026-06-05"
    assert fake_ticker.calls[0]["auto_adjust"] is False


def test_yfinance_accepts_iso_string_inputs():
    """start/end accept str OR date — same tolerance as FD."""
    df = _make_df([
        {"date": "2026-06-04", "Open": 100.0, "High": 101.0,
         "Low": 99.0, "Close": 100.5, "Volume": 1_000_000},
    ])
    fake_ticker = _FakeYFTicker("AAPL", history_return=df)
    src = price_source.YFinancePriceSource(
        ticker_factory=lambda s: fake_ticker,
    )
    out = src.get_prices("AAPL", "2026-06-04", "2026-06-04")
    assert len(out) == 1
    # End was bumped from 2026-06-04 → 2026-06-05 for inclusive semantics
    assert fake_ticker.calls[0]["end"] == "2026-06-05"


def test_yfinance_empty_data_returns_empty_list(caplog):
    """yfinance returns empty DataFrame → returns []; never raises.

    Existing callers all gate on `if not prices` (Stage 0 audit), so
    the empty list is the contract."""
    import pandas as pd
    fake_ticker = _FakeYFTicker("DELISTED", history_return=pd.DataFrame())
    src = price_source.YFinancePriceSource(
        ticker_factory=lambda s: fake_ticker,
    )
    with caplog.at_level(logging.WARNING):
        out = src.get_prices("DELISTED", date(2026, 6, 1), date(2026, 6, 4))
    assert out == []
    assert "returned empty" in caplog.text


def test_yfinance_failure_silent(caplog):
    """Network / rate-limit error → returns [] + WARNING. Never raises."""
    fake_ticker = _FakeYFTicker(
        "NVDA", raise_exc=RuntimeError("yfinance 429 rate limit"),
    )
    src = price_source.YFinancePriceSource(
        ticker_factory=lambda s: fake_ticker,
    )
    with caplog.at_level(logging.WARNING):
        out = src.get_prices("NVDA", date(2026, 6, 1), date(2026, 6, 4))
    assert out == []
    assert "yfinance get_prices" in caplog.text
    assert "rate limit" in caplog.text


def test_yfinance_row_decode_failure_is_partial(caplog):
    """One malformed row (NaN volume) doesn't break the whole series."""
    import math
    import pandas as pd
    df = pd.DataFrame([
        {"Open": 100.0, "High": 102.0, "Low": 99.5,
         "Close": 101.5, "Volume": 1_200_000},
        {"Open": 101.5, "High": 103.0, "Low": 101.0,
         "Close": 102.8, "Volume": math.nan},      # weird but possible
    ], index=pd.to_datetime(["2026-06-03", "2026-06-04"]))
    fake_ticker = _FakeYFTicker("XYZ", history_return=df)
    src = price_source.YFinancePriceSource(
        ticker_factory=lambda s: fake_ticker,
    )
    prices = src.get_prices("XYZ", date(2026, 6, 3), date(2026, 6, 4))
    assert len(prices) == 2
    assert prices[0].volume == 1_200_000
    # NaN volume falls back to 0 (still an int per schema)
    assert prices[1].volume == 0
    assert isinstance(prices[1].volume, int)


# ---------------------------------------------------------------------------
# FDPriceSource tests
# ---------------------------------------------------------------------------

def test_fd_price_source_delegates_to_client():
    """FDPriceSource.get_prices(...) forwards verbatim to fd_client.get_prices."""
    calls: list[tuple] = []

    class _FakeFD:
        def get_prices(self, ticker, start, end):
            calls.append((ticker, start, end))
            return [_Price(time="2026-06-04",
                           open=100.0, high=101.0, low=99.0,
                           close=100.5, volume=1_000_000)]

    src = price_source.FDPriceSource(_FakeFD())
    out = src.get_prices("AAPL", "2026-06-01", "2026-06-04")
    assert calls == [("AAPL", "2026-06-01", "2026-06-04")]
    assert len(out) == 1
    assert out[0].time == "2026-06-04"


# ---------------------------------------------------------------------------
# default_price_source() factory tests
# ---------------------------------------------------------------------------

@pytest.fixture
def _clean_env(monkeypatch):
    """Strip V2_PRICE_SOURCE so tests don't leak between cases."""
    monkeypatch.delenv("V2_PRICE_SOURCE", raising=False)
    yield


def test_default_factory_returns_yfinance_default(_clean_env):
    src = price_source.default_price_source()
    assert isinstance(src, price_source.YFinancePriceSource)


def test_default_factory_returns_fd_with_env(_clean_env, monkeypatch):
    """V2_PRICE_SOURCE=fd → FDPriceSource (ops escape hatch)."""
    monkeypatch.setenv("V2_PRICE_SOURCE", "fd")
    src = price_source.default_price_source()
    assert isinstance(src, price_source.FDPriceSource)


def test_default_factory_case_insensitive_env(_clean_env, monkeypatch):
    """Env var matching is case-insensitive + whitespace-tolerant —
    ops shouldn't get burned by 'FD' vs 'fd' or 'fd ' trailing space."""
    for val in ("FD", "fd", " fd ", "Fd"):
        monkeypatch.setenv("V2_PRICE_SOURCE", val)
        src = price_source.default_price_source()
        assert isinstance(src, price_source.FDPriceSource), (
            f"V2_PRICE_SOURCE={val!r} should route to FDPriceSource"
        )


def test_default_factory_unknown_env_returns_yfinance(_clean_env, monkeypatch):
    """Unknown value (e.g. 'yfinance' / 'auto') → default yfinance.
    Only the literal 'fd' (case-insensitive) flips the path."""
    monkeypatch.setenv("V2_PRICE_SOURCE", "yfinance")
    src = price_source.default_price_source()
    assert isinstance(src, price_source.YFinancePriceSource)


# ---------------------------------------------------------------------------
# Protocol surface contract
# ---------------------------------------------------------------------------

def test_both_implementations_satisfy_protocol():
    """Both classes have a `get_prices(ticker, start, end) -> list[Price]`
    method matching the Protocol contract. This is a runtime check —
    Protocol is structural — to ensure callers can swap implementations
    without static-type changes."""
    assert hasattr(price_source.YFinancePriceSource, "get_prices")
    assert hasattr(price_source.FDPriceSource, "get_prices")


def test_module_exports_public_names():
    """Public surface: PriceSource / FDPriceSource / YFinancePriceSource
    / default_price_source. Matches the Stage 1 spec."""
    for name in (
        "PriceSource", "FDPriceSource", "YFinancePriceSource",
        "default_price_source",
    ):
        assert hasattr(price_source, name), f"missing {name}"
        assert name in price_source.__all__, f"{name} not in __all__"


def test_yfinance_skips_placeholder_rows_without_a_close():
    """A NaN OHLC row (a session not yet printed, or a re-indexed day) is not a bar."""
    df = _make_df([
        {"date": "2026-09-08", "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5, "Volume": 1_000_000},
        {"date": "2026-09-09", "Open": float("nan"), "High": float("nan"), "Low": float("nan"), "Close": float("nan"), "Volume": 79_204_871},
    ])
    src = price_source.YFinancePriceSource(ticker_factory=lambda s: _FakeYFTicker("NVDA", history_return=df))
    prices = src.get_prices("NVDA", date(2026, 9, 1), date(2026, 9, 9))
    assert [p.time for p in prices] == ["2026-09-08"]


def test_usable_bars_drops_rows_without_a_finite_positive_close():
    """The guard any consumer can apply to bars from any source; needs no pandas."""
    from types import SimpleNamespace
    bars = [SimpleNamespace(time="a", close=10.0), SimpleNamespace(time="b", close=float("nan")), SimpleNamespace(time="c", close=0.0), SimpleNamespace(time="d", close=None)]
    assert [b.time for b in price_source.usable_bars(bars)] == ["a"] and price_source.usable_bars(None) == []
