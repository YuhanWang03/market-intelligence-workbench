from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.market_analysis import build_technical_analysis, enrich_bars
from app.routers import dashboard, portfolio, workspace
from v2.bot import state as bot_state


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(bot_state, "_DB_PATH", tmp_path / "bot_state.db")
    portfolio._PRICE_CACHE.clear()
    return TestClient(app)


def test_watchlist_and_price_alert_crud(client: TestClient):
    added = client.post("/api/watchlist", json={"ticker": "nvda", "note": "core"})
    assert added.status_code == 200
    assert added.json()["items"][0]["ticker"] == "NVDA"

    alert = client.post(
        "/api/price-alerts",
        json={"ticker": "nvda", "direction": "above", "target_price": 200},
    )
    assert alert.status_code == 200
    alert_id = alert.json()["id"]
    assert alert.json()["items"][0]["target_price"] == 200

    assert client.delete("/api/watchlist/NVDA").json()["items"] == []
    assert client.delete(f"/api/price-alerts/{alert_id}").json()["items"] == []


def test_monitoring_universe_matches_production_streamer(client: TestClient):
    from v2.screening.universe import TECH_30

    response = client.get("/api/monitoring/universe")
    assert response.status_code == 200
    assert response.json() == {
        "intraday": TECH_30,
        "source": "TECH_30",
        "scan_interval_seconds": 60,
        "price_pct_threshold": 0.03,
        "volume_pace_threshold": 2.5,
    }

    added = client.post("/api/monitoring/universe", json={"ticker": "XYZ"})
    assert added.status_code == 200
    assert added.json()["added"] is True
    assert added.json()["intraday"][-1] == "XYZ"
    assert added.json()["source"] == "TECH_30 + 自定义"

    removed = client.delete("/api/monitoring/universe/AAPL")
    assert removed.status_code == 200
    assert removed.json()["removed"] is True
    assert "AAPL" not in removed.json()["intraday"]

    restored = client.post("/api/monitoring/universe", json={"ticker": "AAPL"})
    assert restored.status_code == 200
    assert restored.json()["added"] is True
    assert "AAPL" in restored.json()["intraday"]

    assert client.delete("/api/monitoring/universe/XYZ").json()["removed"] is True


def test_monitoring_universe_rejects_invalid_ticker(client: TestClient):
    response = client.post("/api/monitoring/universe", json={"ticker": "123"})
    assert response.status_code == 400


def test_activity_reads_archive(client: TestClient, tmp_path, monkeypatch):
    archive_path = tmp_path / "archive.db"
    with sqlite3.connect(archive_path) as conn:
        conn.execute(
            """CREATE TABLE pushes (
                id INTEGER PRIMARY KEY, ts TEXT, agent TEXT, msg_type TEXT,
                tickers TEXT, text_html TEXT, title TEXT,
                priority_tier TEXT, importance_score REAL
            )"""
        )
        conn.execute(
            "INSERT INTO pushes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                1,
                datetime.now(timezone.utc).isoformat(),
                "monitor",
                "intraday_anomaly",
                "NVDA",
                "<b>Volume spike</b>",
                "NVDA anomaly",
                "P1",
                82,
            ),
        )
        conn.execute(
            "INSERT INTO pushes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                2,
                datetime.now(timezone.utc).isoformat(),
                "sec",
                "text",
                "AAPL",
                "<b>Form 4 digest</b>",
                "SEC update",
                "P2",
                40,
            ),
        )
    monkeypatch.setattr(
        workspace,
        "SETTINGS",
        SimpleNamespace(archive_db_path=archive_path),
    )

    response = client.get("/api/activity")
    assert response.status_code == 200
    assert len(response.json()["items"]) == 2

    realtime = client.get("/api/activity?realtime_only=true")
    assert realtime.status_code == 200
    assert [item["title"] for item in realtime.json()["items"]] == ["NVDA anomaly"]


def test_lab_signals_and_run_log(client: TestClient, monkeypatch):
    signals = client.get("/api/lab/signals")
    assert signals.status_code == 200
    assert signals.json()["monitoring"]["volume_spike_threshold"] == 3.0

    monkeypatch.setattr(
        workspace,
        "_run_backtest",
        lambda body: {
            "kind": "backtest",
            "tickers": body.tickers,
            "metrics": {"n_trades": 3, "sharpe": 1.2},
        },
    )
    result = client.post("/api/lab/backtest", json={"tickers": ["AAPL"]})
    assert result.status_code == 200
    assert result.json()["metrics"]["n_trades"] == 3
    runs = client.get("/api/lab/runs").json()["items"]
    assert runs[0]["kind"] == "backtest"
    assert runs[0]["tickers"] == ["AAPL"]


def test_ticker_validation():
    assert workspace._normalize_tickers([" nvda ", "NVDA", "BRK.B"]) == [
        "NVDA",
        "BRK.B",
    ]
    with pytest.raises(ValueError):
        workspace._normalize_tickers(["123"])


def test_ticker_tape_normalizes_non_finite_provider_values(monkeypatch):
    from v2.macro import fred_client, market_client

    monkeypatch.setattr(dashboard, "_TAPE_SYMBOLS", [
        ("sp500", "SPY", "标普500"),
        ("nasdaq100", "QQQ", "纳指100"),
    ])
    monkeypatch.setattr(dashboard, "_TAPE_FRED_SERIES", [
        ("us5y", "DGS5", "美债5Y"),
    ])
    quotes = {
        "SPY": {"value": float("nan"), "pct_change_1d": 0.01},
        "QQQ": {"value": 500, "pct_change_1d": float("inf")},
    }
    monkeypatch.setattr(market_client, "_safe_quote", lambda symbol: quotes.get(symbol))
    monkeypatch.setattr(fred_client, "get_latest_value", lambda _series: float("nan"))

    result = dashboard._fetch_tape()

    assert result == {
        "items": [
            {"key": "sp500", "label": "标普500", "value": None,
             "change_pct": 0.01, "unit": ""},
            {"key": "nasdaq100", "label": "纳指100", "value": 500.0,
             "change_pct": None, "unit": ""},
            {"key": "us5y", "label": "美债5Y", "value": None,
             "change_pct": None, "unit": "%"},
        ],
    }


def test_ticker_tape_catalog_includes_requested_markets():
    symbols = {key: symbol for key, symbol, _label in dashboard._TAPE_SYMBOLS}
    fred = {key: series for key, series, _label in dashboard._TAPE_FRED_SERIES}

    assert symbols["silver"] == "SI=F"
    assert symbols["brent"] == "BZ=F"
    assert symbols["philadelphia_semiconductor"] == "^SOX"
    assert fred["us5y"] == "DGS5"
    assert fred["us10y"] == "DGS10"


def test_position_price_history_returns_consistent_contract(client: TestClient, monkeypatch):
    payload = {
        "symbol": "BRK.B", "range": "3Y", "timeframe": "1w",
        "periodDays": 1096, "visibleStart": "2023-09-04",
        "visibleEnd": "2026-09-04", "warmupBars": 250,
        "source": "YAHOO_FINANCE",
        "adjustmentMode": "SPLIT_ADJUSTED",
        "includesExtendedHours": False,
        "quote": {
            "symbol": "BRK.B", "timestamp": "2026-09-04T16:00:00-04:00",
            "price": 507.5, "regularClose": 507.5, "previousClose": 502.25,
            "dailyChangePct": .01045, "extendedHoursChangePct": None,
            "source": "YAHOO_FINANCE", "session": "CLOSED", "isDelayed": True,
        },
        "bars": [], "technicalAnalysis": {},
    }
    monkeypatch.setattr(portfolio, "_fetch_price_history", lambda ticker, range_key, extended: payload)
    response = client.get("/api/price-history/BRK.B?range=3Y")

    assert response.status_code == 200
    assert response.json() == payload


@pytest.mark.parametrize(('symbol', 'yahoo_symbol'), [
    ('BRK.B', 'BRK-B'), ('BF.B', 'BF-B'), ('AAPL', 'AAPL'), ('BRK-B', 'BRK-B'),
])
@pytest.mark.parametrize('range_key', ['1M', '3M'])
def test_price_history_maps_yahoo_symbol_only(client, monkeypatch, symbol, yahoo_symbol, range_key):
    import pandas as pd
    import yfinance as yf
    from datetime import timedelta

    requested, histories = [], []
    yesterday = datetime.now(portfolio._ET).replace(hour=10, minute=0, second=0, microsecond=0) - timedelta(days=1)
    frame = pd.DataFrame(
        {'Open': [500., 501.], 'High': [504., 505.], 'Low': [499., 500.],
         'Close': [502., 503.], 'Adj Close': [502., 503.], 'Volume': [1000, 1200]},
        index=pd.DatetimeIndex([yesterday - timedelta(days=1), yesterday]),
    )

    def factory(value):
        requested.append(value)
        assert value == yahoo_symbol
        def history(**kwargs):
            histories.append(kwargs)
            return frame.copy()
        return SimpleNamespace(history=history)

    monkeypatch.setattr(yf, 'Ticker', factory)
    response = client.get(f'/api/price-history/{symbol}?range={range_key}')
    assert response.status_code == 200, response.text
    data = response.json()
    assert data['symbol'] == data['quote']['symbol'] == symbol
    assert data['bars'] and all(bar['symbol'] == symbol for bar in data['bars'])
    assert requested == [yahoo_symbol]
    assert len(histories) == 3  # Chart, minute quote and completed daily closes.
    assert (symbol, range_key, False) in portfolio._PRICE_CACHE
    assert client.get(f'/api/price-history/{symbol}?range={range_key}').status_code == 200
    assert requested == [yahoo_symbol]  # Cached requests retain the public symbol.


def test_position_price_history_rejects_invalid_ticker(client: TestClient):
    response = client.get("/api/price-history/not%20a%20ticker")
    assert response.status_code == 400


def test_position_price_history_supports_requested_ranges(client: TestClient):
    assert portfolio._PRICE_RANGES == {
        "1W": {"days": 7, "timeframe": "30m", "warmup_days": 45},
        "1M": {"days": 31, "timeframe": "1h", "warmup_days": 150},
        "3M": {"days": 93, "timeframe": "1d", "warmup_days": 400},
        "6M": {"days": 186, "timeframe": "1d", "warmup_days": 400},
        "1Y": {"days": 366, "timeframe": "1d", "warmup_days": 400},
        "3Y": {"days": 1096, "timeframe": "1w", "warmup_days": 1900},
    }
    response = client.get("/api/price-history/AAPL?range=5Y")
    assert response.status_code == 400


def test_position_price_history_uses_30_minute_bars_for_one_week(
    client: TestClient,
    monkeypatch,
):
    def fake_history(ticker, range_key, extended):
        assert (ticker, range_key, extended) == ("AAPL", "1W", False)
        return {
            "symbol": "AAPL", "range": "1W", "timeframe": "30m",
            "periodDays": 7, "visibleStart": "2026-08-28", "visibleEnd": "2026-09-04",
            "warmupBars": 220, "source": "YAHOO_FINANCE",
            "adjustmentMode": "SPLIT_ADJUSTED", "includesExtendedHours": False,
            "quote": {}, "technicalAnalysis": {},
            "bars": [{"timestamp": "2026-09-04T10:00:00-04:00", "close": 231.5,
                      "high": 232.0, "volume": 740000}],
        }

    monkeypatch.setattr(portfolio, "_fetch_price_history", fake_history)
    response = client.get("/api/price-history/AAPL?range=1W")

    assert response.status_code == 200
    assert response.json()["timeframe"] == "30m"
    assert response.json()["bars"][0]["close"] == 231.5
    assert response.json()["bars"][0]["high"] == 232.0
    assert response.json()["bars"][0]["volume"] == 740000


def _synthetic_bars(count: int = 280, timeframe: str = "1d") -> list[dict]:
    bars = []
    for index in range(count):
        center = 100 + index * .03 + ((index % 20) - 10) * .2
        bars.append({
            "symbol": "TEST", "timestamp": f"2025-01-{(index % 28) + 1:02d}T16:00:00-05:00",
            "open": center - .2, "high": center + 1.2, "low": center - 1.1,
            "close": center + .25, "volume": 1_000_000 + (index % 7) * 100_000,
            "timeframe": timeframe, "session": "REGULAR", "source": "TEST",
            "isFinal": True,
        })
    return bars


def test_indicator_warmup_keeps_sma_complete_at_visible_start():
    bars = enrich_bars(_synthetic_bars(), "1d")
    visible = bars[-20:]

    assert visible[0]["indicators"]["sma20"] is not None
    assert visible[0]["indicators"]["sma50"] is not None
    assert visible[0]["indicators"]["sma200"] is not None
    assert visible[0]["indicators"]["timeframe"] == "1d"
    assert visible[0]["indicators"]["volumeMA20"] is not None


def test_technical_analysis_levels_are_timeframe_bound_and_directional():
    bars = enrich_bars(_synthetic_bars(), "1d")
    quote = {
        "symbol": "TEST", "timestamp": bars[-1]["timestamp"], "price": 105.0,
        "regularClose": 105.0, "dailyChangePct": .01,
        "source": "TEST", "session": "CLOSED", "isDelayed": False,
    }
    analysis = build_technical_analysis(bars, "1d", quote, "SPLIT_ADJUSTED")

    assert analysis["timeframe"] == "1d"
    assert analysis["SMA200"] is not None
    assert analysis["volumeMA20"] is not None
    assert analysis["supports"] and analysis["resistances"]
    assert all(level["timeframe"] == "1d" for level in analysis["supports"] + analysis["resistances"])
    assert all(level["upper"] < analysis["currentPrice"] for level in analysis["supports"])
    assert all(level["lower"] > analysis["currentPrice"] for level in analysis["resistances"])
    assert [level["distancePct"] for level in analysis["supports"]] == sorted(level["distancePct"] for level in analysis["supports"])
    assert [level["distancePct"] for level in analysis["resistances"]] == sorted(level["distancePct"] for level in analysis["resistances"])
    all_levels = analysis["supports"] + analysis["resistances"]
    assert all(
        first["upper"] < second["lower"] or second["upper"] < first["lower"]
        for index, first in enumerate(all_levels)
        for second in all_levels[index + 1:]
    )


def test_hourly_last_regular_bar_closes_at_market_boundary():
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    start = datetime(2026, 9, 4, 15, 30, tzinfo=et)
    assert portfolio._bar_is_final(start, "1h", datetime(2026, 9, 4, 16, 0, tzinfo=et))
    assert not portfolio._bar_is_final(start, "1h", datetime(2026, 9, 4, 15, 59, tzinfo=et))


def test_after_hours_quote_finalizes_last_intraday_regular_bar():
    bars = [{
        "timestamp": "2026-09-04T15:30:00-04:00", "session": "REGULAR",
        "open": 1012.0, "high": 1015.2, "low": 1010.5, "close": 1014.95,
        "isFinal": True, "status": "FINAL", "isStale": False,
    }]
    quote = {
        "timestamp": "2026-09-04T17:20:00-04:00", "session": "AFTER_HOURS",
        "price": 1013.88, "regularClose": 1016.59, "regularCloseIsOfficial": True,
    }

    portfolio._synchronize_forming_bar(bars, quote, "1h")

    assert bars[0]["close"] == 1016.59
    assert bars[0]["high"] == 1016.59
    assert bars[0]["isFinal"] is True
    assert bars[0]["status"] == "FINAL"
    assert bars[0]["finalizationSource"] == "DAILY_REGULAR_CLOSE"


def test_missing_terminal_intraday_bar_is_marked_stale_not_final():
    bars = [{
        "timestamp": "2026-09-04T14:30:00-04:00", "session": "REGULAR",
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
        "isFinal": True, "status": "FINAL", "isStale": False,
    }]
    quote = {
        "timestamp": "2026-09-04T17:20:00-04:00", "session": "AFTER_HOURS",
        "price": 100.2, "regularClose": 101.5, "regularCloseIsOfficial": True,
    }

    portfolio._synchronize_forming_bar(bars, quote, "1h")

    assert bars[0]["close"] == 100.5
    assert bars[0]["isFinal"] is False
    assert bars[0]["status"] == "STALE"
    assert bars[0]["isStale"] is True
