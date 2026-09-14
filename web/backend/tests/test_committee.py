"""Lab · 投资人委员会 endpoint tests. No network, no key, no LLM."""

from __future__ import annotations

import time
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routers import committee, workspace
from v2.personas.fixtures import distressed_snapshot, quality_snapshot
from v2.personas.store import PersonaStore


class _FakeClient:
    """Serves the synthetic snapshots through the persona data protocol."""

    def __init__(self):
        self.snaps = {s.ticker: s for s in (quality_snapshot(), distressed_snapshot())}
        self.calls = 0

    def _snap(self, ticker):
        return self.snaps.get(ticker.upper())

    def get_financial_metrics(self, ticker, end_date, *, period="ttm", limit=10):
        self.calls += 1
        s = self._snap(ticker)
        return [r.to_dict() for r in s.metrics(period, limit)] if s else []

    def search_line_items(self, ticker, line_items, end_date, *, period="ttm", limit=10):
        self.calls += 1
        s = self._snap(ticker)
        return [r.to_dict() for r in s.line_items(period, limit)] if s else []

    def get_market_cap(self, ticker, end_date):
        self.calls += 1
        s = self._snap(ticker)
        return s.market_cap if s else None

    def get_insider_trades(self, ticker, end_date, *, start_date=None, limit=1000):
        self.calls += 1
        s = self._snap(ticker)
        return list(s.insider_trades) if s else []

    def get_company_news(self, ticker, end_date, *, start_date=None, limit=100):
        self.calls += 1
        s = self._snap(ticker)
        return list(s.news) if s else []

    def get_prices(self, ticker, start_date, end_date):
        self.calls += 1
        s = self._snap(ticker)
        return list(s.prices) if s else []


@pytest.fixture()
def fake():
    return _FakeClient()


@pytest.fixture()
def client(tmp_path, monkeypatch, fake):

    @contextmanager
    def _fake_data_client():
        yield fake

    monkeypatch.setattr(committee, "_data_client", _fake_data_client)
    return TestClient(app)


def test_committee_on_explicit_tickers_returns_matrix_and_persists(client, fake):
    body = {"source": "tickers", "tickers": ["qlty", "DSTR", "qlty"], "as_of": "2026-06-30", "personas": ["warren_buffett", "ben_graham", "michael_burry"]}
    res = client.post("/api/lab/committee", json=body)
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["kind"] == "committee" and data["source"] == "tickers"
    assert [v["ticker"] for v in data["verdicts"]] == ["QLTY", "DSTR"]
    assert data["verdicts"][1]["stance"] == "bearish" and data["verdicts"][1]["bearish"] == 3
    assert [m["key"] for m in data["personas_meta"]] == ["warren_buffett", "ben_graham", "michael_burry"]
    assert data["personas_meta"][0]["name_zh"] == "沃伦·巴菲特"
    assert data["top"][0]["ticker"] == "QLTY" and len(data["top"]) == 2
    assert data["cache_hits"] == [] and data["run_id"] and data["data_gaps"] == []
    assert "position" not in data["verdicts"][0]
    first_calls = fake.calls
    assert first_calls > 0

    # persisted: run log + detail + workspace run summary
    runs = client.get("/api/lab/committee/runs").json()["items"]
    assert runs[0]["run_id"] == data["run_id"] and runs[0]["tickers"] == ["QLTY", "DSTR"]
    detail = client.get(f"/api/lab/committee/runs/{data['run_id']}").json()
    assert detail["verdicts"][0]["signals"][0]["persona"] == "warren_buffett"
    lab_runs = client.get("/api/lab/runs").json()["items"]
    assert lab_runs[0]["kind"] == "committee" and lab_runs[0]["top"][0] == "QLTY"
    assert client.get("/api/lab/committee/runs/nope").status_code == 404

    # second run the same day hits the snapshot cache: zero data calls
    again = client.post("/api/lab/committee", json=body).json()
    assert again["cache_hits"] == ["DSTR", "QLTY"] and fake.calls == first_calls
    assert again["verdicts"][0]["consensus"] == data["verdicts"][0]["consensus"]


def test_committee_defaults_to_all_personas_and_rejects_unknown(client):
    res = client.post("/api/lab/committee", json={"tickers": ["QLTY"], "as_of": "2026-06-30"})
    assert res.status_code == 200
    assert len(res.json()["personas_meta"]) == 13 and len(res.json()["verdicts"][0]["signals"]) == 13
    bad = client.post("/api/lab/committee", json={"tickers": ["QLTY"], "personas": ["elon"]})
    assert bad.status_code == 400 and "unknown persona" in bad.json()["detail"]
    assert client.post("/api/lab/committee", json={"tickers": ["bad ticker!"]}).status_code == 400
    assert client.post("/api/lab/committee", json={"tickers": []}).status_code == 400


def test_committee_on_holdings_labels_actions(client, monkeypatch):
    portfolio = {
        "account": {"portfolio_value": 100_000.0},
        "positions": [
            {"symbol": "QLTY", "market_value": 20_000.0, "current_price": 150.0, "side": "PositionSide.LONG", "unrealized_pl_pct": 0.1},
            {"symbol": "DSTR", "market_value": 5_000.0, "current_price": 40.0, "side": "long", "unrealized_pl_pct": -0.2},
            {"symbol": "SHRT", "market_value": 1_000.0, "current_price": 1.0, "side": "PositionSide.SHORT"},
        ],
    }
    import v2.broker.alpaca_client as alpaca

    monkeypatch.setattr(alpaca, "get_portfolio", lambda: portfolio)
    # lean=False: Lynch and Fisher need the news / insider rows to reach their bullish votes
    res = client.post("/api/lab/committee", json={"source": "holdings", "as_of": "2026-06-30", "personas": ["warren_buffett", "peter_lynch", "phil_fisher"], "max_weight": 0.15, "lean": False})
    assert res.status_code == 200, res.text
    by = {v["ticker"]: v for v in res.json()["verdicts"]}
    assert set(by) == {"QLTY", "DSTR"}  # shorts are skipped
    assert by["DSTR"]["action"] == "减持候选" and by["DSTR"]["position"]["weight"] == pytest.approx(0.05)
    assert by["QLTY"]["position"]["weight"] == pytest.approx(0.20)
    # QLTY: Lynch + Fisher bullish, Buffett neutral → consensus > .2, agreement 2/3, but weight 20% >= cap 15% → hold
    assert by["QLTY"]["action"] == "持有" and "上限" in by["QLTY"]["action_reason"]
    assert by["QLTY"]["price"] == 150.0

    relaxed = client.post("/api/lab/committee", json={"source": "holdings", "as_of": "2026-06-30", "personas": ["warren_buffett", "peter_lynch", "phil_fisher"], "max_weight": 0.5, "lean": False}).json()
    assert {v["ticker"]: v["action"] for v in relaxed["verdicts"]}["QLTY"] == "增持候选"

    monkeypatch.setattr(alpaca, "get_portfolio", lambda: {"account": {}, "positions": []})
    assert client.post("/api/lab/committee", json={"source": "holdings"}).status_code == 400


def test_committee_on_watchlist_and_screening(client, monkeypatch, tmp_path):
    from v2.bot import state as bot_state

    monkeypatch.setattr(bot_state, "_DB_PATH", tmp_path / "bot_state.db")
    assert client.post("/api/lab/committee", json={"source": "watchlist"}).status_code == 400  # empty
    client.post("/api/watchlist", json={"ticker": "dstr", "note": ""})
    res = client.post("/api/lab/committee", json={"source": "watchlist", "as_of": "2026-06-30", "personas": ["warren_buffett"]})
    assert res.status_code == 200 and [v["ticker"] for v in res.json()["verdicts"]] == ["DSTR"]

    monkeypatch.setattr(
        workspace, "_run_screening",
        lambda body: {"kind": "screening", "date": "2026-06-30", "universe_size": 30, "candidates": [{"ticker": "QLTY", "price": 150.0, "market_cap": 250e9, "revenue_growth": 0.12, "gross_margin": 0.62}]},
    )
    res = client.post("/api/lab/committee", json={"source": "screening", "as_of": "2026-06-30", "personas": ["warren_buffett"], "screening": {"revenue_growth_min": 0.1}})
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["screening"]["n_candidates"] == 1 and data["screening"]["universe_size"] == 30
    assert [v["ticker"] for v in data["verdicts"]] == ["QLTY"]


def test_personas_and_scoreboard_endpoints(client):
    items = client.get("/api/lab/committee/personas").json()["items"]
    assert len(items) == 13 and items[0] == {
        "key": "warren_buffett", "name": "Warren Buffett", "name_zh": "沃伦·巴菲特",
        "style": "seeks wonderful companies at a fair price", "period": "ttm", "lookback": 10, "needs": [],
    }
    board = client.get("/api/lab/committee/scoreboard").json()
    assert board["kind"] == "scoreboard" and board["items"] == [] and board["baseline"]["n_1m"] == 0 and board["baseline"]["up_rate_1m"] is None
    assert board["counts"] == {"runs": 0, "tickers": 0, "votes": 0, "scored_1m": 0, "scored_3m": 0, "due_1m": 0, "due_3m": 0}
    client.post("/api/lab/committee", json={"tickers": ["QLTY"], "as_of": "2026-06-30", "personas": ["warren_buffett", "ben_graham"]})
    counts = client.get("/api/lab/committee/scoreboard").json()["counts"]
    assert counts["runs"] == 1 and counts["tickers"] == 1 and counts["votes"] == 2 and counts["due_1m"] == 2


def test_store_forward_return_backfill(tmp_path):
    store = PersonaStore(tmp_path / "p.db")
    payload = {
        "as_of": "2026-01-15", "personas": ["warren_buffett"], "elapsed_s": 0.1,
        "verdicts": [{"ticker": "AAA", "price": 100.0, "signals": [
            {"persona": "warren_buffett", "as_of": "2026-01-15", "signal": "bullish", "confidence": 70, "score": 8, "max_score": 10, "abstained": False, "facts": {}},
            {"persona": "ben_graham", "as_of": "2026-01-15", "signal": "neutral", "confidence": 0, "score": 0, "max_score": 0, "abstained": True, "facts": {}},
        ]}],
    }
    run_id = store.save_run(payload, source="tickers")
    assert store.get_run(run_id)["run_id"] == run_id
    pending = store.signals_awaiting_forward_returns(older_than_days=30)
    assert [p["persona"] for p in pending] == ["warren_buffett"]  # abstentions never scored
    store.set_forward_return(pending[0]["id"], column="fwd_1m", value=0.08)
    assert store.signals_awaiting_forward_returns(older_than_days=30) == []
    board = store.persona_scoreboard()
    assert len(board) == 1 and board[0]["persona"] == "warren_buffett"
    row = board[0]
    assert row["n"] == 1 and row["hits"] == 1 and row["hit_rate"] == 1.0 and row["avg_directional_1m"] == 0.08
    assert row["n_3m"] == 0 and row["hit_rate_3m"] is None and row["neutral"] == 0 and row["abstained"] == 0 and row["votes"] == 1
    assert 0.2 < row["ci_low"] < 0.3 and row["ci_high"] == 1.0            # Wilson on 1 / 1: wide
    assert row["baseline_1m"] == 1.0 and row["edge_1m"] == 0.0            # the one ticker rose: always-bullish would also have hit
    base = store.scoreboard_baseline()
    assert base["n_1m"] == 1 and base["up_rate_1m"] == 1.0 and base["avg_return_1m"] == 0.08 and base["n_3m"] == 0
    latest = store.latest_signals("AAA")
    assert {s["persona"] for s in latest} == {"warren_buffett", "ben_graham"}
    with pytest.raises(ValueError):
        store.set_forward_return(1, column="drop table", value=0)


# ------------------------------------------------------------------ narration

def test_narrate_endpoint_adds_grounded_text_without_changing_the_verdict(client, monkeypatch):
    import json as _json

    from v2.agent_common.llm import LLMResponse, ScriptedLLM

    run = client.post("/api/lab/committee", json={"tickers": ["QLTY"], "as_of": "2026-06-30", "personas": ["warren_buffett"]}).json()
    sig = run["verdicts"][0]["signals"][0]
    roe_line = sig["parts"][0]["details"].split(";")[0]
    reply = _json.dumps({"signal": sig["signal"], "confidence": sig["confidence"], "reasoning": f"{roe_line}，估值偏贵，先观望。"})
    monkeypatch.setattr(committee, "_narrate_llm", lambda: ScriptedLLM([LLMResponse(text=reply)]))

    res = client.post("/api/lab/committee/narrate", json={"run_id": run["run_id"], "ticker": "qlty", "persona": "warren_buffett"})
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["narrative"].startswith(roe_line) and data["narrative_grounded"] is True
    assert data["signal"] == sig["signal"] and data["confidence"] == sig["confidence"]
    # persisted into the stored run
    stored = client.get(f"/api/lab/committee/runs/{run['run_id']}").json()
    assert stored["verdicts"][0]["signals"][0]["narrative"] == data["narrative"]

    # a reply that flips the signal is discarded → 503 with the reason
    flipped = _json.dumps({"signal": "bearish" if sig["signal"] != "bearish" else "bullish", "confidence": sig["confidence"], "reasoning": "nope"})
    monkeypatch.setattr(committee, "_narrate_llm", lambda: ScriptedLLM([LLMResponse(text=flipped)]))
    res = client.post("/api/lab/committee/narrate", json={"run_id": run["run_id"], "ticker": "QLTY", "persona": "warren_buffett"})
    assert res.status_code == 503 and "rule-based reasoning stands" in res.json()["detail"]

    assert client.post("/api/lab/committee/narrate", json={"run_id": "nope", "ticker": "QLTY", "persona": "warren_buffett"}).status_code == 404
    assert client.post("/api/lab/committee/narrate", json={"run_id": run["run_id"], "ticker": "QLTY", "persona": "ben_graham"}).status_code == 404


# ------------------------------------------------------------------- backfill

class _Prices:
    """Deterministic price path: 100 on day 0, +0.1 per calendar day."""

    def __init__(self):
        self.calls = 0

    def get_prices(self, ticker, start, end):
        from datetime import date, timedelta

        self.calls += 1
        d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
        base = date(2026, 1, 1)
        return [{"time": (d0 + timedelta(days=i)).isoformat(), "close": 100 + 0.1 * ((d0 + timedelta(days=i)) - base).days} for i in range((d1 - d0).days + 1) if (d0 + timedelta(days=i)).weekday() < 5]


def test_forward_backfill_fills_due_horizons_and_scores_personas(tmp_path):
    from datetime import date

    from v2.personas.forward import backfill_forward_returns

    store = PersonaStore(tmp_path / "p.db")
    payload = {"as_of": "2026-01-15", "personas": ["a", "b"], "verdicts": [{"ticker": "AAA", "signals": [
        {"persona": "a", "as_of": "2026-01-15", "signal": "bullish", "confidence": 70, "score": 8, "max_score": 10, "abstained": False, "facts": {}},
        {"persona": "b", "as_of": "2026-01-15", "signal": "bearish", "confidence": 60, "score": 2, "max_score": 10, "abstained": False, "facts": {}},
    ]}]}
    store.save_run(payload, source="tickers")
    prices = _Prices()

    # 45 days later: 1m due, 3m not yet
    report = backfill_forward_returns(store, prices, today=date(2026, 3, 1))
    assert report.filled == 2 and report.by_column == {"fwd_1m": 2} and prices.calls == 1
    board = {row["persona"]: row for row in store.persona_scoreboard()}
    assert board["a"]["hits"] == 1 and board["b"]["hits"] == 0  # price rose: bull right, bear wrong
    assert board["a"]["avg_directional_1m"] == pytest.approx(0.0296, abs=0.002)

    # idempotent: nothing left for 1m, 3m fills once due
    again = backfill_forward_returns(store, prices, today=date(2026, 3, 1))
    assert again.filled == 0
    later = backfill_forward_returns(store, prices, today=date(2026, 6, 1))
    assert later.by_column == {"fwd_3m": 2}
    assert store.signals_awaiting_forward_returns(older_than_days=91, column="fwd_3m") == []
    board = {row["persona"]: row for row in store.persona_scoreboard()}
    assert board["a"]["n_3m"] == 1 and board["a"]["hit_rate_3m"] == 1.0 and board["b"]["hit_rate_3m"] == 0.0
    # one (ticker, as_of) rose → baseline P(up) = 1; the bull's mix baseline is 1, the bear's is 0
    assert board["a"]["baseline_1m"] == 1.0 and board["b"]["baseline_1m"] == 0.0 and board["b"]["edge_1m"] == 0.0
    assert store.scoreboard_baseline()["n_3m"] == 1


def test_backfill_endpoint_reports_and_returns_scoreboard(client, monkeypatch):
    from v2.personas import forward as forward_mod

    monkeypatch.setattr(forward_mod, "_default_price_source", lambda: _Prices())
    res = client.post("/api/lab/committee/backfill", json={"columns": ["fwd_1m"]})
    assert res.status_code == 200, res.text
    assert res.json()["kind"] == "backfill" and res.json()["checked"] == 0 and res.json()["scoreboard"] == []



# ------------------------------------------------------------------- lab store

def test_lab_runs_can_be_deleted_singly_and_by_age(client, monkeypatch, tmp_path):
    from app.lab_store import LabRunStore

    store = LabRunStore(tmp_path / "lab.db")
    monkeypatch.setattr(workspace, "_LAB_STORE", store)
    a = store.save("backtest", params={"x": 1}, summary={"tickers": ["AAA"]}, result={"kind": "backtest"})
    b = store.save("screening", params={}, summary={"tickers": []}, result={"kind": "screening"})
    # age one row artificially
    with store._conn() as conn:
        conn.execute("UPDATE lab_runs SET created_at = '2020-01-01T00:00:00' WHERE id = ?", (b,))
    assert client.get("/api/lab/runs").json()["counts"] == {"backtest": 1, "screening": 1}
    res = client.post("/api/lab/runs/cleanup", json={"older_than_days": 30}).json()
    assert res["deleted"] == 1 and res["counts"] == {"backtest": 1}
    assert client.delete(f"/api/lab/runs/{a}").json()["counts"] == {}
    assert client.delete(f"/api/lab/runs/{a}").status_code == 404
    assert client.get(f"/api/lab/runs/{a}").status_code == 404


def test_lab_runs_persist_across_kinds_and_reopen(client, monkeypatch):
    monkeypatch.setattr(workspace, "_run_screening", lambda body: {"kind": "screening", "universe": body.universe, "tickers": ["QLTY"], "universe_size": 1, "candidates": [{"ticker": "QLTY"}]})
    scr = client.post("/api/lab/screening", json={"universe": "custom", "tickers": ["qlty"], "revenue_growth_min": 0.1}).json()
    com = client.post("/api/lab/committee", json={"tickers": ["QLTY"], "as_of": "2026-06-30", "personas": ["warren_buffett"]}).json()
    assert scr["lab_run_id"] and com["lab_run_id"] and scr["lab_run_id"] != com["lab_run_id"]

    runs = client.get("/api/lab/runs").json()
    assert [r["kind"] for r in runs["items"]] == ["committee", "screening"]
    assert runs["counts"] == {"committee": 1, "screening": 1}
    assert runs["items"][1]["n_candidates"] == 1 and runs["items"][1]["candidates"] == ["QLTY"]
    assert runs["items"][0]["stances"]["neutral"] + runs["items"][0]["stances"]["bearish"] + runs["items"][0]["stances"]["bullish"] == 1

    detail = client.get(f"/api/lab/runs/{scr['lab_run_id']}").json()
    assert detail["kind"] == "screening" and detail["params"]["revenue_growth_min"] == 0.1 and detail["result"]["candidates"][0]["ticker"] == "QLTY"
    assert client.get("/api/lab/runs/nope").status_code == 404
    assert [r["kind"] for r in client.get("/api/lab/runs?kind=screening").json()["items"]] == ["screening"]


def test_universe_resolution_for_lab_engines(client, monkeypatch):
    import v2.broker.alpaca_client as alpaca
    from app import sources

    monkeypatch.setattr(alpaca, "get_portfolio", lambda: {"account": {"portfolio_value": 100.0}, "positions": [{"symbol": "AAA", "market_value": 50.0, "side": "long"}, {"symbol": "SHRT", "market_value": 1.0, "side": "short"}]})
    assert sources.resolve_universe("holdings")[0] == ["AAA"]
    assert sources.resolve_universe("tech30")[0][:2] == ["AAPL", "MSFT"]
    assert sources.resolve_universe("custom", ["nvda", "nvda", "amd"])[0] == ["NVDA", "AMD"]
    with pytest.raises(ValueError):
        sources.resolve_universe("custom", [])

    seen = {}
    monkeypatch.setattr(workspace, "_run_backtest", lambda body: seen.setdefault("body", body) and {"kind": "backtest", "strategy": body.strategy, "universe": body.universe, "tickers": ["AAA"], "metrics": {"n_trades": 1}})
    res = client.post("/api/lab/backtest", json={"universe": "holdings", "holding_days": 7})
    assert res.status_code == 200 and seen["body"].universe == "holdings" and seen["body"].holding_days == 7
    assert client.post("/api/lab/backtest", json={"universe": "nowhere"}).status_code == 422


# ------------------------------------------------------------------ universes

def test_index_universes_resolve_only_for_the_screener(client, monkeypatch):
    from app import sources

    tickers, meta = sources.resolve_universe("dow30", limit=sources.BIG_LIMIT)
    assert len(tickers) == 30 and "AAPL" in tickers and meta["as_of"]
    with pytest.raises(ValueError, match="at most 60"):
        sources.resolve_universe("sp500")  # default cap is the committee/backtest cap
    from v2.screening.universe import TECH_30

    items = client.get("/api/lab/universes").json()["items"]
    # Nasdaq-100 can contain more than 100 listed securities when one company
    # contributes multiple share classes, so validate the lower bound rather
    # than pinning a live constituent feed to an exact count.
    assert items["sp500"]["size"] > 450 and items["nasdaq100"]["size"] >= 100 and items["tech30"]["size"] == len(TECH_30)
    # paid strategies refuse an index universe cleanly; the free momentum strategy takes the whole index as a job
    res = client.post("/api/lab/backtest", json={"universe": "sp500", "strategy": "pead"})
    assert res.status_code in (400, 503) and "at most 60" in res.json()["detail"]
    monkeypatch.setattr(workspace, "_run_backtest", lambda body, on_tick=None: {"kind": "backtest", "strategy": body.strategy, "universe": body.universe, "tickers": [], "trades": [], "metrics": None, "equity_curve": []})
    job = client.post("/api/lab/backtest", json={"universe": "sp500", "strategy": "momentum"}).json()
    assert job["kind"] == "backtest_job" and job["total"] > 450
    for _ in range(100):  # heavy jobs run one at a time, so let this one finish before starting the next
        if client.get(f"/api/lab/backtest/jobs/{job['job_id']}").json()["status"] != "running":
            break
        time.sleep(0.02)
    # a 100-ticker custom list (handed over from the screener) is fine for momentum, refused clearly for paid strategies
    many = [f"T{chr(65 + i // 26)}{chr(65 + i % 26)}" for i in range(100)]  # TAA … TDV: valid-looking symbols
    job = client.post("/api/lab/backtest", json={"universe": "custom", "tickers": many, "strategy": "momentum"}).json()
    assert job["kind"] == "backtest_job" and job["total"] == 100
    res = client.post("/api/lab/backtest", json={"universe": "custom", "tickers": many, "strategy": "pead"})
    assert res.status_code == 400 and "at most 60" in res.json()["detail"]


def test_large_screening_runs_as_a_polled_job(client, monkeypatch):
    import time

    ticks: list[int] = []

    def fake_run(body, on_tick=None):
        for i in range(3):
            if on_tick:
                on_tick(i)
                ticks.append(i)
        return {"kind": "screening", "universe": body.universe, "tickers": ["AAA"] * 3, "universe_size": 3, "candidates": [{"ticker": "AAA"}]}

    monkeypatch.setattr(workspace, "_run_screening", fake_run)
    from v2.screening.universe import TECH_30

    started = client.post("/api/lab/screening?background=true", json={"universe": "tech30"}).json()
    assert started["status"] in ("running", "completed") and started["job_id"] and started["total"] == len(TECH_30)
    for _ in range(50):
        job = client.get(f"/api/lab/screening/jobs/{started['job_id']}").json()
        if job["status"] == "completed":
            break
        time.sleep(0.05)
    assert job["status"] == "completed" and job["done"] == job["total"] == len(TECH_30)
    assert job["result"]["candidates"][0]["ticker"] == "AAA" and job["result"]["lab_run_id"]
    assert ticks == [0, 1, 2]
    assert client.get("/api/lab/runs").json()["items"][0]["kind"] == "screening"
    assert client.get("/api/lab/screening/jobs/nope").status_code == 404

    # small universes still answer inline
    inline = client.post("/api/lab/screening", json={"universe": "custom", "tickers": ["AAA"]}).json()
    assert inline["kind"] == "screening" and "job_id" not in inline

    def boom(body, on_tick=None):
        raise RuntimeError("provider down")

    monkeypatch.setattr(workspace, "_run_screening", boom)
    failed = client.post("/api/lab/screening?background=true", json={"universe": "dow30"}).json()
    for _ in range(50):
        job = client.get(f"/api/lab/screening/jobs/{failed['job_id']}").json()
        if job["status"] == "failed":
            break
        time.sleep(0.05)
    assert job["status"] == "failed" and "provider down" in job["error"]


def test_screening_rules_are_optional_and_missing_fields_fail_closed():
    from app.screening import CRITERIA, DEFAULT_RULES, Rule, screen

    r = Rule(field="gross_margin", op="gte", value=0.5)
    assert r.passes({"gross_margin": 0.6}) and not r.passes({"gross_margin": 0.4})
    assert not r.passes({"gross_margin": None}) and not r.passes({"gross_margin": float("nan")}) and not r.passes({})
    assert r.describe() == "毛利率 ≥ 50%"
    assert Rule(field="market_cap", op="gte", value=10e9).describe() == "市值 ≥ $10B"
    assert Rule(field="price_to_earnings_ratio", op="lte", value=25).describe() == "市盈率 ≤ 25"
    assert all(rule.field in CRITERIA for rule in DEFAULT_RULES)

    # Legacy threshold fields still work and fold into rules; none given → defaults; unknown field → 400.
    body = workspace.ScreeningInput(universe="custom", tickers=["AAA"])
    assert [x.model_dump() for x in body.effective_rules()] == [x.model_dump() for x in DEFAULT_RULES]
    body = workspace.ScreeningInput(universe="custom", tickers=["AAA"], gross_margin_min=0.3)
    assert [x.describe() for x in body.effective_rules()] == ["毛利率 ≥ 30%"]
    body = workspace.ScreeningInput(universe="custom", tickers=["AAA"], rules=[{"field": "nope", "op": "gte", "value": 1}])
    with pytest.raises(ValueError, match="nope"):
        body.effective_rules()

    # Only the enabled rules are applied; a ticker without enough price history is reported, not silently dropped.
    class Metrics:
        def get_financial_metrics(self, ticker, end_date, limit=1, **kw):
            return [{"market_cap": 5e9 if ticker == "SMALL" else 50e9, "gross_margin": 0.7, "price_to_earnings_ratio": 40 if ticker == "PRICEY" else 15}]

    class Prices:
        def get_prices(self, ticker, start, end):
            if ticker == "NEW":
                return [{"close": 10.0}] * 5
            return [{"close": 100.0 + (i % 7)} for i in range(300)]

    ticks = []
    out = screen(["BIG", "SMALL", "PRICEY", "NEW"], Metrics(), Prices(), [Rule(field="price_to_earnings_ratio", op="lte", value=25)], on_tick=ticks.append)
    assert ticks == [0, 1, 2, 3]
    assert [c["ticker"] for c in out["candidates"]] == ["BIG", "SMALL"]  # market-cap rule not enabled → SMALL passes
    assert out["rejected_count"] == 1 and out["no_data"] == ["NEW"]
    assert out["reject_reasons"] == {"市盈率 ≤ 25": 1} and out["rules_text"] == ["市盈率 ≤ 25"]
    assert {"volatility", "return_3m", "pct_from_52w_high"} <= set(out["candidates"][0])


def test_screening_criteria_endpoint_lists_fields_and_defaults(client):
    body = client.get("/api/lab/screening/criteria").json()
    assert body["kind"] == "criteria" and len(body["items"]) >= 20
    assert body["items"]["gross_margin"] == {"label": "毛利率", "unit": "pct", "source": "metrics"}
    assert body["defaults"][0] == {"field": "market_cap", "op": "gte", "value": 10e9}



def test_screen_data_skips_uncovered_tickers_and_counts_fd_requests():
    class Boom(Exception):
        pass

    class Metrics:
        misses = 3

        def get_financial_metrics(self, ticker, end, limit=1):
            if ticker == "BRK.B":
                raise Boom("Financial Datasets EMPTY_DATA at /financial-metrics/ (HTTP 404)")
            return [{"market_cap": 1.0}]

        def close(self):
            self.closed = True

    class Earnings:
        def get_earnings(self, ticker):
            if ticker == "AAPL":
                raise Boom("no earnings")
            return {"eps": 1}

    metrics, earnings = Metrics(), Earnings()
    with workspace._ScreenData(metrics, metrics_is_fd=True, earnings_client=earnings) as fd:
        assert fd.get_financial_metrics("AAPL", "2026-06-30", limit=1) == [{"market_cap": 1.0}]
        assert fd.get_financial_metrics("BRK.B", "2026-06-30", limit=1) == []
        assert fd.get_earnings("AAPL") is None and fd.get_earnings("MSFT") == {"eps": 1}
        assert fd.misses == 3  # pass-through
    assert set(fd.skipped) == {"BRK.B", "AAPL"} and "HTTP 404" in fd.skipped["BRK.B"]
    assert fd.fd_requests == {"financial_metrics": 2, "earnings": 2}
    assert metrics.closed

    free = workspace._ScreenData(Metrics(), metrics_is_fd=False)
    free.get_financial_metrics("AAPL", "2026-06-30")
    assert free.fd_requests == {} and free.get_earnings("AAPL") is None


def test_screen_clients_default_to_free_yfinance_and_bill_only_when_asked(monkeypatch):
    from app.routers.workspace import ScreeningInput, _screen_clients

    free = _screen_clients(ScreeningInput())
    assert free._metrics_is_fd is False and free._earnings is None
    paid = _screen_clients(ScreeningInput(data_source="fd", with_earnings=True))
    assert paid._metrics_is_fd is True and paid._earnings is not None
    mixed = _screen_clients(ScreeningInput(data_source="yfinance", with_earnings=True))
    assert mixed._metrics_is_fd is False and mixed._earnings is not None


def test_committee_lean_mode_skips_news_and_insiders_and_reports_cost(client, fake):
    lean = client.post("/api/lab/committee", json={"tickers": ["QLTY"], "as_of": "2026-06-30", "personas": ["charlie_munger"]}).json()
    assert lean["lean"] is True
    assert set(lean["fd_requests"]) == {"financial_metrics", "line_items"} and "news" not in lean["fd_requests"]
    assert lean["fd_cost_usd"] == pytest.approx(0.08)  # 2 metrics + 2 line items at $0.02
    full = client.post("/api/lab/committee", json={"tickers": ["DSTR"], "as_of": "2026-06-30", "personas": ["charlie_munger"], "lean": False}).json()
    assert {"news", "insider_trades"} <= set(full["fd_requests"]) and full["fd_cost_usd"] > lean["fd_cost_usd"]
    assert "company_facts" not in full["fd_requests"]  # market cap comes from the metrics row
    pricing = client.get("/api/lab/committee/pricing").json()
    assert pricing["committee_per_ticker"]["lean"] == pytest.approx(0.10) and pricing["committee_per_ticker"]["full"] == pytest.approx(0.14)


def test_fd_prices_override(monkeypatch):
    from app import fd_pricing

    monkeypatch.setenv("FD_PRICES", '{"financial_metrics": 0.02, "bogus": 9}')
    assert fd_pricing.prices()["financial_metrics"] == 0.02 and "bogus" not in fd_pricing.prices()
    assert fd_pricing.cost({"financial_metrics": 10, "earnings": 3}) == pytest.approx(0.26)
    monkeypatch.setenv("FD_PRICES", "not json")
    assert fd_pricing.prices()["financial_metrics"] == 0.02


def test_backtest_strategies_and_data_feeds(client, monkeypatch, tmp_path):
    """Strategy names validate; momentum on yfinance costs nothing; the committee runs as a job."""
    from v2.backtesting.strategies import BacktestData, PriceCache
    from datetime import date, timedelta

    class Bar:
        def __init__(self, t, c):
            self.time, self.close = t, c

    class Src:
        def get_prices(self, ticker, start, end):
            d, out, i = date.fromisoformat(start), [], 0
            while d <= date.fromisoformat(end):
                if d.weekday() < 5:
                    out.append(Bar(d.isoformat(), 100 * (1.001 ** i))); i += 1
                d += timedelta(days=1)
            return out

    @contextmanager
    def fake_data(body):
        yield BacktestData(prices=PriceCache(Src()), fd=None, raw=None)

    monkeypatch.setattr(workspace, "_backtest_data", fake_data)
    res = client.post("/api/lab/backtest", json={"universe": "custom", "tickers": ["AAA", "BBB"], "strategy": "momentum", "history_days": 200, "holding_days": 21, "top_n": 1})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["strategy"] == "momentum" and body["data_source"] == "yfinance" and body["fd_cost_usd"] == 0 and body["fd_requests"] == {}
    assert body["metrics"]["n_trades"] >= 5 and body["params"]["lookback_days"] == 252 and body["notes"]["price_failures"] == {}
    assert body["params"]["cost_bps"] == 10 and body["metrics"]["cost_bps"] == 10 and body["metrics"]["n_periods"] >= 5
    # SPY buy-and-hold over the same span, and the strategy's excess over it
    assert body["benchmark"]["ticker"] == "SPY" and body["benchmark"]["start"] == body["trades"][0]["entry_date"] and body["benchmark"]["total_return_pct"] > 0
    assert abs(body["excess_return_pct"] - (body["metrics"]["total_return_pct"] - body["benchmark"]["total_return_pct"])) < 1e-6
    assert client.get("/api/lab/runs?kind=backtest").json()["items"][0]["fd_cost_usd"] == 0

    assert client.post("/api/lab/backtest", json={"tickers": ["AAA"], "strategy": "bollinger"}).status_code == 422
    assert client.post("/api/lab/backtest", json={"tickers": ["AAA"], "strategy": "momentum", "data_source": "bloomberg"}).status_code == 422

    # the committee strategy always runs as a polled job (it is slow and paid)
    def fake_run(body, on_tick=None):
        for i in range(3):
            on_tick(i)
        return {"kind": "backtest", "strategy": body.strategy, "data_source": body.data_source, "universe": body.universe, "tickers": ["AAA"],
                "params": body.params(), "fd_requests": {"financial_metrics": 8, "line_items": 8}, "fd_cost_usd": 0.32, "notes": {}, "trades": [], "metrics": None, "equity_curve": []}

    monkeypatch.setattr(workspace, "_run_backtest", fake_run)
    job = client.post("/api/lab/backtest", json={"tickers": ["AAA"], "strategy": "committee", "history_days": 365, "holding_days": 63}).json()
    assert job["kind"] == "backtest_job" and job["status"] == "running" and job["total"] == 4  # 1 ticker × 4 quarterly dates
    for _ in range(50):
        job = client.get(f"/api/lab/backtest/jobs/{job['job_id']}").json()
        if job["status"] != "running":
            break
        time.sleep(0.05)
    assert job["status"] == "completed" and job["result"]["fd_cost_usd"] == 0.32 and job["result"]["params"]["lean"] is True
    assert client.get(f"/api/lab/screening/jobs/{job['job_id']}").status_code == 404  # wrong kind
    assert client.get("/api/lab/runs?kind=backtest").json()["items"][0]["strategy"] == "committee"


def test_backtest_result_carries_a_yearly_table_against_the_benchmark(client, monkeypatch):
    from v2.backtesting.strategies import BacktestData, PriceCache
    from datetime import date, timedelta

    class Bar:
        def __init__(self, t, c):
            self.time, self.close = t, c

    class Src:
        def get_prices(self, ticker, start, end):
            d, out, i = date.fromisoformat(start), [], 0
            rate = 1.0005 if ticker == "SPY" else 1.001
            while d <= date.fromisoformat(end):
                if d.weekday() < 5:
                    out.append(Bar(d.isoformat(), 100 * (rate ** i))); i += 1
                d += timedelta(days=1)
            return out

    @contextmanager
    def fake_data(body):
        yield BacktestData(prices=PriceCache(Src()), fd=None, raw=None)

    monkeypatch.setattr(workspace, "_backtest_data", fake_data)
    res = client.post("/api/lab/backtest", json={"universe": "custom", "tickers": ["AAA", "BBB"], "strategy": "momentum", "history_days": 800, "holding_days": 63, "top_n": 2, "cost_bps": 0})
    assert res.status_code == 200, res.text
    body = res.json()
    yearly = body["yearly"]
    assert len(yearly) >= 2 and [r["year"] for r in yearly] == sorted(r["year"] for r in yearly)
    assert sum(r["periods"] for r in yearly) == body["metrics"]["n_periods"] and sum(r["trades"] for r in yearly) == body["metrics"]["n_trades"]
    for r in yearly:
        assert r["return_pct"] > 0 and r["benchmark_pct"] > 0 and abs(r["excess_pct"] - (r["return_pct"] - r["benchmark_pct"])) < 1e-6  # 2 × $10k on $100k: equity return is diluted
        assert r["start"][:4] == r["year"] and r["end"] >= r["start"]
    # equity chains: start of year n+1 = start of year n + that year's P&L
    assert abs(yearly[1]["start_equity"] - (yearly[0]["start_equity"] + yearly[0]["pnl"])) < 0.02
    # 2 positions × $10k on $100k: 20 % utilisation; the same P&L on the deployed $20k is 5× the diluted figure
    dep = body["deployment"]
    assert dep["positions_per_period"] == 2 and dep["deployed_usd"] == 20_000 and dep["utilization"] == 0.2
    assert abs(dep["on_deployed"]["total_return_pct"] - body["metrics"]["total_return_pct"] * 5) < 1e-4
    assert dep["on_deployed"]["max_drawdown_pct"] >= body["metrics"]["max_drawdown_pct"]
    assert abs(dep["on_deployed"]["excess_return_pct"] - (dep["on_deployed"]["total_return_pct"] - body["benchmark"]["total_return_pct"])) < 1e-6
    for r in yearly:
        assert abs(r["return_on_deployed_pct"] - r["pnl"] / 20_000) < 1e-6 and abs(r["excess_on_deployed_pct"] - (r["return_on_deployed_pct"] - r["benchmark_pct"])) < 1e-6


def test_momentum_sweep_runs_the_grid_on_one_price_load(client, monkeypatch):
    from v2.backtesting.strategies import BacktestData, PriceCache
    from datetime import date, timedelta

    class Bar:
        def __init__(self, t, c):
            self.time, self.close = t, c

    calls: list[str] = []

    class Src:
        def get_prices(self, ticker, start, end):
            calls.append(ticker)
            d, out, i = date.fromisoformat(start), [], 0
            rate = {"AAA": 1.0012, "BBB": 1.0006, "CCC": 0.9995, "SPY": 1.0004}.get(ticker, 1.0)
            while d <= date.fromisoformat(end):
                if d.weekday() < 5:
                    out.append(Bar(d.isoformat(), 100 * (rate ** i))); i += 1
                d += timedelta(days=1)
            return out

    @contextmanager
    def fake_bundle(data_source, *, needs_fd, persona_client=False):
        assert needs_fd is False
        yield BacktestData(prices=PriceCache(Src()), fd=None, raw=None)

    monkeypatch.setattr(workspace, "_data_bundle", fake_bundle)
    grid = {"top_ns": [1, 2], "holding_days_list": [21, 42], "near_high_pcts": [None, 0.10]}
    job = client.post("/api/lab/backtest/sweep", json={"universe": "custom", "tickers": ["AAA", "BBB", "CCC"], "history_days": 400, "cost_bps": 5, **grid}).json()
    assert job["kind"] == "backtest_job" and job["total"] == 3 + 8  # tickers to load + 8 combos
    for _ in range(100):
        job = client.get(f"/api/lab/backtest/jobs/{job['job_id']}").json()
        if job["status"] != "running":
            break
        time.sleep(0.05)
    assert job["status"] == "completed", job
    result = job["result"]
    assert result["kind"] == "sweep" and result["strategy"] == "momentum" and result["fd_cost_usd"] == 0
    assert result["params"]["grid"] == grid and result["params"]["cost_bps"] == 5 and result["params"]["sizing"] == "capital / top_n"
    rows = result["rows"]
    assert {r["per_trade"] for r in rows} == {100_000.0, 50_000.0}  # fully invested: capital / top_n
    assert len(rows) == 8 and {(r["top_n"], r["holding_days"], r["near_high_pct"]) for r in rows} == {(n, h, nh) for n in (1, 2) for h in (21, 42) for nh in (None, 0.10)}
    # one price fetch per ticker (+ SPY) for the whole grid
    assert sorted(calls) == ["AAA", "BBB", "CCC", "SPY"]
    for r in rows:
        assert r["n_trades"] > 0 and r["n_periods"] > 0 and r["benchmark_pct"] is not None
        assert abs(r["excess_return_pct"] - (r["total_return_pct"] - r["benchmark_pct"])) < 1e-6
    by_key = {(r["top_n"], r["holding_days"], r["near_high_pct"]): r for r in rows}
    assert by_key[(1, 21, None)]["n_trades"] < by_key[(2, 21, None)]["n_trades"]           # more names per period → more trades
    assert by_key[(2, 21, None)]["n_periods"] > by_key[(2, 42, None)]["n_periods"]         # shorter holding → more periods
    # the run is listed with the best combination
    run = client.get("/api/lab/runs?kind=backtest").json()["items"][0]
    assert run["sweep"] is True and run["n_combos"] == 8 and run["best"]["top_n"] in (1, 2) and run["strategy"] == "momentum"
    reopened = client.get(f"/api/lab/runs/{run['id']}").json()
    assert reopened["kind"] == "backtest" and reopened["result"]["kind"] == "sweep"

    # one heavy job at a time: a second sweep / big backtest while one runs is refused with 409
    with workspace._JOBS_LOCK:
        workspace._JOBS["busy"] = {"job_id": "busy", "kind": "backtest_job", "status": "running", "done": 3, "total": 21, "universe": "sp500", "started_at": "2026-09-07T00:00:00"}
    try:
        res = client.post("/api/lab/backtest/sweep", json={"universe": "custom", "tickers": ["AAA"], "history_days": 400})
        assert res.status_code == 409 and "3 / 21" in res.json()["detail"]
        assert client.post("/api/lab/backtest", json={"tickers": ["AAA"], "strategy": "committee"}).status_code == 409
    finally:
        with workspace._JOBS_LOCK:
            workspace._JOBS.pop("busy", None)

    # grid validation
    assert client.post("/api/lab/backtest/sweep", json={"tickers": ["AAA"], "top_ns": []}).status_code == 422
    assert client.post("/api/lab/backtest/sweep", json={"tickers": ["AAA"], "near_high_pcts": [1.5]}).status_code == 422
    assert client.post("/api/lab/backtest/sweep", json={"tickers": [f"T{chr(65 + i // 26)}{chr(65 + i % 26)}" for i in range(601)]}).status_code == 422


def test_momentum_index_backtest_uses_point_in_time_members_when_history_exists(monkeypatch, tmp_path):
    import json
    from v2.screening import universes as U

    path = tmp_path / "universes.json"
    monkeypatch.setattr(U, "DATA_PATH", path)
    body = workspace.BacktestInput(universe="sp500", strategy="momentum", history_days=200, holding_days=63)
    tickers, meta = workspace._backtest_universe(body)
    assert meta["membership"] == {"point_in_time": False, "changes": 0, "mode": "none"} and len(tickers) > 450
    assert workspace._build_strategy(body).universe_at is None

    changes = [{"date": "2026-06-01", "added": "NEWCO", "removed": "OLDCO"}]
    path.write_text(json.dumps({"sp500": {"tickers": ["AAA", "BBB", "NEWCO"], "as_of": "2026-09-01", "changes": changes}}))
    tickers, meta = workspace._backtest_universe(body)
    assert meta["membership"]["point_in_time"] is True and "OLDCO" in tickers and "NEWCO" in tickers  # union over the window
    sweep = workspace.SweepInput(universe="sp500", history_days=200, holding_days_list=[21, 63])
    sweep_tickers, sweep_meta = workspace._sweep_universe(sweep)
    assert set(sweep_tickers) >= set(tickers) and sweep_meta["membership"]["point_in_time"] is True
    assert sweep.combos()[0] == {"top_n": 10, "holding_days": 21, "near_high_pct": None} and len(sweep.combos()) == 3 * 2 * 2
    strat = workspace._build_strategy(body)
    assert strat.universe_at is not None and strat.universe_at("2026-05-01") == ["AAA", "BBB", "OLDCO"] and "NEWCO" in strat.universe_at("2026-07-01")


def test_backtest_data_bundle_opens_fd_only_when_needed():
    from v2.backtesting.strategies import PriceCache

    free = workspace.BacktestInput(tickers=["AAA"], strategy="momentum", data_source="yfinance")
    with workspace._backtest_data(free) as data:
        assert data.raw is None and data.fd is None and isinstance(data.prices, PriceCache)
    paid_prices = workspace.BacktestInput(tickers=["AAA"], strategy="momentum", data_source="fd")
    with workspace._backtest_data(paid_prices) as data:
        assert data.raw is not None and data.fd is None and data.prices._chunk == workspace.FD_PRICE_CHUNK_DAYS
    pead = workspace.BacktestInput(tickers=["AAA"], strategy="pead")
    with workspace._backtest_data(pead) as data:
        assert data.raw is not None and data.fd is None  # PEAD reads earnings through the raw client
    assert free.params()["top_n"] == 5 and "earnings_limit" not in free.params() and pead.params()["earnings_limit"] == 8


def test_event_study_takes_prices_from_the_chosen_feed_and_bills_earnings(client, monkeypatch):
    """Earnings history is always FD (one request per ticker); prices follow data_source."""
    import v2.event_study as es
    from v2.event_study.models import EventStudyResult
    from v2.backtesting.strategies import BacktestData

    seen = {}

    def fake_compute_car(tickers, data, *, earnings_limit, n_bootstrap, require_eps_surprise, dedupe, group_by):
        seen["mode"] = (dedupe, group_by)
        assert isinstance(data, BacktestData)
        seen["chunk"] = data.prices._chunk
        for t in tickers:
            data.count("earnings")            # what get_earnings_history does per ticker
        data.prices.requests = 3              # pretend three price fetches happened
        return EventStudyResult(events=[], aggregates=[], skipped_tickers=list(tickers))

    monkeypatch.setattr(es, "compute_car", fake_compute_car)
    free = client.post("/api/lab/event-study", json={"tickers": ["AAA", "BBB"]}).json()
    assert free["data_source"] == "yfinance" and free["fd_requests"] == {"earnings": 2} and free["fd_cost_usd"] == 0.04 and seen["chunk"] is None
    assert seen["mode"] == (True, "surprise") and free["params"]["group_by"] == "surprise"
    paid = client.post("/api/lab/event-study", json={"tickers": ["AAA", "BBB"], "data_source": "fd"}).json()
    assert paid["fd_requests"] == {"earnings": 2, "prices": 3} and paid["fd_cost_usd"] == 0.1 and seen["chunk"] == workspace.FD_PRICE_CHUNK_DAYS
    assert client.get("/api/lab/runs?kind=event_study").json()["items"][0]["fd_cost_usd"] == 0.1
    assert client.post("/api/lab/event-study", json={"tickers": ["AAA"], "data_source": "nope"}).status_code == 422
