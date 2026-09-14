from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

from v2.research.cache import ResearchCache
from v2.research.engine import ResearchEngine, resolve_modules
from v2.research.services import NewsDataService, ResearchServices, TechnicalAnalysisService
from v2.research.store import ENGINE_VERSION, ResearchStore

from web.backend.tests.test_research import FakeFD, FakePrices, fake_services


def test_dependency_resolver_and_single_module_run(tmp_path: Path):
    fd = FakeFD()
    engine = ResearchEngine(fd=fd, prices=FakePrices(), cache=ResearchCache(tmp_path / "research.db"), services=fake_services())
    result = engine.run("AAPL", modules=["fund_flow"])
    assert result["resolved_modules"] == ["fund_flow"]
    assert set(fd.calls) == {"insiders"}
    assert result["modules"]["valuation"]["status"] == "SKIPPED"
    assert "fundamental" in resolve_modules(["risk"])
    assert "catalyst" in resolve_modules(["risk"])


def test_module_cache_hit_expiry_and_force_refresh(tmp_path: Path):
    fd = FakeFD()
    cache = ResearchCache(tmp_path / "research.db")
    engine = ResearchEngine(fd=fd, prices=FakePrices(), cache=cache, services=fake_services())
    first = engine.run("NVDA", modules=["fundamental"])
    second = engine.run("NVDA", modules=["fundamental"])
    assert second["modules"]["fundamental"]["cache_hit"] is True
    assert fd.calls["metrics"] == 1
    engine.run("NVDA", modules=["fundamental"], force_refresh=True)
    assert fd.calls["metrics"] == 2
    with sqlite3.connect(cache.path) as conn:
        conn.execute("UPDATE research_module_cache SET expires_at=?", (time.time() - 1,))
    engine.run("NVDA", modules=["fundamental"])
    assert fd.calls["metrics"] == 3
    assert first["engine_version"] == ENGINE_VERSION


def test_run_and_snapshots_survive_store_reopen(tmp_path: Path):
    path = tmp_path / "research.db"
    store = ResearchStore(path)
    store.create_run("run-1", "AAPL", ["fundamental"], "MODULE")
    store.update_run("run-1", "RUNNING")
    assert ResearchStore(path).get_run("run-1")["status"] == "RUNNING"
    store.recover_incomplete_runs()
    assert ResearchStore(path).get_run("run-1")["status"] == "FAILED"
    for index in range(2):
        result = {"run_id": f"snapshot-{index}", "ticker": "AAPL", "status": "COMPLETED", "generated_at": f"2026-09-0{index + 1}T00:00:00+00:00", "engine_version": ENGINE_VERSION, "modules": {}, "scores": {"fundamental": index}}
        store.save_snapshot(result["run_id"], result)
    history = ResearchStore(path).history("AAPL")
    assert len(history) == 2
    assert {row["run_id"] for row in history} == {"snapshot-0", "snapshot-1"}


class EmptyNewsFD(FakeFD):
    pass


class DiagnosticProvider:
    def __init__(self, results=None, error=None):
        self.results = results or []
        self.last_diagnostics = {"error": error, "warning": "zero" if not results and not error else None}

    def search(self, query, **kwargs):
        return self.results


def test_news_fallback_and_zero_diagnostics(tmp_path: Path):
    store = ResearchStore(tmp_path / "research.db")
    provider = DiagnosticProvider([{"title": "Apple launches product", "url": "https://example.com/a", "content": "AAPL"}])
    result = NewsDataService(store, provider).collect("AAPL", EmptyNewsFD())
    assert len(result["items"]) == 1
    assert result["items"][0]["provider"] == "Tavily"
    zero = NewsDataService(ResearchStore(tmp_path / "empty.db"), DiagnosticProvider(error="TimeoutError: timed out")).collect("AAPL", EmptyNewsFD())
    assert zero["items"] == []
    assert zero["diagnostics"]["provider_attempts"][-1]["error"].startswith("TimeoutError")


def test_technical_service_is_chart_implementation(tmp_path: Path):
    prices = FakePrices().get_prices("AAPL", "", "")
    analysis = TechnicalAnalysisService().analyze(prices, "1d")
    assert analysis["SMA20"] is not None
    assert analysis["ATR14"] is not None
    assert analysis["timeframe"] == "1d"


def test_relationship_persistence(tmp_path: Path):
    store = ResearchStore(tmp_path / "research.db")
    relation_id = store.upsert_relationship({"source_ticker": "AAPL", "target_ticker": "TSM", "target_name": "TSMC", "relationship_type": "SUPPLIER", "status": "VERIFIED", "confidence": .91, "description": "chip supplier"}, [{"provider": "Tavily", "url": "https://example.com/evidence"}])
    reopened = ResearchStore(tmp_path / "research.db")
    relation = reopened.relationship(relation_id)
    assert relation["status"] == "VERIFIED"
    assert relation["sources"][0]["provider"] == "Tavily"
