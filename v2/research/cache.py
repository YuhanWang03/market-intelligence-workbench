"""Research cache policy and backward-compatible cache facade."""

from __future__ import annotations

from pathlib import Path

from v2.research.store import ResearchStore

CACHE_POLICY: dict[str, int] = {
    "company_profile": 24 * 3600, "financial_statements": 24 * 3600,
    "financial_metrics": 12 * 3600, "sec": 6 * 3600,
    "earnings": 6 * 3600, "expectations": 2 * 3600,
    "news": 60 * 60, "news_empty": 10 * 60, "catalyst": 60 * 60,
    "institutional": 24 * 3600, "fund_flow": 30 * 60,
    "technical": 15 * 60, "macro": 30 * 60,
    "supply_chain": 24 * 3600, "fundamental": 24 * 3600,
    "valuation": 12 * 3600, "risk": 30 * 60, "aggregate": 30 * 60,
}


class ResearchCache:
    """Compatibility adapter; new execution uses module-level cache methods."""

    def __init__(self, path: Path | None = None, ttl_hours: float = 6.0) -> None:
        self.store = ResearchStore(path)
        self.path = self.store.path
        self.ttl_seconds = int(ttl_hours * 3600)

    def get(self, ticker: str) -> dict | None:
        payload = self.store.latest(ticker)
        if payload:
            payload["from_cache"] = True
        return payload

    def put(self, ticker: str, payload: dict) -> None:
        run_id = payload.get("run_id") or f"legacy-{ticker}-{payload.get('generated_at', '')}"
        self.store.save_snapshot(run_id, payload)

    def get_module(self, ticker: str, module: str, period: str = "default") -> dict | None:
        return self.store.get_module_cache(ticker, module, period)

    def put_module(self, ticker: str, module: str, payload: dict, period: str = "default") -> None:
        self.store.put_module_cache(ticker, module, payload, CACHE_POLICY.get(module, self.ttl_seconds), period)
