"""Reusable data services shared by Research Engine and compatibility views.

These services deliberately expose structured Python data.  They do not know
about slash commands, chat intents, HTML cards, or the web transport layer.
"""

from __future__ import annotations

import time

import sqlite3
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from v2.research.cache import CACHE_POLICY
from v2.research.consensus import collect_consensus
from v2.research.relationship_evidence import label_sources
from v2.research.depth import parse_sec_filings, sanitize_error
from v2.research.store import ResearchStore


def _value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _model_dict(value: Any) -> dict:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return dict(value) if isinstance(value, dict) else {}


@dataclass
class EarningsSecDataService:
    """Normalize earnings expectations and SEC filing metadata.

    Financial Datasets filing links are reused first.  SEC EDGAR is queried
    directly through the existing low-level client for recent corporate
    filings; no bot responder or text formatter is involved.
    """

    sec_days: int = 365

    def collect(self, ticker: str, earnings: list[Any], upcoming: Any) -> dict:
        filings: list[dict] = []
        filing_objects: dict[str, list[Any]] = {"10-K": [], "10-Q": [], "8-K": []}
        seen: set[tuple[str, str, str]] = set()
        for row in earnings:
            item = {
                "form": _value(row, "source_type"),
                "filing_date": _value(row, "filing_date"),
                "report_period": _value(row, "report_period"),
                "accession_number": _value(row, "accession_number"),
                "url": _value(row, "filing_url"),
                "source": "Financial Datasets",
            }
            key = (str(item["form"] or ""), str(item["filing_date"] or ""), str(item["url"] or ""))
            if item["url"] and key not in seen:
                seen.add(key)
                filings.append(item)

        warnings: list[str] = []
        try:
            from v2.sec.client import get_recent_filings

            today = date.today()
            since = (today - timedelta(days=self.sec_days)).isoformat()
            for form in ("10-K", "10-Q", "8-K"):
                recent = get_recent_filings(ticker, form, since, today.isoformat())[:12]
                filing_objects[form] = recent
                for filing in recent:
                    item = {
                        "form": str(_value(filing, "form", form) or form),
                        "filing_date": str(_value(filing, "filing_date", "") or ""),
                        "report_period": str(_value(filing, "report_period", "") or ""),
                        "accession_number": str(_value(filing, "accession_number", _value(filing, "accession_no", "")) or ""),
                        "url": str(_value(filing, "homepage_url", _value(filing, "filing_url", "")) or ""),
                        "source": "SEC EDGAR",
                    }
                    key = (item["form"], item["filing_date"], item["url"])
                    if key not in seen:
                        seen.add(key)
                        filings.append(item)
        except Exception as exc:
            warnings.append(f"SEC EDGAR: {type(exc).__name__}")

        sec_depth: dict = {"findings": [], "risk_factor_changes": [], "guidance": [], "company_profile": {}, "parsed_forms": []}
        if any(filing_objects.values()):
            try:
                sec_depth = parse_sec_filings(ticker, filing_objects)
            except Exception as exc:
                # Never include the exception string: SDK/network messages can
                # contain request metadata. Provider keys must not reach API/UI.
                warnings.append(f"SEC parsing: {type(exc).__name__}")

        return {
            "expectations": {
                "consensus": collect_consensus(ticker),
                "upcoming_earnings": _model_dict(upcoming) or None,
                "revision_history": [],
            },
            "filings": sorted(filings, key=lambda item: item.get("filing_date") or "", reverse=True),
            "sec_depth": sec_depth,
            "warnings": warnings,
        }


class InstitutionalDataService:
    """Read normalized 13F and ETF exposure from the existing SQLite stores."""

    def collect(self, ticker: str, insiders: list[Any]) -> dict:
        holdings, changes, warnings = self._institutional_holdings(ticker)
        etf_exposure = self._etf_exposure(ticker, warnings)
        insider_rows = []
        for trade in insiders:
            insider_rows.append({
                "name": _value(trade, "name"),
                "title": _value(trade, "title"),
                "filing_date": _value(trade, "filing_date"),
                "transaction_date": _value(trade, "transaction_date"),
                "transaction_type": _value(trade, "transaction_type"),
                "shares": _value(trade, "transaction_shares"),
                "price": _value(trade, "transaction_price_per_share"),
                "value": _value(trade, "transaction_value"),
                "shares_before": _value(trade, "shares_owned_before_transaction"),
                "shares_after": _value(trade, "shares_owned_after_transaction"),
            })
        return {
            "institutional_holdings": holdings,
            "13f_changes": changes,
            "insider_transactions": insider_rows,
            "ownership_changes": changes,
            "etf_exposure": etf_exposure,
            "warnings": warnings,
        }

    @staticmethod
    def _institutional_holdings(ticker: str) -> tuple[list[dict], list[dict], list[str]]:
        holdings: list[dict] = []
        changes: list[dict] = []
        warnings: list[str] = []
        try:
            from v2.institutional.managers import MANAGERS
            from v2.institutional.tracker import get_db

            with get_db() as conn:
                for cik, manager in MANAGERS:
                    filings = conn.execute(
                        """SELECT accession, quarter, period_of_report, portfolio_value
                           FROM filings WHERE cik=? ORDER BY period_of_report DESC LIMIT 2""",
                        (cik,),
                    ).fetchall()
                    if not filings:
                        continue
                    current = filings[0]
                    position = conn.execute(
                        """SELECT shares, market_value, issuer_name FROM positions
                           WHERE accession=? AND ticker=?""",
                        (current["accession"], ticker),
                    ).fetchone()
                    if position is None:
                        continue
                    portfolio_value = float(current["portfolio_value"] or 0)
                    holdings.append({
                        "manager": manager,
                        "quarter": current["quarter"],
                        "period_of_report": current["period_of_report"],
                        "issuer_name": position["issuer_name"],
                        "shares": position["shares"],
                        "market_value": position["market_value"],
                        "portfolio_weight": (position["market_value"] / portfolio_value) if portfolio_value else None,
                    })
                    if len(filings) > 1:
                        previous = conn.execute(
                            """SELECT shares, market_value FROM positions
                               WHERE accession=? AND ticker=?""",
                            (filings[1]["accession"], ticker),
                        ).fetchone()
                        prev_shares = float(previous["shares"] or 0) if previous else 0.0
                        prev_value = float(previous["market_value"] or 0) if previous else 0.0
                        changes.append({
                            "manager": manager,
                            "quarter": current["quarter"],
                            "shares_change": float(position["shares"] or 0) - prev_shares,
                            "market_value_change": float(position["market_value"] or 0) - prev_value,
                            "change_type": "new" if previous is None else "increase" if float(position["shares"] or 0) > prev_shares else "decrease" if float(position["shares"] or 0) < prev_shares else "unchanged",
                        })
        except Exception as exc:
            warnings.append(f"13F database: {type(exc).__name__}")
        holdings.sort(key=lambda item: float(item.get("market_value") or 0), reverse=True)
        return holdings, changes, warnings

    @staticmethod
    def _etf_exposure(ticker: str, warnings: list[str]) -> list[dict]:
        try:
            from v2.etf.tracker import _DB_PATH

            if not Path(_DB_PATH).exists():
                return []
            conn = sqlite3.connect(str(_DB_PATH), timeout=10.0)
            conn.row_factory = sqlite3.Row
            try:
                rows = conn.execute(
                    """SELECT s.etf, s.date, s.company, s.shares, s.market_value, s.weight_pct
                       FROM snapshots s
                       JOIN (SELECT etf, MAX(date) AS latest_date FROM snapshots GROUP BY etf) latest
                         ON latest.etf=s.etf AND latest.latest_date=s.date
                       WHERE s.ticker=? ORDER BY s.weight_pct DESC""",
                    (ticker,),
                ).fetchall()
                return [dict(row) for row in rows]
            finally:
                conn.close()
        except Exception as exc:
            warnings.append(f"ETF database: {type(exc).__name__}")
            return []


class MacroDataService:
    """Expose the existing FRED/Yahoo macro pipeline as structured data."""

    #: The FRED plus Yahoo snapshot is minutes when FRED is slow and does not change within ten; one per process per window.
    _SNAPSHOT_TTL_SECONDS = 600.0
    _snapshot_cache: dict[str, tuple[float, dict, list[str]]] = {}

    def collect(self, as_of: date | None = None) -> dict:
        as_of = as_of or date.today()
        warnings: list[str] = []
        snapshot = None
        cached = self._snapshot_cache.get(as_of.isoformat())
        if cached is not None and time.monotonic() - cached[0] < self._SNAPSHOT_TTL_SECONDS:
            snapshot, warnings = dict(cached[1]), list(cached[2])
        else:
            try:
                from v2.macro import build_macro_snapshot

                snapshot = asdict(build_macro_snapshot(as_of.isoformat()))
                warnings.extend(snapshot.pop("warnings", []) or [])
                self._snapshot_cache[as_of.isoformat()] = (time.monotonic(), dict(snapshot), list(warnings))
            except Exception as exc:
                warnings.append(f"macro snapshot: {type(exc).__name__}")
        events: list[dict] = []
        try:
            from v2.macro.release_calendar import get_releases_in_window

            window = get_releases_in_window(as_of.isoformat(), (as_of + timedelta(days=45)).isoformat())
            for event_date, entries in sorted(window.items()):
                for release_type, label, source in entries:
                    events.append({"event_date": event_date, "release_type": release_type, "title": label, "source": source})
        except Exception as exc:
            warnings.append(f"macro calendar: {type(exc).__name__}")
        return {"snapshot": snapshot, "upcoming_events": events, "warnings": warnings}


class SupplyChainDataService:
    """Run the validated lateral-discovery pipeline without chat coupling."""

    store: ResearchStore | None = None
    _TYPE_MAP = {"supplier": "SUPPLIER", "customer": "CUSTOMER", "smaller_peer": "COMPETITOR", "beneficiary": "BENEFICIARY", "partner": "PARTNER", "platform": "PLATFORM", "substitute": "SUBSTITUTE"}
    _CATEGORY_MAP = {value: key for key, value in _TYPE_MAP.items()}

    def collect(self, ticker: str, fd_client: Any) -> dict:
        if self.store:
            existing = self.store.relationships(ticker)
            cutoff = datetime.now(timezone.utc) - timedelta(seconds=CACHE_POLICY["supply_chain"])
            reusable = []
            refreshable = []
            for row in existing:
                try:
                    verified_at = datetime.fromisoformat(row.get("last_verified_at") or "")
                except ValueError:
                    verified_at = datetime.min.replace(tzinfo=timezone.utc)
                if row.get("status") in ("CO_MENTION", "EVIDENCE_FOUND") and datetime.fromisoformat(row['updated_at']) >= cutoff:
                    reusable.append(row)
                else:
                    refreshable.append(row)
            # Existing graphs are updated incrementally: fresh/high-confidence
            # edges are reused untouched; only stale/low-confidence edges are
            # sent back through relation verification. Full LLM discovery is
            # reserved for tickers that have no graph yet.
            tavily_calls = 0
            if existing and refreshable:
                try:
                    from v2.lateral.models import Label, Neighbor
                    from v2.lateral.verify import verify_relation
                    for row in refreshable:
                        category = self._CATEGORY_MAP.get(row["relationship_type"], row["relationship_type"].lower())
                        neighbor = Neighbor(ticker=row["target_ticker"], labels=[Label(seed=ticker, category=category, reason=row.get("description") or "relationship revalidation")], exists=True)
                        tavily_calls += verify_relation(neighbor)
                        label = neighbor.labels[0]
                        self.store.upsert_relationship({**row, "status": label.evidence_status, "confidence": 0}, label_sources(label))
                except Exception:
                    pass
                existing = self.store.relationships(ticker)
            if existing:
                return {"date": date.today().isoformat(), "seeds": [ticker], "neighbors": [{"ticker": row["target_ticker"], "name": row.get("target_name"), "exists": True, "relation_verified": row.get("status") == "VERIFIED", "relation_checked": True, "relation_evidence_url": next((s["url"] for s in row.get("sources", []) if s.get("url")), None), "labels": [{"seed": ticker, "category": self._CATEGORY_MAP.get(row["relationship_type"], row["relationship_type"].lower()), "reason": row.get("description") or "persisted relationship"}]} for row in existing], "llm_tokens": 0, "api_calls": 0, "tavily_calls": tavily_calls, "from_relationship_store": True, "reused_relationships": len(reusable), "revalidated_relationships": len(refreshable)}
        from v2.lateral import LATERAL_FILTERS, run_lateral_expansion
        from v2.screening import TECH_30

        result = run_lateral_expansion(
            seeds=[ticker],
            universe=set(TECH_30),
            fd_client=fd_client,
            filter_config=LATERAL_FILTERS,
        )
        payload = result.model_dump(mode="json")
        store = self.store
        if store:
            for neighbor in payload.get("neighbors", []):
                if not neighbor.get("exists"):
                    continue
                for label in neighbor.get("labels", []):
                    if str(label.get("seed", "")).upper() != ticker.upper():
                        continue
                    status = label.get("evidence_status", "UNCHECKED")
                    relation = {
                        "source_ticker": ticker, "target_ticker": neighbor.get("ticker"),
                        "target_name": neighbor.get("name"), "relationship_type": self._TYPE_MAP.get(label.get("category", ""), "PARTNER"),
                        "status": status, "confidence": 0,
                        "description": label.get("reason"),
                    }
                    from v2.lateral.models import Label
                    sources = label_sources(Label(**label))
                    store.upsert_relationship(relation, sources)
        return payload


class NewsDataService:
    """Normalized FD news with observable Tavily fallback and short empty TTL."""

    def __init__(self, store: ResearchStore | None = None, provider: Any = None) -> None:
        self.store = store
        self.provider = provider

    @staticmethod
    def _normalize(item: Any, provider: str) -> dict | None:
        title = _value(item, "title") or _value(item, "headline")
        url = _value(item, "url") or _value(item, "article_url")
        if not title:
            return None
        return {"title": str(title), "date": _value(item, "date") or _value(item, "published_at"), "url": str(url or ""), "content": str(_value(item, "content", _value(item, "summary", "")) or ""), "provider": provider}

    def collect(self, ticker: str, fd_client: Any, *, force_refresh: bool = False) -> dict:
        if self.store and not force_refresh:
            cached = self.store.get_data_cache(ticker, "news", "90d")
            if cached:
                rows, diagnostics = cached
                return {"items": rows, "diagnostics": {**diagnostics, "cache_hit": True}}
        attempts: list[dict] = []
        rows: list[dict] = []
        today = date.today()
        try:
            # Financial Datasets rejects/behaves inconsistently with oversized
            # limits. Ten is its stable news page size; dedup happens below.
            raw = fd_client.get_news(ticker, today.isoformat(), (today - timedelta(days=90)).isoformat(), limit=10) or []
            normalized = [row for item in raw if (row := self._normalize(item, "Financial Datasets"))]
            attempts.append({"provider": "Financial Datasets", "query": ticker, "raw_count": len(raw), "filtered_count": len(normalized), "dedup_count": 0, "error": None, "warning": "provider returned zero results" if not raw else None})
            rows.extend(normalized)
        except Exception as exc:
            attempts.append({"provider": "Financial Datasets", "query": ticker, "raw_count": 0, "filtered_count": 0, "dedup_count": 0,
                             "error": f"{type(exc).__name__}: {sanitize_error(exc)}",
                             "error_type": str(getattr(exc, "error_type", "PROVIDER_ERROR")), "warning": None})
        if not rows:
            provider = self.provider
            if provider is None:
                from v2.data.news_provider import default_news_provider
                provider = default_news_provider()
            query = f"{ticker} company stock latest news"
            raw = provider.search(query, days=30, max_results=10)
            normalized = [row for item in raw if (row := self._normalize(item, "Tavily"))]
            diag = dict(getattr(provider, "last_diagnostics", {}) or {})
            attempts.append({"provider": "Tavily", "query": query, "raw_count": len(raw), "filtered_count": len(normalized), "dedup_count": 0, "error": diag.get("error"), "warning": diag.get("warning")})
            rows.extend(normalized)
        unique: list[dict] = []
        seen: set[str] = set()
        for row in rows:
            key = (row.get("url") or row["title"]).strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique.append(row)
        for attempt in attempts:
            attempt["dedup_count"] = len(unique)
        diagnostics = {"query": attempts[-1]["query"] if attempts else ticker, "provider_attempts": attempts, "raw_count": sum(x["raw_count"] for x in attempts), "filtered_count": sum(x["filtered_count"] for x in attempts), "dedup_count": len(unique), "cache_hit": False}
        if self.store:
            ttl = CACHE_POLICY["news"] if unique else CACHE_POLICY["news_empty"]
            self.store.put_data_cache(ticker, "news", unique, ttl, diagnostics, "90d")
        return {"items": unique, "diagnostics": diagnostics}


class TechnicalAnalysisService:
    """Single technical-analysis entry point shared with the K-line module."""

    def analyze(self, prices: list[Any], timeframe: str = "1d") -> dict:
        bars = [{"timestamp": _value(p, "time"), "open": _value(p, "open"), "high": _value(p, "high"), "low": _value(p, "low"), "close": _value(p, "close"), "volume": _value(p, "volume"), "timeframe": timeframe, "session": "REGULAR", "source": "Yahoo Finance", "isFinal": True} for p in prices]
        try:
            # The chart and research path call the exact same implementation.
            from app.market_analysis import build_technical_analysis, enrich_bars
            enriched = enrich_bars(bars, timeframe)
            quote = {"symbol": "", "price": enriched[-1]["close"], "regularClose": enriched[-1]["close"], "dailyChangePct": None, "dayReturn": None, "previousClose": enriched[-2]["close"] if len(enriched) > 1 else enriched[-1]["close"], "session": "CLOSED", "timestamp": enriched[-1]["timestamp"], "source": "Yahoo Finance", "isDelayed": True}
            return build_technical_analysis(enriched, timeframe, quote, "SPLIT_ADJUSTED")
        except ImportError:
            closes = [float(row["close"]) for row in bars if row.get("close") is not None]
            sma20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else None
            sma50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else None
            sma200 = sum(closes[-200:]) / 200 if len(closes) >= 200 else None
            true_ranges = []
            for index, row in enumerate(bars):
                previous = float(bars[index - 1]["close"]) if index else float(row["close"])
                true_ranges.append(max(float(row["high"]) - float(row["low"]), abs(float(row["high"]) - previous), abs(float(row["low"]) - previous)))
            atr14 = sum(true_ranges[-14:]) / 14 if len(true_ranges) >= 14 else None
            return {"timeframe": timeframe, "currentPrice": closes[-1] if closes else None, "sma20": sma20, "sma50": sma50, "sma200": sma200, "SMA20": sma20, "SMA50": sma50, "SMA200": sma200, "atr14": atr14, "ATR14": atr14, "supports": [], "resistances": []}


@dataclass
class ResearchServices:
    earnings_sec: EarningsSecDataService
    institutional: InstitutionalDataService
    macro: MacroDataService
    supply_chain: SupplyChainDataService
    news: NewsDataService | None = None
    technical: TechnicalAnalysisService | None = None

    @classmethod
    def defaults(cls) -> "ResearchServices":
        return cls(
            earnings_sec=EarningsSecDataService(),
            institutional=InstitutionalDataService(),
            macro=MacroDataService(),
            supply_chain=SupplyChainDataService(),
            news=NewsDataService(),
            technical=TechnicalAnalysisService(),
        )
