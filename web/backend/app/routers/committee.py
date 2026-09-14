"""Lab · 投资人委员会 — run the thirteen simulated investors from the workbench.

``POST /api/lab/committee`` takes a ticker source (explicit list, current
holdings, watchlist, or the output of the Lab screener), builds one
fundamentals snapshot per ticker, lets every selected persona vote, and
returns the matrix plus a confidence-weighted consensus.  Every run is
persisted (``v2/personas/store.py``) so the run log survives restarts and
forward returns can be back-filled later.

No LLM is involved.  Holdings get a transparent action label
(增持候选 / 持有 / 减持候选) derived from consensus, agreement and the
position's current weight — the rule is spelled out in :func:`_holding_action`.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.auth import require_owner
from app.fd_pricing import cost as fd_cost, prices as fd_prices
from app.routers import workspace
from app.sources import MAX_TICKERS, holdings as _holdings, normalize_tickers, watchlist as _watchlist
from v2.personas.committee import CommitteeResult, TickerVerdict, run_committee
from v2.personas.forward import backfill_forward_returns
from v2.personas.models import PersonaSignal
from v2.personas.data import adapt_client
from v2.personas.registry import PERSONAS, get_persona
from v2.personas.store import DEFAULT_PATH, PersonaStore

logger = logging.getLogger("web.committee")

router = APIRouter(prefix="/api/lab/committee", tags=["lab"], dependencies=[Depends(require_owner)])

_STORE: PersonaStore | None = None

Source = Literal["tickers", "holdings", "watchlist", "screening"]


def _store() -> PersonaStore:
    global _STORE
    if _STORE is None:
        _STORE = PersonaStore(os.environ.get("WEB_PERSONAS_DB") or DEFAULT_PATH)
    return _STORE


class CommitteeInput(BaseModel):
    source: Source = "tickers"
    tickers: list[str] = Field(default_factory=list, max_length=MAX_TICKERS)
    personas: list[str] | None = None
    as_of: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    top_n: int | None = Field(default=None, ge=1, le=MAX_TICKERS)  # only meaningful for source=screening; None = rank everything
    use_cache: bool = True
    max_weight: float = Field(default=0.15, gt=0, le=1.0)
    #: 省流模式: skip the news and insider-trade fetches (two paid calls per ticker
    #: that only feed a few personas' small sentiment sub-scores)
    lean: bool = True
    screening: workspace.ScreeningInput | None = None


# --------------------------------------------------------------------------- inputs

def _resolve_personas(keys: list[str] | None) -> list[str]:
    if not keys:
        return list(PERSONAS)
    unknown = [k for k in keys if k not in PERSONAS]
    if unknown:
        raise ValueError(f"unknown persona: {', '.join(unknown)}")
    return list(dict.fromkeys(keys))


def _screened(body: workspace.ScreeningInput | None) -> tuple[list[str], dict[str, Any]]:
    result = workspace._run_screening(body or workspace.ScreeningInput())
    candidates = result.get("candidates") or []
    tickers = [str(c.get("ticker")).upper() for c in candidates if c.get("ticker")]
    summary = {
        "universe_size": result.get("universe_size"),
        "n_candidates": len(candidates),
        "date": result.get("date"),
        "candidates": [{"ticker": c.get("ticker"), "price": c.get("price"), "market_cap": c.get("market_cap"), "revenue_growth": c.get("revenue_growth"), "gross_margin": c.get("gross_margin")} for c in candidates],
    }
    return tickers, summary


@contextmanager
def _data_client() -> Iterator[Any]:
    """Production FDClient when available (VPS), else the stdlib HTTP client."""
    try:
        from v2.data import CachedFDClient  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001 — repo checkout has no v2/data
        yield adapt_client(None)
        return
    with CachedFDClient() as raw:
        yield adapt_client(raw)


# ---------------------------------------------------------------------------- rules

def _holding_action(v: TickerVerdict, weight: float | None, max_weight: float) -> tuple[str, str]:
    """Transparent rule: (label, why)."""
    if v.voters == 0:
        return "数据不足", "所有投资人弃权"
    if v.consensus >= 0.2 and v.agreement >= 0.5:
        if weight is not None and weight >= max_weight:
            return "持有", f"共识看多 {v.consensus:+.2f}，但权重 {weight:.1%} 已达上限 {max_weight:.0%}"
        return "增持候选", f"共识看多 {v.consensus:+.2f}，{v.bullish}/{v.voters} 位看多"
    if v.consensus <= -0.2:
        return "减持候选", f"共识看空 {v.consensus:+.2f}，{v.bearish}/{v.voters} 位看空"
    return "持有", f"意见分歧（{v.bullish}▲ {v.bearish}▼ {v.neutral}·），共识 {v.consensus:+.2f}"


def _persona_meta(keys: list[str]) -> list[dict[str, Any]]:
    out = []
    for key in keys:
        p = get_persona(key)
        out.append({"key": key, "name": p.name, "name_zh": p.name_zh, "style": p.style, "period": p.period, "lookback": p.lookback, "needs": sorted(p.needs)})
    return out


# ------------------------------------------------------------------------------ run

def _run(body: CommitteeInput) -> dict[str, Any]:
    keys = _resolve_personas(body.personas)
    positions: dict[str, dict[str, Any]] = {}
    screening: dict[str, Any] | None = None

    if body.source == "tickers":
        tickers = normalize_tickers(body.tickers, limit=MAX_TICKERS)
    elif body.source == "holdings":
        tickers, positions = _holdings()
        if not tickers:
            raise ValueError("no long positions in the account")
    elif body.source == "watchlist":
        tickers = _watchlist()
        if not tickers:
            raise ValueError("watchlist is empty")
    else:
        tickers, screening = _screened(body.screening)
        if not tickers:
            raise ValueError("screening returned no candidates")
    tickers = tickers[:MAX_TICKERS]

    store = _store()
    as_of = body.as_of
    cached = {}
    if body.use_cache:
        from datetime import date

        key_date = as_of or date.today().isoformat()
        for t in tickers:
            snap = store.cached_snapshot(t, key_date)
            if snap is not None:
                cached[t] = snap

    with _data_client() as client:
        result: CommitteeResult = run_committee(tickers, client, personas=keys, as_of=as_of, snapshots=cached, max_workers=4,
                                                exclude_needs=("news", "insiders") if body.lean else ())

    for t, snap in result.snapshots.items():
        if t not in cached:
            store.save_snapshot(snap)

    payload = result.to_dict()
    payload["kind"] = "committee"
    payload["source"] = body.source
    payload["personas_meta"] = _persona_meta(keys)
    payload["cache_hits"] = sorted(cached)
    payload["lean"] = body.lean
    requests: dict[str, int] = {}
    for t, snap in result.snapshots.items():
        if t in cached:
            continue
        for endpoint, n in snap.requests.items():
            requests[endpoint] = requests.get(endpoint, 0) + n
    payload["fd_requests"] = requests
    payload["fd_cost_usd"] = fd_cost(requests)
    gaps: dict[str, list[str]] = {}
    for t, snap in result.snapshots.items():
        for g in snap.gaps:
            gaps.setdefault(g, []).append(t)
    payload["data_gaps"] = [{"gap": g, "tickers": sorted(ts)} for g, ts in sorted(gaps.items(), key=lambda kv: -len(kv[1]))]
    for v_dict, v in zip(payload["verdicts"], result.verdicts):
        snap = result.snapshots.get(v.ticker)
        if snap is not None and snap.prices:
            last = snap.prices[-1]
            try:
                v_dict["price"] = float(last.close) if last.close is not None else None
            except (TypeError, ValueError):
                v_dict["price"] = None
        pos = positions.get(v.ticker)
        if pos:
            label, why = _holding_action(v, pos.get("weight"), body.max_weight)
            v_dict["position"] = pos
            v_dict["action"] = label
            v_dict["action_reason"] = why
            v_dict["price"] = pos.get("current_price")
    if screening is not None:
        payload["screening"] = screening
    payload["top"] = [
        {"rank": v.rank, "ticker": v.ticker, "stance": v.stance, "consensus": round(v.consensus, 4), "bullish": v.bullish, "bearish": v.bearish, "neutral": v.neutral, "agreement": round(v.agreement, 4)}
        for v in result.top(body.top_n or max(1, len(result.verdicts)))
    ]
    payload["run_id"] = store.save_run(payload, source=body.source)
    workspace._remember_run("committee", payload, body.model_dump())
    return payload


@router.post("")
async def run(body: CommitteeInput) -> dict:
    return await workspace._lab_call(_run, body)


@router.get("/personas")
async def personas() -> dict:
    return {"kind": "personas", "items": _persona_meta(list(PERSONAS))}


@router.get("/runs")
async def runs(limit: int = 50) -> dict:
    return {"kind": "committee_runs", "items": _store().list_runs(limit=max(1, min(limit, 200)))}


@router.get("/runs/{run_id}")
async def run_detail(run_id: str) -> dict:
    payload = _store().get_run(run_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="committee run not found")
    return payload


@router.get("/scoreboard")
async def scoreboard() -> dict:
    """Per-persona hit rate once forward returns have been back-filled, plus vote counts."""
    store = _store()
    items = store.persona_scoreboard()
    names = {m["key"]: m for m in _persona_meta(list(PERSONAS))}
    for row in items:
        meta = names.get(row["persona"], {})
        row["name_zh"] = meta.get("name_zh", row["persona"])
        row["name"] = meta.get("name", row["persona"])
    return {"kind": "scoreboard", "items": items, "counts": store.signal_counts(), "baseline": store.scoreboard_baseline()}


# ------------------------------------------------------------------- narration

class NarrateInput(BaseModel):
    run_id: str
    ticker: str
    persona: str
    language: Literal["zh", "en"] = "zh"


def _narrate_llm() -> Any:
    """Seam for tests; production uses the agent loop's configured provider."""
    from v2.agent_common.llm import build_llm

    return build_llm()


def _narrate(body: NarrateInput) -> dict[str, Any]:
    from v2.personas.narrate import narrate

    store = _store()
    payload = store.get_run(body.run_id)
    if payload is None:
        raise LookupError("committee run not found")
    ticker = body.ticker.upper()
    found = None
    for v in payload.get("verdicts") or []:
        if v.get("ticker") == ticker:
            for sig in v.get("signals") or []:
                if sig.get("persona") == body.persona:
                    found = sig
    if found is None:
        raise LookupError("signal not found in this run")
    signal = PersonaSignal.from_dict(found)
    if signal.abstained:
        raise ValueError("this persona abstained; nothing to narrate")
    # A fresh request means a fresh narration: forget any stored text so a
    # discarded reply surfaces as an error instead of echoing the old one.
    signal.narrative = None
    signal.narrative_grounded = None
    narrate(signal, llm=_narrate_llm(), language=body.language)
    if not signal.narrative:
        raise RuntimeError("the model returned nothing usable; the rule-based reasoning stands")
    store.update_signal_narrative(body.run_id, ticker, body.persona, signal.narrative, signal.narrative_grounded)
    return {
        "kind": "narrative", "run_id": body.run_id, "ticker": ticker, "persona": body.persona,
        "signal": signal.signal, "confidence": signal.confidence,
        "narrative": signal.narrative, "narrative_grounded": signal.narrative_grounded,
    }


@router.post("/narrate")
async def narrate_signal(body: NarrateInput) -> dict:
    """LLM explanation for one cell of a stored run. Never changes the verdict."""
    try:
        return await workspace._lab_call(_narrate, body)
    except HTTPException as exc:
        if isinstance(exc.__cause__, LookupError):
            raise HTTPException(status_code=404, detail=str(exc.__cause__)) from exc
        raise


# -------------------------------------------------------------------- backfill

class BackfillInput(BaseModel):
    columns: list[Literal["fwd_1m", "fwd_3m"]] = Field(default_factory=lambda: ["fwd_1m", "fwd_3m"])


def _backfill(body: BackfillInput) -> dict[str, Any]:
    report = backfill_forward_returns(_store(), columns=tuple(body.columns))
    return {"kind": "backfill", **report.to_dict(), "scoreboard": _store().persona_scoreboard()}


@router.post("/backfill")
async def backfill(body: BackfillInput | None = None) -> dict:
    """Run the forward-return backfill now (the scheduler also does this nightly)."""
    return await workspace._lab_call(_backfill, body or BackfillInput())


@router.get("/pricing")
async def pricing() -> dict:
    """Per-request prices used for estimates (override with FD_PRICES) and the per-ticker committee recipe."""
    table = fd_prices()
    per_ticker_full = {"financial_metrics": 2, "line_items": 2, "prices": 1, "insider_trades": 1, "news": 1}
    per_ticker_lean = {"financial_metrics": 2, "line_items": 2, "prices": 1}
    return {
        "kind": "pricing",
        "prices_usd": table,
        "committee_per_ticker": {"full": fd_cost(per_ticker_full), "lean": fd_cost(per_ticker_lean), "requests_full": per_ticker_full, "requests_lean": per_ticker_lean},
        "note": "estimates only; cached snapshots and the 24h metrics cache cost nothing",
    }
