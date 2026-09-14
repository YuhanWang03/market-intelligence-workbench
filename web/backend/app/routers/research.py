"""Persistent HTTP surface for the modular stock research engine."""

from __future__ import annotations

import asyncio
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app.auth import require_owner
from v2.research import ResearchEngine, normalize_ticker
from v2.research.engine import MODULES, resolve_modules
from v2.research.depth import sanitize_error
from v2.research.provider_health import ProviderHealthService
from v2.research.store import ResearchStore
from v2.research.localization import localize_result
from v2.research.sec_links import enrich_sec_links
from v2.research.expectations import prepare_expectations
from v2.research.relationship_evidence import present_relationships, label_sources


def present_result(result):
    return enrich_sec_links(localize_result(prepare_expectations(present_relationships(result, _store()))))

router = APIRouter(prefix="/api/research", tags=["research"], dependencies=[Depends(require_owner)])
_RUNNING_TASKS: set[asyncio.Task[None]] = set()
_STORE: ResearchStore | None = None
# Deprecated compatibility handles. They are intentionally not authoritative;
# persisted SQLite state replaced them in Phase 3A.
_JOBS: dict[str, dict] = {}
_ACTIVE_BY_TICKER: dict[str, str] = {}
_PROVIDER_HEALTH = ProviderHealthService()


def _store() -> ResearchStore:
    global _STORE
    if _STORE is None:
        _STORE = ResearchStore()
        _STORE.recover_incomplete_runs()
    return _STORE


class ResearchRunInput(BaseModel):
    ticker: str
    modules: list[str] | None = None
    force_refresh: bool = False
    refresh: bool = False


class RetryInput(BaseModel):
    modules: list[str] = Field(min_length=1)


class PeerPreferenceInput(BaseModel):
    peer_ticker: str


@router.get("/health/providers")
async def get_provider_health(force: bool = Query(default=False)) -> dict:
    providers = await run_in_threadpool(_PROVIDER_HEALTH.check_all, force=force)
    return {"providers": providers, "secrets_exposed": False}


def _new_engine() -> ResearchEngine:
    return ResearchEngine()


async def _execute(job_id: str, ticker: str, modules: list[str] | None, force_refresh: bool) -> None:
    store = _store()
    store.update_run(job_id, "RUNNING")

    def progress(module: str, status: str) -> None:
        if module in MODULES:
            store.update_module(job_id, module, status)

    try:
        engine = _new_engine()
        try:
            result = await run_in_threadpool(engine.run, ticker, modules=modules, force_refresh=force_refresh, progress=progress, run_id=job_id)
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            result = await run_in_threadpool(engine.run, ticker, refresh=force_refresh, progress=progress)
        result["run_id"] = job_id
        result.setdefault("engine_version", "research-v3.1")
        store.save_snapshot(job_id, result)
        for module in resolve_modules(modules):
            item = result.get("modules", {}).get(module)
            if item:
                display_status = "CACHED" if item.get("cache_hit") else item.get("status", "FAILED")
                store.update_module(job_id, module, display_status, item)
        store.update_run(job_id, result["status"])
    except Exception as exc:
        store.update_run(job_id, "FAILED", error=f"{type(exc).__name__}: {sanitize_error(exc)}")


async def _start(ticker: str, modules: list[str] | None, force_refresh: bool, trigger_type: str) -> dict:
    resolved = resolve_modules(modules)
    active = _store().active_run(ticker, resolved)
    if active:
        return {**active, "deduplicated": True}
    job_id = uuid4().hex
    job = _store().create_run(job_id, ticker, resolved, trigger_type)
    task = asyncio.create_task(_execute(job_id, ticker, modules, force_refresh))
    _RUNNING_TASKS.add(task)
    task.add_done_callback(_RUNNING_TASKS.discard)
    return job


@router.post("/runs")
async def create_research_run(body: ResearchRunInput) -> dict:
    try:
        ticker = normalize_ticker(body.ticker)
        resolve_modules(body.modules)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return await _start(ticker, body.modules, body.force_refresh or body.refresh, "MODULE" if body.modules else "FULL")


@router.get("/runs/{job_id}")
async def get_research_run(job_id: str) -> dict:
    job = await run_in_threadpool(_store().get_run, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="research run not found")
    if job.get("result"):
        job["result"] = await run_in_threadpool(present_result, job["result"])
    return job


@router.post("/runs/{run_id}/retry")
async def retry_research_modules(run_id: str, body: RetryInput) -> dict:
    source = await run_in_threadpool(_store().get_run, run_id)
    if not source:
        raise HTTPException(status_code=404, detail="research run not found")
    requested = set(body.modules)
    invalid = requested.difference(MODULES)
    if invalid:
        raise HTTPException(status_code=400, detail=f"unknown research module: {', '.join(sorted(invalid))}")
    retryable = {row["module"] for row in source["module_runs"] if row["status"] in {"PARTIAL_DATA", "PARTIAL_ERROR", "FAILED"} or (row.get("result") or {}).get("status") in {"PARTIAL_DATA", "PARTIAL_ERROR", "FAILED"}}
    modules = [name for name in MODULES if name in requested and name in retryable]
    if not modules:
        raise HTTPException(status_code=409, detail="none of the requested modules are failed or partial")
    return await _start(source["ticker"], modules, True, "RETRY")


@router.get("/results/{ticker}")
async def get_latest_research_result(ticker: str) -> dict:
    try:
        normalized = normalize_ticker(ticker)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    result = await run_in_threadpool(_store().latest, normalized)
    if result is None:
        raise HTTPException(status_code=404, detail="no research result")
    return await run_in_threadpool(present_result, result)


@router.get("/history/{ticker}")
async def get_research_history(ticker: str, limit: int = Query(default=50, ge=1, le=200)) -> dict:
    try:
        normalized = normalize_ticker(ticker)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ticker": normalized, "history": await run_in_threadpool(_store().history, normalized, limit)}


@router.get("/history/{ticker}/compare")
async def compare_research_history(ticker: str) -> dict:
    try:
        normalized = normalize_ticker(ticker)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    result = await run_in_threadpool(_store().compare_latest, normalized)
    if result is None:
        raise HTTPException(status_code=404, detail="at least two snapshots are required")
    return result


@router.get("/peers/{ticker}")
async def get_peer_preferences(ticker: str) -> dict:
    try:
        normalized = normalize_ticker(ticker)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ticker": normalized, **(await run_in_threadpool(_store().peer_preferences, normalized))}


@router.post("/peers/{ticker}")
async def add_peer(ticker: str, body: PeerPreferenceInput) -> dict:
    try:
        normalized, peer = normalize_ticker(ticker), normalize_ticker(body.peer_ticker)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if normalized == peer:
        raise HTTPException(status_code=400, detail="ticker cannot be its own peer")
    return {"ticker": normalized, **(await run_in_threadpool(_store().set_peer_preference, normalized, peer, "ADD"))}


@router.delete("/peers/{ticker}/{peer_ticker}")
async def remove_peer(ticker: str, peer_ticker: str) -> dict:
    try:
        normalized, peer = normalize_ticker(ticker), normalize_ticker(peer_ticker)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ticker": normalized, **(await run_in_threadpool(_store().set_peer_preference, normalized, peer, "REMOVE"))}


@router.post("/supply-chain/relationships/{relationship_id}/revalidate")
async def revalidate_relationship(relationship_id: int) -> dict:
    relation = await run_in_threadpool(_store().relationship, relationship_id)
    if not relation:
        raise HTTPException(status_code=404, detail="relationship not found")

    def perform() -> dict:
        from v2.data import CachedFDClient
        from v2.lateral.models import Label, Neighbor
        from v2.lateral.verify import verify_relation
        category = {"SUPPLIER": "supplier", "CUSTOMER": "customer", "COMPETITOR": "smaller_peer", "BENEFICIARY": "beneficiary"}.get(relation["relationship_type"], "beneficiary")
        neighbor = Neighbor(ticker=relation["target_ticker"], labels=[Label(seed=relation["source_ticker"], category=category, reason=relation.get("description") or "relationship revalidation")], exists=True)
        calls = verify_relation(neighbor)
        label = neighbor.labels[0]
        relation_id = _store().upsert_relationship({**relation, "status": label.evidence_status, "confidence": 0}, label_sources(label))
        return {**(_store().relationship(relation_id) or {}), "tavily_calls": calls}

    return await run_in_threadpool(perform)
