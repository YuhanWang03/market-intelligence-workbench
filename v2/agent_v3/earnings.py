"""Upcoming earnings across the account's holdings and watchlist, as field-level evidence.

V2 answers ``account.earnings_schedule`` with the Telegram calendar card.
V3 reads the same universe (Alpaca positions ∪ bot watchlist) and the same
per-ticker calendar (yfinance through ``v2.earnings``), and emits one
evidence item per scheduled release inside the horizon, so the verifier can
trace every date and estimate.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

from v2.agent_v2.catalog import CapabilityCatalog, CapabilitySpec
from v2.agent_v2.models import EvidenceItem, ResultStatus, ToolEnvelope

CAPABILITY = "account.earnings_schedule"
SOURCE_ID = "yfinance_earnings_calendar"
DEFAULT_DAYS = 14
_WHEN = {"bmo": "盘前", "amc": "盘后", "unknown": "时段未公布"}


def default_universe() -> tuple[set[str], set[str]]:
    """(held, watchlist) tickers, the way V2's calendar card builds them; a missing broker is not a failure."""
    watchlist: set[str] = set()
    try:
        from v2.bot import state as bot_state
        watchlist = {str(row["ticker"]).upper() for row in bot_state.watchlist_list()}
    except Exception:  # noqa: BLE001 — the bot state store is optional outside production
        pass
    held: set[str] = set()
    try:
        from v2.broker import get_portfolio
        held = {str(row["symbol"]).upper() for row in get_portfolio().get("positions", [])}
    except Exception:  # noqa: BLE001 — Alpaca unavailable: watchlist-only, as V2 does
        pass
    return held, watchlist


def default_calendar(tickers: list[str]):
    from v2.earnings import get_upcoming_batch
    return get_upcoming_batch(tickers)


def _event_fields(event) -> dict:
    if isinstance(event, dict):
        return dict(event)
    return {key: getattr(event, key, None) for key in ("ticker", "release_date", "when", "eps_estimate", "revenue_estimate", "n_analysts", "source")}


def schedule_envelope(held: set[str], watchlist: set[str], batch, *, days: int = DEFAULT_DAYS, today: date | None = None, run_id: str = "") -> ToolEnvelope:
    today = today or date.today()
    horizon_end = today + timedelta(days=days)
    universe = sorted(held | watchlist)
    if not universe:
        return ToolEnvelope(CAPABILITY, ResultStatus.PARTIAL_DATA, subject="portfolio", limitations=["持仓和关注列表都为空，没有可查询财报日期的标的。"])
    events = getattr(batch, "events", None) or {}
    rows = sorted((_event_fields(event) for event in events.values()), key=lambda row: (str(row.get("release_date") or ""), str(row.get("ticker") or "")))
    within, beyond = [], []
    for row in rows:
        try:
            release = datetime.strptime(str(row["release_date"]), "%Y-%m-%d").date()
        except (TypeError, ValueError):
            continue
        (within if today <= release <= horizon_end else beyond).append((release, row))
    base = dict(source_id=SOURCE_ID, source_title="Yahoo Finance earnings calendar (yfinance)", producer_run_id=run_id, as_of=today.isoformat())
    evidence = [EvidenceItem(
        f"earnings-window-{today.isoformat()}-{days}d", "portfolio",
        f"截至 {today.isoformat()}，持仓与关注列表共 {len(universe)} 个标的，未来 {days} 天内有 {len(within)} 个已排期的财报发布。",
        metric="scheduled_releases", value=len(within), unit="count", period=f"{today.isoformat()}/{horizon_end.isoformat()}",
        metadata={"universe": universe, "held": sorted(held), "watchlist": sorted(watchlist), "horizon_days": days, "date_basis": "scheduled"}, **base,
    )]
    for release, row in within:
        ticker = str(row["ticker"]).upper()
        membership = "持仓" if ticker in held and ticker in watchlist else ("持仓" if ticker in held else "关注列表")
        when = _WHEN.get(str(row.get("when") or "unknown"), "时段未公布")
        d_minus = (release - today).days
        extras = []
        if row.get("eps_estimate") is not None:
            extras.append(f"EPS 一致预期 {float(row['eps_estimate']):.2f} 美元")
        if row.get("revenue_estimate") is not None:
            extras.append(f"营收一致预期 {float(row['revenue_estimate']):,.0f} 美元")
        if row.get("n_analysts") is not None:
            extras.append(f"{int(row['n_analysts'])} 位分析师")
        evidence.append(EvidenceItem(
            f"earnings-{ticker}-{release.isoformat()}", ticker,
            f"{ticker}（{membership}）下一次财报发布日期为 {release.isoformat()}（{when}），距今 {d_minus} 天" + ("，" + "，".join(extras) if extras else "") + "。",
            metric="days_until_earnings", value=d_minus, unit="days", period=release.isoformat(),
            source_url=f"https://finance.yahoo.com/quote/{ticker}/", metadata={"release_date": release.isoformat(), "when": row.get("when"), "held": ticker in held, "watchlist": ticker in watchlist, "eps_estimate": row.get("eps_estimate"), "revenue_estimate": row.get("revenue_estimate"), "n_analysts": row.get("n_analysts"), "date_basis": "scheduled", "provider": row.get("source") or "yfinance"}, **base,
        ))
    limitations = ["财报日期来自 Yahoo Finance 日历，公司未正式确认前可能变动；盘前/盘后时段以公司公告为准。"]
    if beyond:
        limitations.append(f"另有 {len(beyond)} 个标的的下一次财报在 {days} 天窗口之外（最近的是 {beyond[0][1]['ticker']} {beyond[0][0].isoformat()}），未列入。")
    for label, names in (("日历未覆盖", getattr(batch, "skipped_unsupported", None)), ("供应商没有排期信息", getattr(batch, "skipped_empty", None))):
        if names:
            limitations.append(f"{label}：{'、'.join(sorted(str(n).upper() for n in names))}。")
    errors = [str(e) for e in (getattr(batch, "errors", None) or [])]
    status = ResultStatus.PARTIAL_DATA if errors or (getattr(batch, "skipped_empty", None)) else ResultStatus.COMPLETED
    metrics = {"universe": len(universe), "scheduled_within_horizon": len(within), "beyond_horizon": len(beyond), "horizon_days": days}
    return ToolEnvelope(CAPABILITY, status, subject="portfolio", as_of=today.isoformat(),
                        summary=f"未来 {days} 天内 {len(within)} 个财报发布（{len(universe)} 个标的）。",
                        metrics=metrics, findings=[{"ticker": str(row["ticker"]).upper(), "release_date": release.isoformat(), "when": row.get("when")} for release, row in within],
                        evidence=evidence, limitations=limitations, errors=[f"日历查询出错：{e}" for e in errors],
                        metadata={"horizon_days": days, "universe": universe, "today": today.isoformat()})


def register_earnings(registry, *, universe=None, calendar=None, today=None) -> None:
    """Register ``account.earnings_schedule``; ``universe()`` → (held, watchlist), ``calendar(tickers)`` → batch result."""
    universe = universe or default_universe
    calendar = calendar or default_calendar
    schema = {"type": "object", "properties": {"days": {"type": "integer", "minimum": 1, "maximum": 90}}, "additionalProperties": False}
    spec = CapabilitySpec(CAPABILITY, "account", "Upcoming earnings releases across holdings and watchlist within a horizon (default 14 days).", schema,
                          answer_guidance="earnings_schedule：按日期顺序列出窗口内的财报，标明持仓还是关注列表、盘前或盘后；说明窗口之外和没有排期信息的标的，不要推测日期。")
    registry.catalog = CapabilityCatalog([*(s for s in registry.catalog.specs() if s.name != CAPABILITY), spec])

    def handler(arguments: dict[str, Any], context) -> ToolEnvelope:
        days = int(arguments.get("days") or DEFAULT_DAYS)
        run_id = getattr(getattr(context, "v3_run", None), "run_id", "") or ""
        try:
            held, watchlist = universe()
            tickers = sorted(held | watchlist)
            batch = calendar(tickers) if tickers else None
        except Exception as exc:  # noqa: BLE001 — a data-source failure is a partial result, not a crash
            return ToolEnvelope(CAPABILITY, ResultStatus.PARTIAL_ERROR, subject="portfolio", errors=[f"财报日历读取失败：{type(exc).__name__}: {str(exc)[:160]}"])
        return schedule_envelope(held, watchlist, batch, days=days, today=today() if callable(today) else today, run_id=run_id)

    registry.register(CAPABILITY, handler)
