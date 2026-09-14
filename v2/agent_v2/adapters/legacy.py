"""Lazy adapters for useful read-only responders that Research Engine does not replace."""

from __future__ import annotations

import hashlib
import time
import threading
import importlib
import json
import re
from typing import Any, Callable

from v2.agent_v2.entities import extract_entities
from v2.agent_v2.execution import CapabilityRegistry, ExecutionContext
from v2.agent_v2.models import EvidenceItem, ResultStatus, ToolEnvelope
from v2.agent_v2.text import plain_text

#: Capabilities whose card lists the user's own tickers; fan-out tasks read them.
_LISTS_TICKERS = frozenset({"account.portfolio", "state.read"})


def _resolve(path: str) -> Callable[..., Any]:
    module_name, _, attr = path.rpartition(".")
    return getattr(importlib.import_module(module_name), attr)


def _text(value: Any) -> str:
    if isinstance(value, tuple):
        value = value[0]
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


_POSITION_HEAD = re.compile(r"^(?:[^\w\s]+\s*)?(?P<ticker>[A-Z][A-Z0-9.\-]{0,9})\s+(?P<qty>[\d,]+(?:\.\d+)?)\s*sh\s*@\s*\$(?P<entry>[\d,]+(?:\.\d+)?)")
_POSITION_TAIL = re.compile(r"市值\s*(?P<value>\$[\d,.]+[KMBT]?)\s*·\s*P/L\s*(?P<pl>\$[\d,.]+[KMBT]?)\s*(?P<pct>[+-]\d+(?:\.\d+)?%)")
_SCALE = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}

#: How a ranking question maps onto the position table; the synthesizer reads
#: these instead of knowing what a portfolio is.
PORTFOLIO_RANKABLE: tuple[dict[str, str], ...] = (
    {"field": "pl_pct", "text": "pl_pct_text", "label": "买入以来的浮动盈亏", "low": "跌|亏|差|弱|回撤|最惨", "high": "涨|赚|好|强|盈利"},
    {"field": "market_value", "text": "market_value_text", "label": "市值", "low": "小|轻", "high": "大|重", "topic": "市值|仓位|占比|权重|头寸"},
)


def _money(text: str) -> float | None:
    raw = text.lstrip("$").replace(",", "")
    scale = 1.0
    if raw and raw[-1].upper() in _SCALE:
        scale = _SCALE[raw[-1].upper()]
        raw = raw[:-1]
    try:
        return float(raw) * scale
    except ValueError:
        return None


def parse_portfolio_card(text: str) -> list[dict[str, Any]]:
    """Read the position rows of the portfolio card into structured rows.

    The card prints ``TICKER qty sh @ $entry`` and, on the next line,
    ``市值 $value · P/L $abs +pct%``; the sign of the percentage carries the
    direction of the dollar P/L.  Rows that do not parse are skipped.
    """

    lines = [line.strip() for line in plain_text(text).split("\n")]
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        head = _POSITION_HEAD.match(line)
        if head is None or index + 1 >= len(lines):
            continue
        tail = _POSITION_TAIL.search(lines[index + 1])
        if tail is None:
            continue
        pct_text = tail.group("pct")
        pct = float(pct_text.rstrip("%"))
        pl_abs = _money(tail.group("pl"))
        rows.append(
            {
                "ticker": head.group("ticker"),
                "qty": float(head.group("qty").replace(",", "")),
                "avg_entry_price": float(head.group("entry").replace(",", "")),
                "market_value": _money(tail.group("value")),
                "market_value_text": tail.group("value"),
                "pl": None if pl_abs is None else (pl_abs if pct >= 0 else -pl_abs),
                "pl_pct": pct,
                "pl_pct_text": pct_text,
            }
        )
    return rows


def _wrap(capability: str, subject: str, value: Any) -> ToolEnvelope:
    # Responder cards are Telegram HTML; the answer layer wants prose, and the
    # verifier traces numbers through the claim text either way.
    content = plain_text(_text(value))
    digest = hashlib.sha256(f"{capability}:{subject}:{content}".encode("utf-8")).hexdigest()[:16]
    evidence = EvidenceItem(
        id=f"legacy-{digest}",
        entity=subject,
        claim=content,
        source_id=capability,
        source_title="Existing deterministic responder",
        metadata={"legacy_formatted_output": True},
    )
    metadata: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    if capability in _LISTS_TICKERS:
        metadata["tickers"] = list(extract_entities(content))
    if capability == "account.portfolio":
        positions = parse_portfolio_card(content)
        if positions:
            metadata["positions"] = positions
            metadata["rankable"] = [dict(rule) for rule in PORTFOLIO_RANKABLE]
            metadata["tickers"] = [row["ticker"] for row in positions]
            metrics["positions"] = [{key: row[key] for key in ("ticker", "qty", "market_value", "pl", "pl_pct")} for row in positions]
    return ToolEnvelope(
        capability,
        ResultStatus.COMPLETED,
        subject=subject,
        summary=content[:6000],
        metrics=metrics,
        evidence=[evidence],
        limitations=["Legacy formatted output; structured field-level evidence is not yet available."],
        metadata=metadata,
    )


#: Responders whose answer does not change within the window: the macro board
#: (FRED plus Yahoo, minutes when FRED is slow — its P90 was 218 s), 13F and ARK.
_CACHE_TTL_SECONDS = {"macro.overview": 600.0, "macro.release": 600.0, "institutional.manager_portfolio": 3600.0, "etf.ark_activity": 1800.0}
_cache: dict[tuple[str, str], tuple[float, Any]] = {}
_cache_lock = threading.Lock()


def cached_value(capability: str, subject: str, produce: Callable[[], Any], *, now: float | None = None) -> tuple[Any, bool]:
    """The responder's text for ``(capability, subject)``, fresh or from the cache; the flag says which."""

    ttl = _CACHE_TTL_SECONDS.get(capability)
    if not ttl:
        return produce(), False
    moment = time.monotonic() if now is None else now
    key = (capability, subject)
    with _cache_lock:
        entry = _cache.get(key)
        if entry is not None and moment - entry[0] < ttl:
            return entry[1], True
    value = produce()
    with _cache_lock:
        _cache[key] = (moment, value)
    return value, False


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


def register_legacy_capabilities(registry: CapabilityRegistry) -> None:
    def call(path: str, capability: str, invoke: Callable[[Callable[..., Any], dict[str, Any]], Any], subject: Callable[[dict[str, Any]], str] = lambda _: ""):
        def handler(arguments: dict[str, Any], context: ExecutionContext) -> ToolEnvelope:
            value, hit = cached_value(capability, subject(arguments), lambda: invoke(_resolve(path), arguments))
            envelope = _wrap(capability, subject(arguments), value)
            if hit:
                envelope.cache_hit = True
                envelope.status = ResultStatus.CACHED
            return envelope

        registry.register(capability, handler)

    call("v2.bot.responders.portfolio_view", "account.portfolio", lambda fn, args: fn(), lambda args: "portfolio")
    call("v2.bot.responders.pnl_period", "account.performance", lambda fn, args: fn({"period": args.get("period", "day")}), lambda args: "portfolio")
    call("v2.bot.responders.risk_view", "account.risk", lambda fn, args: fn({}), lambda args: "portfolio")
    call("v2.bot.responders.earnings_calendar", "account.earnings_schedule", lambda fn, args: fn({"days_horizon": args.get("days", 14)}), lambda args: "portfolio")
    call("v2.bot.responders.institutional_quick", "institutional.manager_portfolio", lambda fn, args: fn(args["manager"]), lambda args: str(args.get("manager", "")))
    call("v2.bot.responders.etf_view", "etf.ark_activity", lambda fn, args: fn(args["symbol"]), lambda args: str(args.get("symbol", "")))
    call("v2.bot.responders.macro_view", "macro.overview", lambda fn, args: fn({}), lambda args: "macro")
    call("v2.bot.responders.release_check", "macro.release", lambda fn, args: fn({"release_type": args["release_type"]}), lambda args: str(args.get("release_type", "")))

    def state_read(arguments: dict[str, Any], context: ExecutionContext) -> ToolEnvelope:
        section = str(arguments.get("section") or "watchlist")
        if section == "watchlist":
            value = _resolve("v2.bot.state.watchlist_list")()
        elif section == "alerts":
            value = _resolve("v2.bot.state.alert_list")(False)
        else:
            value = _resolve("v2.bot.state.settings_all")()
        return _wrap("state.read", section, value)

    registry.register("state.read", state_read)

    def state_mutate(arguments: dict[str, Any], context: ExecutionContext) -> ToolEnvelope:
        operation = str(arguments.get("operation") or "")
        payload = dict(arguments.get("payload") or {})
        state = importlib.import_module("v2.bot.state")
        if operation == "watchlist.add":
            ticker = str(payload.get("ticker") or "").upper()
            added = state.watchlist_add(ticker)
            message = f"已将 {ticker} 加入关注列表。" if added else f"{ticker} 已在关注列表中，无需重复添加。"
        elif operation == "watchlist.remove":
            ticker = str(payload.get("ticker") or "").upper()
            removed = state.watchlist_remove(ticker)
            message = f"已将 {ticker} 移出关注列表。" if removed else f"{ticker} 不在关注列表中。"
        elif operation == "alert.add":
            ticker = str(payload.get("ticker") or "").upper()
            direction = str(payload.get("direction") or "above")
            target = float(payload.get("target_price") or 0)
            alert_id = state.alert_add(ticker, direction, target)
            label = "涨到" if direction == "above" else "跌到"
            message = f"已设置提醒 #{alert_id}：{ticker} {label} {target:g} 美元时通知。"
        elif operation == "alert.remove":
            alert_id = int(payload.get("alert_id") or 0)
            removed = state.alert_remove(alert_id)
            message = f"已取消提醒 #{alert_id}。" if removed else f"提醒 #{alert_id} 不存在。"
        else:
            return ToolEnvelope("state.mutate", ResultStatus.FAILED, errors=[f"unsupported operation: {operation}"])
        return _wrap("state.mutate", operation, message)

    registry.register("state.mutate", state_mutate)
