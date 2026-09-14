"""Structured business ports; no display-card parsing or legacy responders."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
import hashlib
import json
import math

from v2.agent_v2.models import EvidenceItem, ResultStatus, ToolEnvelope


def json_value(value):
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return json_value(value.value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Business port returned unsupported type: {type(value).__name__}")


def structured_envelope(capability, payload, context, *, subject="account", warnings=()):
    payload = json_value(payload)
    as_of = datetime.now(timezone.utc).isoformat()
    evidence = []
    # Preserve each object's fields together: position identity and units travel with numbers.
    rows = payload if isinstance(payload, list) else [payload]
    for index, row in enumerate(rows[:80]):
        text = json.dumps(row, ensure_ascii=False, allow_nan=False)
        digest = hashlib.sha256(f"{capability}|{text}".encode()).hexdigest()[:16]
        evidence.append(EvidenceItem(id=f"v3-business-{digest}", entity=str(row.get("ticker", row.get("symbol", subject))) if isinstance(row, dict) else subject, claim=text, as_of=as_of, source_id=capability, producer_run_id=context.run_id, metadata={"structured": row}))
    limits = list(warnings)
    if len(rows) > 80:
        limits.append("Result limited to the first 80 structured rows")
    return ToolEnvelope(capability, ResultStatus.PARTIAL_DATA if limits else ResultStatus.COMPLETED, subject=subject, as_of=as_of, evidence=evidence, limitations=limits)


def register_account(registry, *, portfolio_source=None, pnl_source=None, risk_source=None):
    def portfolio(args, ctx):
        if portfolio_source is None:
            from v2.broker import get_portfolio
            source = get_portfolio
        else:
            source = portfolio_source
        snapshot = source()
        positions = [{**row, "ticker": row["symbol"], "pl_pct": float(row["unrealized_pl_pct"])*100 if row.get("unrealized_pl_pct") is not None else None, "pl_pct_unit": "percent", "unrealized_pl_pct_unit": "fraction"} for row in snapshot.get("positions", [])]
        envelope = structured_envelope("account.portfolio", positions, ctx)
        account = structured_envelope("account.portfolio", snapshot.get("account", {}), ctx)
        envelope.evidence.extend(account.evidence)
        envelope.metadata = {"positions": positions, "tickers": [row["ticker"] for row in positions], "account": json_value(snapshot.get("account", {})), "positions_available": "positions" in snapshot}
        return envelope
    def pnl(args, ctx):
        if pnl_source is None:
            from v2.portfolio.pnl import compute_pnl
            metrics, warnings = compute_pnl()
        else:
            metrics, warnings = pnl_source()
        values = json_value(metrics)
        period = args["period"]
        fields = {"day": ("daily_pnl", "daily_pnl_pct"), "week": ("weekly_pnl_pct",), "month": ("monthly_pnl_pct",)}[period]
        data = {"period": period, "percentage_unit": "fraction", **{key: values.get(key) for key in fields}}
        return structured_envelope("account.performance", data, ctx, warnings=warnings)
    def risk(args, ctx):
        if risk_source is None:
            from v2.portfolio import build_risk_report
            source = build_risk_report
        else:
            source = risk_source
        report = json_value(source())
        return structured_envelope("account.risk", report, ctx, warnings=report.get("warnings", []))
    registry.register("account.portfolio", portfolio)
    registry.register("account.performance", pnl)
    registry.register("account.risk", risk)
    from v2.agent_v3.portfolio_overview import register_overview
    register_overview(registry)


def register_user_state(registry, *, state_source=None, enable_mutations=False):
    def source():
        if state_source is not None:
            return state_source
        from v2.bot import state
        return state
    def read(args, ctx):
        state = source()
        section = args["section"]
        value = {"watchlist": state.watchlist_list, "alerts": lambda: state.alert_list(False), "settings": state.settings_all}[section]()
        result = structured_envelope("state.read", value, ctx, subject=section)
        if section == "watchlist":
            result.metadata["tickers"] = [row["ticker"] for row in value]
        return result
    def mutate(args, ctx):
        state, payload, operation = source(), args["payload"], args["operation"]
        actions = {
            "watchlist.add": lambda: state.watchlist_add(payload["ticker"]),
            "watchlist.remove": lambda: state.watchlist_remove(payload["ticker"]),
            "alert.add": lambda: state.alert_add(payload["ticker"], payload["direction"], payload["target_price"]),
            "alert.remove": lambda: state.alert_remove(payload["alert_id"]),
        }
        value = actions[operation]()
        return structured_envelope("state.mutate", {"operation": operation, "payload": payload, "result": value}, ctx, subject=operation)
    registry.register("state.read", read)
    if enable_mutations:
        registry.register("state.mutate", mutate)
