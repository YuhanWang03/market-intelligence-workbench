"""ARK fund holdings and daily activity, as field-level evidence.

ARK publishes each fund's holdings every trading day as a CSV; the V2
scheduler snapshots them into ``data/etf.db``.  V3 fetches the latest CSV
(falling back to the newest tracked snapshot when the CDN is unavailable),
diffs it against the newest earlier snapshot with the existing detector, and
emits one evidence item per fact: the snapshot summary, each of the top
holdings, and each significant day-over-day change.
"""
from __future__ import annotations

import re
import sqlite3
from typing import Any

from v2.agent_v2.catalog import CapabilityCatalog, CapabilitySpec
from v2.agent_v2.models import EvidenceItem, ResultStatus, ToolEnvelope

CAPABILITY = "etf.ark_activity"
SOURCE_ID = "ark_daily_holdings_csv"
DEFAULT_TOP = 10
MAX_CHANGES = 10

_FUND_NAMES = {"ARKK": "ARK Innovation ETF", "ARKW": "ARK Next Generation Internet ETF", "ARKG": "ARK Genomic Revolution ETF", "ARKF": "ARK Fintech Innovation ETF"}
_ALIASES = {"ark": "ARKK", "arkinvest": "ARKK", "arkinnovation": "ARKK", "cathiewood": "ARKK", "木头姐": "ARKK", "方舟": "ARKK", "方舟创新": "ARKK", "arkgenomic": "ARKG", "arkfintech": "ARKF", "arkweb": "ARKW"}
_STRIP = re.compile(r"[\s\.\-_,&'’\"()（）·]+")


def supported_funds() -> list[str]:
    try:
        from v2.etf.client import SUPPORTED_FUNDS
        return list(SUPPORTED_FUNDS)
    except Exception:  # noqa: BLE001 — the module is optional in stripped test environments
        return list(_FUND_NAMES)


def resolve_fund(text: str) -> str | None:
    """Fund symbol for a code, alias or Chinese name; None when unsupported."""
    key = _STRIP.sub("", str(text or "")).lower()
    if not key:
        return None
    if re.fullmatch(r"ark[a-z]", key):
        # An explicit fund code (ARKQ, ARKX, ...) is exact: never rewritten to another fund.
        return key.upper() if key.upper() in supported_funds() else None
    symbol = _ALIASES.get(key)
    if symbol is None:
        for alias, candidate in sorted(_ALIASES.items(), key=lambda item: -len(item[0])):
            if len(alias) >= 3 and alias in key:
                symbol = candidate
                break
    return symbol if symbol in supported_funds() else None


def _csv_url(symbol: str) -> str:
    try:
        from v2.etf.client import _ARK_URLS
        return _ARK_URLS.get(symbol, "https://www.ark-funds.com/")
    except Exception:  # noqa: BLE001
        return "https://www.ark-funds.com/"


def _row(holding) -> dict:
    """ETFHolding dataclass or tracker row → one plain dict shape."""
    if isinstance(holding, dict):
        return dict(holding)
    return {key: getattr(holding, key) for key in ("etf", "date", "ticker", "cusip", "company", "shares", "market_value", "weight_pct")}


def tracked_snapshot(symbol: str, *, before: str | None = None, db_path=None) -> tuple[str, list[dict]] | None:
    """Newest tracked snapshot for a fund (strictly before ``before`` when given): (date, rows)."""
    from v2.etf import tracker

    conn = sqlite3.connect(str(db_path or tracker._DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='snapshots'").fetchone():
            return None
        query, params = ("SELECT MAX(date) FROM snapshots WHERE etf=? AND date<?", (symbol, before)) if before else ("SELECT MAX(date) FROM snapshots WHERE etf=?", (symbol,))
        latest = conn.execute(query, params).fetchone()[0]
        if not latest:
            return None
        return latest, [dict(row) for row in conn.execute("SELECT * FROM snapshots WHERE etf=? AND date=?", (symbol, latest))]
    finally:
        conn.close()


def default_reader(symbol: str) -> tuple[str, list[dict], str]:
    """(snapshot date, holdings, provenance): today's CSV when ARK serves it, else the newest tracked snapshot."""
    try:
        from v2.etf import fetch_holdings

        holdings, snapshot_date = fetch_holdings(symbol)
        if holdings:
            return snapshot_date, [_row(h) for h in holdings], "live"
    except Exception:  # noqa: BLE001 — the CDN is best-effort; the tracker is the fallback
        pass
    tracked = tracked_snapshot(symbol)
    if tracked is None:
        raise LookupError(f"{symbol}: ARK CSV unavailable and no tracked snapshot")
    return tracked[0], tracked[1], "tracked"


def default_prior(symbol: str, before: str) -> tuple[str, list[dict]] | None:
    return tracked_snapshot(symbol, before=before)


def default_saver(symbol: str, snapshot_date: str, rows: list[dict]) -> None:
    from v2.etf import save_snapshot
    from v2.etf.models import ETFHolding

    save_snapshot(symbol, snapshot_date, [ETFHolding(etf=symbol, date=snapshot_date, ticker=r.get("ticker"), cusip=r.get("cusip"), company=r.get("company") or "", shares=float(r["shares"]), market_value=float(r["market_value"]), weight_pct=float(r["weight_pct"])) for r in rows])


def _label(row: dict) -> str:
    ticker, company = row.get("ticker"), row.get("company") or ""
    return f"{ticker}（{company}）" if ticker and ticker != "?" else company or (row.get("cusip") or "?")


def activity_envelope(symbol: str, snapshot_date: str, rows: list[dict], prior: tuple[str, list[dict]] | None, *, top: int = DEFAULT_TOP, provenance: str = "live", run_id: str = "") -> ToolEnvelope:
    name = _FUND_NAMES.get(symbol, symbol)
    url = _csv_url(symbol)
    total = sum(float(r["market_value"]) for r in rows)
    base = dict(source_id=SOURCE_ID, source_title=f"ARK Invest daily holdings CSV ({symbol})", source_url=url, period=snapshot_date, as_of=snapshot_date, producer_run_id=run_id)
    evidence = [EvidenceItem(
        f"ark-{symbol}-{snapshot_date}-snapshot", symbol,
        f"{symbol}（{name}）{snapshot_date} 持仓快照：共 {len(rows)} 个持仓，持仓总市值 {total:,.0f} 美元。",
        metric="holdings_value", value=round(total, 2), unit="USD",
        metadata={"fund_name": name, "position_count": len(rows), "date_basis": "publication", "provenance": provenance}, **base,
    )]
    ranked = sorted(rows, key=lambda r: -float(r["weight_pct"]))
    for rank, row in enumerate(ranked[:top], start=1):
        weight = round(float(row["weight_pct"]), 2)
        evidence.append(EvidenceItem(
            f"ark-{symbol}-{snapshot_date}-pos-{row.get('ticker') or row.get('cusip') or rank}", symbol,
            f"{symbol} 第 {rank} 大持仓 {_label(row)}：{float(row['shares']):,.0f} 股，市值 {float(row['market_value']):,.0f} 美元，权重 {weight}%。",
            metric="holding_weight", value=weight, unit="%",
            metadata={"rank": rank, "ticker": row.get("ticker"), "company": row.get("company"), "cusip": row.get("cusip"), "shares": float(row["shares"]), "market_value": round(float(row["market_value"]), 2), "date_basis": "publication"}, **base,
        ))
    limitations = [
        f"仅列出权重最大的 {min(top, len(ranked))} 个持仓，不是完整组合；权重为 ARK 公布的组合权重，未重新归一。",
        "ARK 每个交易日公布持仓，快照日期是公布日期；日内交易不会体现。",
    ]
    if provenance == "tracked":
        limitations.append("ARK 数据源暂不可用，本次使用本地跟踪库中最新的一份快照，可能不是最新公布。")
    changes_out: list[dict] = []
    status = ResultStatus.COMPLETED
    if prior and prior[1]:
        from v2.etf.detector import compute_daily_changes
        from v2.etf.models import ETFHolding

        prior_date, prior_rows = prior
        today = [ETFHolding(etf=symbol, date=snapshot_date, ticker=r.get("ticker"), cusip=r.get("cusip"), company=r.get("company") or "", shares=float(r["shares"]), market_value=float(r["market_value"]), weight_pct=float(r["weight_pct"])) for r in rows]
        changes = compute_daily_changes(prior_rows, today)
        for change in changes[:MAX_CHANGES]:
            kind = "新建仓" if change["is_new"] else ("清仓" if change["is_exit"] else ("加仓" if change["shares_diff"] > 0 else "减仓"))
            pct = round(float(change["shares_diff_pct"]) * 100, 2)
            weight_diff = round(float(change["weight_diff_pp"]), 2)
            body = f"{kind} {change['ticker']}（{change.get('company') or ''}）：股数变动 {float(change['shares_diff']):+,.0f} 股" + (f"（{pct:+}%）" if not change["is_new"] else "") + f"，权重变动 {weight_diff:+} 个百分点。"
            evidence.append(EvidenceItem(
                f"ark-{symbol}-{snapshot_date}-chg-{change['ticker']}", symbol,
                f"{symbol} {snapshot_date} 相比 {prior_date} {body}",
                metric="shares_change", value=round(float(change["shares_diff"]), 2), unit="shares",
                metadata={"change_type": "new" if change["is_new"] else ("exit" if change["is_exit"] else ("increase" if change["shares_diff"] > 0 else "decrease")), "ticker": change["ticker"], "company": change.get("company"), "shares_diff_pct": pct, "weight_pct": round(float(change["weight_pct"]), 2), "weight_diff_pp": weight_diff, "previous_date": prior_date, "date_basis": "publication"}, **base,
            ))
            changes_out.append({"ticker": change["ticker"], "change_type": "new" if change["is_new"] else ("exit" if change["is_exit"] else ("increase" if change["shares_diff"] > 0 else "decrease")), "shares_diff": float(change["shares_diff"]), "weight_diff_pp": weight_diff})
        limitations.append(f"变动对比的是上一份可用快照（{prior_date}），不一定是前一交易日；变动阈值为股数变化至少 1%。")
        if not changes:
            limitations.append(f"与 {prior_date} 相比没有达到阈值的持仓变动。")
    else:
        status = ResultStatus.PARTIAL_DATA
        limitations.append("没有更早的快照可比，无法给出持仓变动。")
    metrics = {"holdings_value_usd": round(total, 2), "position_count": len(rows), "top_listed": min(top, len(ranked)), "significant_changes": len(changes_out)}
    return ToolEnvelope(CAPABILITY, status, subject=symbol, as_of=snapshot_date, summary=f"{symbol} {snapshot_date}：{len(rows)} 个持仓，总市值 {total:,.0f} 美元，{len(changes_out)} 项持仓变动。",
                        metrics=metrics, findings=changes_out, evidence=evidence, limitations=limitations,
                        metadata={"fund": symbol, "fund_name": name, "snapshot_date": snapshot_date, "previous_date": prior[0] if prior else None, "provenance": provenance})


def register_ark(registry, *, reader=None, prior=None, saver=None) -> None:
    """Register ``etf.ark_activity``; ``reader(symbol)`` → (date, rows, provenance), ``prior(symbol, before)`` → (date, rows) | None."""
    reader = reader or default_reader
    prior = prior or default_prior
    saver = default_saver if saver is None else saver
    schema = {"type": "object", "properties": {"symbol": {"type": "string", "minLength": 1, "maxLength": 40}, "top": {"type": "integer", "minimum": 1, "maximum": 25}}, "required": ["symbol"], "additionalProperties": False}
    spec = CapabilitySpec(CAPABILITY, "research", "ARK fund daily holdings: snapshot summary, top holdings by weight and significant changes versus the previous snapshot.", schema,
                          answer_guidance="ark：先写快照日期和它只是公布日持仓，再说主要持仓，最后说相对上一份快照的新建仓、清仓和明显加减仓；没有可比快照时明说。")
    registry.catalog = CapabilityCatalog([*(s for s in registry.catalog.specs() if s.name != CAPABILITY), spec])

    def handler(arguments: dict[str, Any], context) -> ToolEnvelope:
        asked = str(arguments["symbol"])
        symbol = resolve_fund(asked)
        if symbol is None:
            return ToolEnvelope(CAPABILITY, ResultStatus.PARTIAL_DATA, subject=asked.upper(), limitations=[f"不支持的 ARK 基金“{asked}”。目前可查询：{'、'.join(supported_funds())}（ARKQ 的每日持仓文件 ARK 已停止在原地址发布）。"])
        run_id = getattr(getattr(context, "v3_run", None), "run_id", "") or ""
        try:
            snapshot_date, rows, provenance = reader(symbol)
        except Exception as exc:  # noqa: BLE001 — a data-source failure is a partial result, not a crash
            return ToolEnvelope(CAPABILITY, ResultStatus.PARTIAL_ERROR, subject=symbol, errors=[f"ARK 持仓读取失败：{type(exc).__name__}: {str(exc)[:160]}"])
        previous = None
        try:
            previous = prior(symbol, snapshot_date)
        except Exception:  # noqa: BLE001 — no comparison is a limitation, not an error
            previous = None
        if provenance == "live" and saver:
            try:
                saver(symbol, snapshot_date, rows)
            except Exception:  # noqa: BLE001 — caching the snapshot never blocks the answer
                pass
        return activity_envelope(symbol, snapshot_date, rows, previous, top=int(arguments.get("top") or DEFAULT_TOP), provenance=provenance, run_id=run_id)

    registry.register(CAPABILITY, handler)
