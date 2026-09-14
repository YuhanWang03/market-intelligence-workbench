"""Legacy slash-command formatting backed by the Research Engine.

This module is intentionally a one-way adapter:

    slash command -> Research Engine -> HTML

The Research Engine never imports this module or any bot responder.
"""

from __future__ import annotations

import html
from typing import Any

from v2.research.engine import ResearchEngine


def _result(ticker: str) -> dict:
    return ResearchEngine().run(ticker)


def _fmt(value: Any, *, pct: bool = False, money: bool = False) -> str:
    if value is None:
        return "N/A"
    if pct:
        return f"{float(value):+.1%}"
    if money:
        amount = float(value)
        for unit, divisor in (("B", 1e9), ("M", 1e6)):
            if abs(amount) >= divisor:
                return f"${amount / divisor:,.2f}{unit}"
        return f"${amount:,.0f}"
    if isinstance(value, float):
        return f"{value:,.2f}"
    return html.escape(str(value))


def earnings(ticker: str) -> str:
    result = _result(ticker)
    earnings_module = result["modules"]["earnings"]
    expectations = result["modules"]["expectations"]
    sec = result["modules"]["sec"]
    upcoming = expectations.get("details", {}).get("upcoming_earnings") or {}
    return "<br>".join([
        f"<b>💰 {html.escape(ticker)} · 财报与 SEC</b>",
        f"状态 <code>{earnings_module['status']}</code> · {html.escape(earnings_module['summary'])}",
        f"下次财报 <code>{_fmt(upcoming.get('release_date'))}</code> · {_fmt(upcoming.get('when'))}",
        f"最近 EPS Surprise <code>{_fmt(earnings_module['metrics'].get('latest_eps_surprise'), pct=True)}</code>",
        f"最近 Revenue Surprise <code>{_fmt(earnings_module['metrics'].get('latest_revenue_surprise'), pct=True)}</code>",
        f"SEC 文件 <code>{_fmt(sec['metrics'].get('filing_count'))}</code> · SEC直接索引 <code>{_fmt(sec['metrics'].get('direct_sec_count'))}</code>",
        "<small>兼容命令：数据来自统一 Research Engine</small>",
    ])


def holders(ticker: str) -> str:
    result = _result(ticker)
    module = result["modules"]["institutional"]
    holdings = module.get("details", {}).get("institutional_holdings", [])
    lines = [
        f"<b>🏛 {html.escape(ticker)} · 机构与内部人</b>",
        html.escape(module["summary"]),
    ]
    for item in holdings[:10]:
        lines.append(
            f"• {html.escape(str(item.get('manager') or 'Unknown'))} · "
            f"{_fmt(item.get('market_value'), money=True)} · "
            f"{_fmt(item.get('portfolio_weight'), pct=True)}"
        )
    if not holdings:
        lines.append("<i>当前13F归档中没有该股票的跟踪机构持仓</i>")
    lines.append("<small>兼容命令：数据来自统一 Research Engine</small>")
    return "<br>".join(lines)


def macro() -> str:
    result = _result("SPY")
    module = result["modules"]["macro"]
    metrics = module["metrics"]
    return "<br>".join([
        "<b>🌐 美国宏观环境</b>",
        html.escape(module["summary"]),
        f"VIX <code>{_fmt(metrics.get('vix'))}</code> · DXY <code>{_fmt(metrics.get('dxy'))}</code>",
        f"美债2Y <code>{_fmt(metrics.get('dgs2'))}</code> · 美债10Y <code>{_fmt(metrics.get('dgs10'))}</code>",
        f"10Y-2Y <code>{_fmt(metrics.get('t10y2y'))}</code> · WTI <code>{_fmt(metrics.get('wti_crude'))}</code>",
        "<small>兼容命令：数据来自统一 Research Engine</small>",
    ])


def chain(ticker: str) -> tuple[str, dict]:
    result = _result(ticker)
    return format_chain_result(result)


def format_chain_result(result: dict) -> tuple[str, dict]:
    ticker = str(result.get("ticker") or "")
    module = result["modules"]["supply_chain"]
    relationships = module.get("details", {}).get("relationships", [])
    labels = {"supplier": "供应商", "customer": "客户", "smaller_peer": "同业", "beneficiary": "间接受益"}
    lines = [f"<b>🔗 {html.escape(ticker)} · 产业链</b>", html.escape(module["summary"])]
    for item in relationships[:20]:
        evidence = "✓" if item.get("verified") else "⚠"
        lines.append(
            f"{evidence} {html.escape(labels.get(str(item.get('relationship_type')), str(item.get('relationship_type'))))} · "
            f"<b>{html.escape(str(item.get('target_company') or '—'))}</b> · "
            f"{html.escape(str(item.get('description') or ''))}"
        )
    if not relationships:
        lines.append("<i>本次没有识别到可靠产业关系</i>")
    lines.append("<small>兼容命令：数据来自统一 Research Engine</small>")
    return "<br>".join(lines), {"relationships": relationships, "research_run_id": result["run_id"]}
