"""Recorded tool outputs — lets the loop run with no data-provider keys.

Why fixtures live in the repo
-----------------------------
Two problems have the same solution. An agent whose tools hit the network cannot
be regression-tested (the same query gives different answers next Tuesday), and
a reader who clones this repo has no Financial Datasets or Alpaca key. Canned
observations fix both: deterministic evaluation, and a demo that costs only the
LLM call.

These are hand-written in the shape the real responders emit, not captured from
a live account — the numbers are synthetic and internally consistent so that the
grounding check has something real to verify against. The interface is identical
to the live executor, so replacing this file with genuinely recorded output is a
drop-in change.

The book is built so the interesting question has a non-obvious answer: NVDA is
the largest *mover*, but CRWD is the largest *risk* (top weight, earnings in
three days, insider selling). A single-hop router cannot get there.
"""

from __future__ import annotations

from typing import Any


class SimulatedToolFailure(RuntimeError):
    """Raised by a fixture to exercise the error-as-observation path."""


_PORTFOLIO = """💼 <b>持仓总览</b>（Alpaca paper · 2026-09-03 收盘）
总市值 $184,320.55 · 现金 $12,480.10

| Ticker | 数量 | 市值 | 权重 | 未实现盈亏 |
|---|---|---|---|---|
| CRWD  | 92  | $41,287.00 | 22.4% | +$3,910.20 (+10.5%) |
| NVDA  | 148 | $33,546.32 | 18.2% | +$7,204.88 (+27.3%) |
| MSFT  | 51  | $25,989.21 | 14.1% | +$1,802.44 (+7.4%) |
| AMD   | 121 | $20,828.23 | 11.3% | -$1,447.10 (-6.5%) |
| TSLA  | 63  | $17,879.09 | 9.7%  | -$2,013.77 (-10.1%) |
| AAPL  | 74  | $16,588.85 | 9.0%  | +$986.33 (+6.3%) |
| SMCI  | 210 | $15,851.57 | 8.6%  | -$4,332.61 (-21.5%) |
| GOOGL | 58  | $12,349.28 | 6.7%  | +$451.06 (+3.8%) |"""

_RISK = """💼 <b>组合风险</b>（2026-09-03）

<b>集中度</b>
· 前 3 大持仓合计 54.7%（阈值 50% ⚠️ 超标）
· 最大单一持仓 CRWD 22.4%（阈值 20% ⚠️ 超标）
· HHI 0.147

<b>行业暴露</b>
· BROAD（大盘 ETF）0.0%
· 半导体 38.1%（NVDA + AMD + SMCI）
· 软件/安全 36.5%（CRWD + MSFT）
· 消费/其他 25.4%

<b>回撤</b>
· 组合当前回撤 -4.20%（峰值 2026-08-14）
· 最大成分回撤 SMCI -21.5%

<b>7 日内财报</b>
· 2 只持仓将在 7 天内发布财报（详情用 earnings_calendar 查询）"""

_EARNINGS: dict[str, str] = {
    "CRWD": """📞 <b>CRWD 财报</b>
· 下次财报：2026-09-06（盘后）— <b>距今 3 天</b>
· 上次：2026-06-04，EPS $1.04 vs 预期 $0.98（beat +6.1%）
· 营收 $1.29B vs 预期 $1.26B（beat +2.4%）
· 财报后次日股价 -8.30%（历史 4 次财报平均绝对波动 9.10%）""",
    "SMCI": """📞 <b>SMCI 财报</b>
· 下次财报：2026-09-09（盘后）— 距今 6 天
· 上次：2026-06-11，EPS $0.42 vs 预期 $0.55（miss -23.6%）
· 营收 $5.10B vs 预期 $5.48B（miss -6.9%）
· 财报后次日股价 -14.20%""",
    "NVDA": """📞 <b>NVDA 财报</b>
· 下次财报：2026-11-19（盘后）— 距今 77 天
· 上次：2026-08-27，EPS $1.31 vs 预期 $1.24（beat +5.6%）""",
    "MSFT": "📞 <b>MSFT 财报</b>\n· 下次财报：2026-10-28 — 距今 55 天\n· 上次：2026-07-29，EPS $3.42 vs 预期 $3.35（beat +2.1%）",
    "AMD":  "📞 <b>AMD 财报</b>\n· 下次财报：2026-11-03 — 距今 61 天\n· 上次：2026-08-05，EPS $0.71 vs 预期 $0.69（beat +2.9%）",
    "TSLA": "📞 <b>TSLA 财报</b>\n· 下次财报：2026-10-21 — 距今 48 天\n· 上次：2026-07-22，EPS $0.44 vs 预期 $0.51（miss -13.7%）",
    "AAPL": "📞 <b>AAPL 财报</b>\n· 下次财报：2026-10-29 — 距今 56 天\n· 上次：2026-07-31，EPS $1.58 vs 预期 $1.54（beat +2.6%）",
    "GOOGL": "📞 <b>GOOGL 财报</b>\n· 下次财报：2026-10-27 — 距今 54 天\n· 上次：2026-07-23，EPS $2.31 vs 预期 $2.18（beat +6.0%）",
    "_": "📞 该 ticker 无财报记录。",
}

_INSIDERS: dict[str, str] = {
    "CRWD": """📥 <b>CRWD 内部人交易</b>（过去 90 天，SEC Form 4）
· 2026-08-21 CFO 卖出 18,000 股 @ $442.10（$7.96M）
· 2026-08-19 CEO 卖出 25,000 股 @ $438.75（$10.97M）
· 2026-08-12 SVP Eng 卖出 6,400 股 @ $431.20（$2.76M）
⚠️ 90 天内 3 次卖出、0 次买入 — 达到集群卖出阈值（≥3）""",
    "SMCI": "📥 <b>SMCI 内部人交易</b>（过去 90 天）\n· 无 Form 4 记录。",
    "NVDA": "📥 <b>NVDA 内部人交易</b>（过去 90 天）\n· 2026-08-04 董事卖出 12,000 股 @ $221.40（预设 10b5-1 计划，噪音）",
    "_": "📥 过去 90 天无 Form 4 记录。",
}


def _eight_k(args: dict[str, Any]) -> str:
    """8-K fixture. SMCI deliberately fails, to exercise recovery in the demo."""
    ticker = str(args.get("ticker", "")).upper()
    if ticker == "SMCI":
        raise SimulatedToolFailure("EDGAR request timed out after 30s (simulated)")
    table = {
        "CRWD": ("📋 <b>CRWD 8-K</b>（近 30 天）\n"
                 "· 2026-08-25 Item 5.02 — CFO 宣布将于 2026-10-01 离任\n"
                 "· 2026-08-11 Item 1.01 — 签署 $180M 云基础设施采购协议"),
        "NVDA": "📋 <b>NVDA 8-K</b>（近 30 天）\n· 2026-08-27 Item 2.02 — 公布 FY27 Q2 业绩",
    }
    return table.get(ticker, f"📋 <b>{ticker} 8-K</b>（近 30 天）\n· 无重大事项申报。")


_EXPLAIN: dict[str, str] = {
    "NVDA": """📈 <b>NVDA 为什么动</b>
· 今日 +3.85%，成交量 1.8× 30 日均量
· 同期 SMH +2.90% → 相对强度 +0.95pp（跟随板块）
· Tier-1 归因：路透 2026-09-03「NVDA 与两家主权 AI 基金签署供货框架」""",
    "CRWD": """📉 <b>CRWD 为什么动</b>
· 今日 -2.10%，成交量 2.4× 30 日均量
· 同期 XLK +0.60% → 相对强度 -2.70pp ★ 逆势
· Tier-1 归因：Bloomberg 2026-09-02「CFO 离任叠加财报临近，卖方下调短期评级」""",
    "SMCI": """📉 <b>SMCI 为什么动</b>
· 今日 -5.40%，成交量 3.1× 30 日均量
· 同期 SMH +2.90% → 相对强度 -8.30pp ★ 逆势
· Tier-2 归因：Seeking Alpha「渠道调研指出 AI 服务器订单递延」（单一来源，未交叉验证）""",
    "_": "📊 该 ticker 近期无显著异动。",
}

_PNL: dict[str, str] = {
    "day":   "📊 <b>当日盈亏</b>\n· 今日 +$1,204.33（+0.65%）\n· 总权益 $196,800.65",
    "week":  """📊 <b>本周盈亏</b>（2026-08-31 ~ 09-03）
· 合计 -$3,880.12（-1.93%）
贡献分解：
· SMCI -$2,940.55
· TSLA -$1,502.30
· CRWD -$880.44
· NVDA +$1,443.17""",
    "month": "📊 <b>本月盈亏</b>\n· 合计 -$6,120.70（-3.02%）",
    "_":     "📊 无该周期数据。",
}

_MONEYFLOW: dict[str, str] = {
    "CRWD": "💧 <b>CRWD 资金流</b>\n· CMF(20) -0.18 → 资金净流出\n· RSI(14) 41.2，价格新低但 RSI 未新低 → 底背离（未确认）",
    "SMCI": "💧 <b>SMCI 资金流</b>\n· CMF(20) -0.31 → 持续流出\n· RSI(14) 28.4 → 超卖区",
    "_":    "💧 该 ticker 资金流数据不足。",
}


PORTFOLIO_FIXTURES: dict[str, Any] = {
    "portfolio_view": _PORTFOLIO,
    "risk_view": _RISK,
    "pnl_view": _PNL["day"],
    "pnl_period": _PNL,
    "earnings_view": _EARNINGS,
    "earnings_calendar": ("📅 <b>未来 14 天财报</b>\n"
                          "· 2026-09-06 CRWD（持仓 22.4%）\n"
                          "· 2026-09-09 SMCI（持仓 8.6%）"),
    "eight_k_view": _eight_k,
    "insider_view": _INSIDERS,
    "explain_move": _EXPLAIN,
    "moneyflow_view": _MONEYFLOW,
    "summary": {"_": "📊 综合概览：数据源在 fixture 模式下未记录该 ticker。"},
    "macro_view": ("🌐 <b>宏观</b>\n· VIX 18.40（+1.20）\n· 10Y 4.18%\n"
                   "· DXY 101.30 · WTI $71.60 · 黄金 $3,410"),
    "release_check": {"cpi": "📈 <b>CPI</b>\n· 2026-08 同比 2.70%（前值 2.90%）\n· 核心 3.10%",
                      "_": "📈 该序列 fixture 未记录。"},
    "watchlist_view": "关注列表：ARM, PLTR, AVGO",
    "alert_list": "🔔 当前无未触发提醒。",
    "settings_view": "推送阈值：涨跌 ≥3.0%，量比 ≥2.5×",
    "holders": {"_": "🏛 fixture 模式未记录机构持仓。"},
    "institutional_13f": {"_": "🏛 fixture 模式未记录 13F。"},
    "etf_view": {"_": "🚀 fixture 模式未记录 ARK 持仓。"},
    "chain": {"_": "🔗 fixture 模式未记录产业链。"},
}


# --- B1: anomalies whose cause lives in filings rather than in the news -------
#
# ARM is the case the top-up exists for: a real 8-K explains the move, but the
# news search that runs first found nothing quotable. SMCI keeps its simulated
# timeout so the failure path stays exercised here too.

_ANOMALY_EXTRA: dict[str, Any] = {
    "ARM": ("📋 <b>ARM 8-K</b>（近 30 天）\n"
            "· 2026-09-03 Item 1.01 — 与一家超大规模云厂商签署为期 5 年、"
            "总额 $2.40B 的架构授权协议\n"
            "· 2026-08-28 Item 5.02 — 任命新任 COO"),
    "PLTR": "📋 <b>PLTR 8-K</b>（近 30 天）\n· 无重大事项申报。",
}

_ANOMALY_INSIDERS: dict[str, str] = {
    "ARM": "📥 <b>ARM 内部人交易</b>（过去 90 天）\n· 无 Form 4 记录。",
    "PLTR": ("📥 <b>PLTR 内部人交易</b>（过去 90 天）\n"
             "· 2026-09-02 CEO 卖出 400,000 股 @ $38.20（$15.28M）\n"
             "· 2026-08-29 CTO 卖出 120,000 股 @ $37.60（$4.51M）\n"
             "⚠️ 90 天内 2 次卖出、0 次买入"),
}

_ANOMALY_EARNINGS: dict[str, str] = {
    "ARM": "📞 <b>ARM 财报</b>\n· 下次财报：2026-11-05 — 距今 63 天",
    "PLTR": "📞 <b>PLTR 财报</b>\n· 下次财报：2026-11-03 — 距今 61 天",
}

_ANOMALY_FLOW: dict[str, str] = {
    "ARM": "💧 <b>ARM 资金流</b>\n· CMF(20) +0.22 → 资金净流入\n· RSI(14) 68.5",
    "PLTR": "💧 <b>PLTR 资金流</b>\n· CMF(20) -0.14 → 资金流出\n· RSI(14) 44.1",
}


def _anomaly_eight_k(args: dict[str, Any]) -> str:
    ticker = str(args.get("ticker", "")).upper()
    if ticker in _ANOMALY_EXTRA:
        return _ANOMALY_EXTRA[ticker]
    return _eight_k(args)


ANOMALY_FIXTURES: dict[str, Any] = {
    **PORTFOLIO_FIXTURES,
    "eight_k_view": _anomaly_eight_k,
    "insider_view": {**_INSIDERS, **_ANOMALY_INSIDERS},
    "earnings_view": {**_EARNINGS, **_ANOMALY_EARNINGS},
    "moneyflow_view": {**_MONEYFLOW, **_ANOMALY_FLOW},
}
