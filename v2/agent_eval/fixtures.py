"""Recorded observations for the evaluation set.

Extends the demo fixtures with enough breadth that ~80 labelled questions have
checkable answers: every ticker in the book gets earnings, insider, 8-K, flow and
move-attribution cards, plus institutional, ETF, chain and macro coverage.

Two rules kept these usable as an answer key:

* **Internally consistent.** Weights sum, P&L contributions reconcile with the
  positions, earnings dates agree between the per-ticker card and the calendar.
  A case can therefore assert a fact and be sure exactly one right answer exists.
* **Distinctive numbers.** Values are deliberately not round, so a substring
  check for "22.4%" cannot pass by coincidence the way "10%" could.
"""

from __future__ import annotations

from typing import Any

from v2.agent_eval.base_fixtures import (
    PORTFOLIO_FIXTURES,
    SimulatedToolFailure,
    _EARNINGS,
    _EXPLAIN,
    _INSIDERS,
    _MONEYFLOW,
    _PNL,
)

# --- price / move attribution -----------------------------------------------

_EXPLAIN_EXTRA: dict[str, str] = {
    "AMD": ("📉 <b>AMD 为什么动</b>\n· 今日 -1.32%，成交量 1.1× 30 日均量\n"
            "· 同期 SMH +2.90% → 相对强度 -4.22pp ★ 逆势\n"
            "· Tier-2 归因：MarketWatch「数据中心 GPU 竞争加剧」"),
    "TSLA": ("📉 <b>TSLA 为什么动</b>\n· 今日 -2.74%，成交量 1.6× 30 日均量\n"
             "· 同期 SPY +0.40% → 相对强度 -3.14pp ★ 逆势\n"
             "· Tier-1 归因：Reuters「欧洲 8 月交付量同比 -14%」"),
    "AAPL": ("📈 <b>AAPL 为什么动</b>\n· 今日 +0.86%，成交量 0.9× 30 日均量\n"
             "· 同期 XLK +0.60% → 相对强度 +0.26pp（跟随板块）\n"
             "· 无 Tier-1/2 归因，属常规波动"),
    "MSFT": ("📈 <b>MSFT 为什么动</b>\n· 今日 +1.14%，成交量 1.0× 30 日均量\n"
             "· 同期 XLK +0.60% → 相对强度 +0.54pp\n"
             "· Tier-2 归因：Investors「Azure 增速指引上修」"),
    "GOOGL": ("📈 <b>GOOGL 为什么动</b>\n· 今日 +0.42%，成交量 0.8× 30 日均量\n"
              "· 同期 XLK +0.60% → 相对强度 -0.18pp\n· 无显著归因"),
    "ARM": ("📈 <b>ARM 为什么动</b>\n· 今日 +7.42%，成交量 3.4× 30 日均量\n"
            "· 同期 SMH +2.90% → 相对强度 +4.52pp\n"
            "· Tier-1 归因：8-K 披露 $2.40B 架构授权协议"),
    "PLTR": ("📉 <b>PLTR 为什么动</b>\n· 今日 -5.18%，成交量 2.9× 30 日均量\n"
             "· 同期 SPY +0.40% → 相对强度 -5.58pp ★ 逆势\n"
             "· 无 Tier-1/2 归因"),
    "AVGO": ("📈 <b>AVGO 为什么动</b>\n· 今日 +2.16%，成交量 1.3× 30 日均量\n"
             "· 同期 SMH +2.90% → 相对强度 -0.74pp"),
}

_EARNINGS_EXTRA: dict[str, str] = {
    "ARM": "📞 <b>ARM 财报</b>\n· 下次财报：2026-11-05 — 距今 63 天\n· 上次：2026-08-06，EPS $0.39 vs 预期 $0.36（beat +8.3%）",
    "PLTR": "📞 <b>PLTR 财报</b>\n· 下次财报：2026-11-03 — 距今 61 天\n· 上次：2026-08-04，EPS $0.19 vs 预期 $0.18（beat +5.6%）",
    "AVGO": "📞 <b>AVGO 财报</b>\n· 下次财报：2026-09-11（盘后）— 距今 8 天\n· 上次：2026-06-05，EPS $1.71 vs 预期 $1.66（beat +3.0%）",
}

_INSIDERS_EXTRA: dict[str, str] = {
    "AMD": "📥 <b>AMD 内部人交易</b>（过去 90 天）\n· 2026-07-18 SVP 卖出 9,500 股 @ $168.30（$1.60M，10b5-1 计划）",
    "TSLA": "📥 <b>TSLA 内部人交易</b>（过去 90 天）\n· 无 Form 4 记录。",
    "AAPL": "📥 <b>AAPL 内部人交易</b>（过去 90 天）\n· 2026-08-15 COO 卖出 41,000 股 @ $224.10（$9.19M，10b5-1 计划）",
    "MSFT": "📥 <b>MSFT 内部人交易</b>（过去 90 天）\n· 无 Form 4 记录。",
    "GOOGL": "📥 <b>GOOGL 内部人交易</b>（过去 90 天）\n· 无 Form 4 记录。",
    "AVGO": "📥 <b>AVGO 内部人交易</b>（过去 90 天）\n· 2026-08-22 CEO 卖出 30,000 股 @ $198.40（$5.95M）",
}

_MONEYFLOW_EXTRA: dict[str, str] = {
    "NVDA": "💧 <b>NVDA 资金流</b>\n· CMF(20) +0.26 → 资金净流入\n· RSI(14) 63.7",
    "AMD": "💧 <b>AMD 资金流</b>\n· CMF(20) -0.09 → 小幅流出\n· RSI(14) 46.8",
    "TSLA": "💧 <b>TSLA 资金流</b>\n· CMF(20) -0.21 → 资金流出\n· RSI(14) 38.9",
    "AAPL": "💧 <b>AAPL 资金流</b>\n· CMF(20) +0.11 → 小幅流入\n· RSI(14) 55.2",
    "MSFT": ("💧 <b>MSFT 资金流</b>\n· CMF(20) +0.14 → 资金流入\n· RSI(14) 58.4\n"
             "· 距 52 周高点 -3.10%，位于 200 日均线上方"),
}

_EIGHT_K: dict[str, str] = {
    "CRWD": ("📋 <b>CRWD 8-K</b>（近 30 天）\n"
             "· 2026-08-25 Item 5.02 — CFO 宣布将于 2026-10-01 离任\n"
             "· 2026-08-11 Item 1.01 — 签署 $180M 云基础设施采购协议"),
    "NVDA": "📋 <b>NVDA 8-K</b>（近 30 天）\n· 2026-08-27 Item 2.02 — 公布 FY27 Q2 业绩",
    "ARM": ("📋 <b>ARM 8-K</b>（近 30 天）\n"
            "· 2026-09-03 Item 1.01 — 与一家超大规模云厂商签署为期 5 年、总额 $2.40B 的架构授权协议\n"
            "· 2026-08-28 Item 5.02 — 任命新任 COO"),
    "TSLA": "📋 <b>TSLA 8-K</b>（近 30 天）\n· 2026-08-19 Item 5.07 — 年度股东大会表决结果",
    "AMD": "📋 <b>AMD 8-K</b>（近 30 天）\n· 无重大事项申报。",
    "AAPL": "📋 <b>AAPL 8-K</b>（近 30 天）\n· 无重大事项申报。",
    "MSFT": "📋 <b>MSFT 8-K</b>（近 30 天）\n· 无重大事项申报。",
    "GOOGL": "📋 <b>GOOGL 8-K</b>（近 30 天）\n· 无重大事项申报。",
    "AVGO": "📋 <b>AVGO 8-K</b>（近 30 天）\n· 2026-08-30 Item 1.01 — 完成一笔 $4.10B 软件资产收购",
    "PLTR": "📋 <b>PLTR 8-K</b>（近 30 天）\n· 无重大事项申报。",
}


def _eight_k(args: dict[str, Any]) -> str:
    """SMCI still times out — the eval must include a broken data source."""
    ticker = str(args.get("ticker", "")).upper()
    if ticker == "SMCI":
        raise SimulatedToolFailure("EDGAR request timed out after 30s (simulated)")
    return _EIGHT_K.get(ticker, f"📋 <b>{ticker} 8-K</b>（近 30 天）\n· 无重大事项申报。")


_SUMMARY: dict[str, str] = {
    "CRWD": ("📊 <b>CRWD 综合</b>\n· 价格 $448.77（今日 -2.10%）\n"
             "· 下次财报 2026-09-06，距今 3 天\n· 90 天内 3 次内部人卖出\n"
             "· 8-K：CFO 将于 2026-10-01 离任"),
    "NVDA": ("📊 <b>NVDA 综合</b>\n· 价格 $226.66（今日 +3.85%）\n"
             "· 上次财报 beat +5.6%，下次 2026-11-19\n· CMF +0.26，资金流入"),
    "SMCI": ("📊 <b>SMCI 综合</b>\n· 价格 $75.48（今日 -5.40%）\n"
             "· 上次财报 EPS miss -23.6%，次日 -14.20%\n· 下次财报 2026-09-09"),
    "_": "📊 fixture 模式未记录该 ticker 的综合概览。",
}

_HOLDERS: dict[str, str] = {
    "NVDA": ("🏛 <b>NVDA 机构持有</b>（最新 13F）\n"
             "· Vanguard 8.94% · BlackRock 7.31% · Fidelity 4.62%\n· Coatue 新建仓 1.20%"),
    "CRWD": ("🏛 <b>CRWD 机构持有</b>（最新 13F）\n"
             "· Vanguard 7.88% · BlackRock 6.45% · Tiger Global 2.10%"),
    "_": "🏛 fixture 模式未记录该 ticker 的机构持仓。",
}

_THIRTEEN_F: dict[str, str] = {
    "buffett": ("🏛 <b>Berkshire Hathaway</b> 2026-Q2 13F\n"
                "· AAPL $57.80B（22.0%）· BAC $31.20B · KO $27.40B\n"
                "· 本季减持 AAPL 6.2%，新建仓 CVX"),
    "burry": ("🏛 <b>Scion Asset Management</b> 2026-Q2 13F\n"
              "· 组合仅 9 个持仓，合计 $84.30M\n· 新建仓 BABA、JD；清仓 NVDA"),
    "ark": ("🏛 <b>ARK Investment Management</b> 2026-Q2 13F\n"
            "· TSLA 9.80% · COIN 7.20% · ROKU 4.10%"),
    "_": "🏛 fixture 模式未记录该 manager。",
}

_ETF: dict[str, str] = {
    "ARKK": ("🚀 <b>ARKK</b> 每日持仓（2026-09-03）\n"
             "· 前三：TSLA 9.80% · COIN 7.20% · ROKU 4.10%\n"
             "· 今日买入 PLTR 42,000 股；卖出 PATH 118,000 股"),
    "_": "🚀 fixture 模式未记录该 ETF。",
}

_CHAIN: dict[str, str] = {
    "NVDA": ("🔗 <b>NVDA 产业链</b>\n· 上游：TSM（代工）· ASML（光刻）\n"
             "· 下游：MSFT / META（云采购）\n· 同业：AMD · AVGO"),
    "_": "🔗 fixture 模式未记录该 ticker 的产业链。",
}

_RELEASES: dict[str, str] = {
    "cpi": "📈 <b>CPI</b>\n· 2026-08 同比 2.70%（前值 2.90%）\n· 核心 3.10%",
    "pce": "📈 <b>PCE</b>\n· 2026-07 同比 2.40%（前值 2.50%）\n· 核心 2.80%",
    "nfp": "📈 <b>NFP</b>\n· 2026-08 新增 14.2 万人（前值 18.7 万）\n· 失业率 4.30%",
    "fomc": "📈 <b>FOMC</b>\n· 2026-07-29 维持 4.25-4.50% 不变\n· 下次会议 2026-09-17",
    "_": "📈 该序列 fixture 未记录。",
}


EVAL_FIXTURES: dict[str, Any] = {
    **PORTFOLIO_FIXTURES,
    "explain_move": {**_EXPLAIN, **_EXPLAIN_EXTRA},
    "earnings_view": {**_EARNINGS, **_EARNINGS_EXTRA},
    "insider_view": {**_INSIDERS, **_INSIDERS_EXTRA},
    "moneyflow_view": {**_MONEYFLOW, **_MONEYFLOW_EXTRA},
    "eight_k_view": _eight_k,
    "summary": _SUMMARY,
    "holders": _HOLDERS,
    "institutional_13f": _THIRTEEN_F,
    "etf_view": _ETF,
    "chain": _CHAIN,
    "release_check": _RELEASES,
    "pnl_period": _PNL,
    # Date first, ticker after — the layout the live card uses, and the reason a
    # calendar in which every row was correct came back as four
    # misattributions: each day-of-month parses as a negative number and lands
    # in the *previous* row's ownership window. A fixture that avoids the
    # awkward shape cannot measure whether the check survives it.
    "earnings_calendar": ("📅 <b>未来 14 天财报</b>\n"
                          "· 2026-09-06 (D-3) CRWD（持仓 22.4%）\n"
                          "· 2026-09-09 (D-6) SMCI（持仓 8.6%）\n"
                          "· 2026-09-11 (D-8) AVGO（关注列表）\n"
                          "· 09-30 (D-27) MSFT（持仓 14.1%）\n"
                          "· 10-21 (D-48) AMD（持仓 11.3%）"),
}
