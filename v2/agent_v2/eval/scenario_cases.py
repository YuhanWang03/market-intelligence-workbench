"""Scenario cases: two-turn conversations the coordinator must carry end to end.

A scenario is the seed set's answer to "does the follow-up still refer to
the right thing": the same fixtures every turn, no network, no model.  The
first scenario is the drawdown conversation that drove the filing reader
("哪只跌得最多" then "为什么跌这么狠"); its expectations pin the plan the
second turn gets, the sentences the deterministic answer must open with,
and that the sub-agent's events and the news reach the answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from v2.agent_v2.catalog import default_catalog
from v2.agent_v2.execution import CapabilityRegistry
from v2.agent_v2.models import EvidenceItem, ResultStatus, ToolEnvelope
from v2.agent_v2.orchestrator import AgentV2, AgentV2Config
from v2.agent_v2.session import ShortTermSession

#: The position card behind every scenario: ARM is the biggest loser since purchase.
PORTFOLIO_CARD = """<b>💼 Alpaca 账户 · 📝 PAPER</b>
━━━━━━━━━━━━━━━━━━━━
组合价值 <code>$100,754</code>  ·  现金 <code>$18,877</code>

<b>持仓（4）</b>
  🟢 <b>IVV</b>  <code>70 sh</code>  @ <code>$755.41</code>
     市值 <code>$53,951</code>  ·  P/L <code>$1,073</code> <b>+2.03%</b>
  🔴 <b>BRK.B</b>  <code>10 sh</code>  @ <code>$505.87</code>
     市值 <code>$5,050</code>  ·  P/L <code>$8</code> <b>-0.17%</b>
  🔴 <b>ARM</b>  <code>5 sh</code>  @ <code>$389.52</code>
     市值 <code>$1,320</code>  ·  P/L <code>$628</code> <b>-32.22%</b>
  🔴 <b>MRVL</b>  <code>5 sh</code>  @ <code>$310.82</code>
     市值 <code>$1,129</code>  ·  P/L <code>$425</code> <b>-27.33%</b>"""


def build_drawdown_registry() -> CapabilityRegistry:
    from v2.agent_v2.adapters.legacy import _wrap

    registry = CapabilityRegistry(default_catalog())
    registry.register("account.portfolio", lambda a, c: _wrap("account.portfolio", "portfolio", PORTFOLIO_CARD))

    def performance(a, c):
        from v2.agent_v2.adapters.market import _INTRADAY_PRICE_RULE

        ticker = a["ticker"]
        price = EvidenceItem(f"P-{ticker}", ticker, f"{ticker} 截至 2026-09-09 13:42 ET 的盘中价格为 264.00 美元，相对前一交易日收盘价 +0.94%。", metadata={"evidence_scope": "price", "is_intraday": True, "constraints": [_INTRADAY_PRICE_RULE]})
        windows = EvidenceItem(f"W-{ticker}", ticker, f"{ticker} 截至查询时的区间回报（含当前盘中价格）：1d +0.94%，5d +12.32%，1m -1.53%，3m -21.40%，1y +89.85%。", metadata={"evidence_scope": "returns"})
        benchmark = EvidenceItem(f"B-{ticker}", ticker, f"{ticker} 相对 SMH：1d +0.94%。", metadata={"evidence_scope": "benchmark"})
        narrative = f"{ticker} 最近的股价表现分化。近 5 日回报 +12.32%，近 1 月回报 -1.53%[W-{ticker}]。\n相对 SMH，单日超额 +0.94%[B-{ticker}]。\n成交量尚未定型。"
        return ToolEnvelope("market.performance", ResultStatus.COMPLETED, subject=ticker, summary=f"{ticker} 近 1 月 -1.53%", metrics={"returns": {"1d": 0.0094, "5d": 0.1232, "1m": -0.0153, "3m": -0.2140, "1y": 0.8985}, "is_intraday": True}, evidence=[price, windows, benchmark], metadata={"narrative": narrative, "require_cited_numbers": True})

    registry.register("market.performance", performance)
    registry.register("market.explain_move", lambda a, c: ToolEnvelope("market.explain_move", ResultStatus.COMPLETED, subject=a["ticker"], summary=f"{a['ticker']} 异动", metrics={"price_change_pct": 0.0094}, evidence=[EvidenceItem(f"M-{a['ticker']}", a["ticker"], f"{a['ticker']} 今日 +0.94%", metadata={"evidence_scope": "price"})]))
    registry.register(
        "research.stock",
        lambda a, c: ToolEnvelope(
            "research.stock",
            ResultStatus.COMPLETED,
            subject=a["ticker"],
            summary=f"{a['ticker']}: the evidence currently balances growth against valuation risk.",
            evidence=[
                EvidenceItem(f"R-{a['ticker']}-{a['focus']}", a["ticker"], f"{a['ticker']} 2026-08-05 发布财报，指引低于预期。"),
                EvidenceItem(f"R-{a['ticker']}-fundamental", a["ticker"], "Revenue growth is +22.8% on the latest available basis."),
                EvidenceItem(f"R-{a['ticker']}-metrics", a["ticker"], "派生评分", metadata={"citation_kind": "metrics"}),
            ],
        ),
    )

    def drawdown(a, c):
        ticker = a["ticker"]
        window = EvidenceItem(f"D-{ticker}-window", ticker, f"{ticker} 近 3 月（2026-06-10 至 2026-09-08）区间回报 -21.40%。", metadata={"evidence_scope": "window_return"})
        worst = EvidenceItem(f"D-{ticker}-0805", ticker, f"{ticker} 2026-08-05 单日 -13.21%，收盘 275.10 美元。", metadata={"evidence_scope": "worst_day", "date": "2026-08-05"})
        peak = EvidenceItem(f"D-{ticker}-peak", ticker, f"{ticker} 从 2026-06-18 的高点 439.46 美元到 2026-08-05 的低点 275.10 美元回撤 -37.40%。", metadata={"evidence_scope": "peak_trough", "peak_date": "2026-06-18", "trough_date": "2026-08-05"})
        narrative = f"{ticker} 近 3 月（2026-06-10 至 2026-09-08）区间回报 -21.40%[D-{ticker}-window]。\n{peak.claim.rstrip('。')}[D-{ticker}-peak]。\n{ticker} 近 3 月跌幅最大的交易日：2026-08-05 -13.21%[D-{ticker}-0805]。"
        return ToolEnvelope("market.drawdown", ResultStatus.COMPLETED, subject=ticker, metrics={"window": "3m", "window_start": "2026-06-10", "window_return": -0.2140, "worst_days": [{"date": "2026-08-05", "return": -0.1321}], "peak": {"date": "2026-06-18", "close": 439.46}, "trough": {"date": "2026-08-05", "close": 275.10}, "drawdown": -0.3740}, evidence=[window, peak, worst], metadata={"narrative": narrative, "require_cited_numbers": True, "worst_dates": ["2026-08-05"], "queries": [f"why did {ticker} stock fall on 2026-08-05"]})

    registry.register("market.drawdown", drawdown)

    def runup(a, c):
        ticker = a["ticker"]
        window = EvidenceItem(f"U-{ticker}-window", ticker, f"{ticker} 近 3 月（2026-06-10 至 2026-09-08）区间回报 +6.20%。", metadata={"evidence_scope": "window_return"})
        best = EvidenceItem(f"U-{ticker}-0512", ticker, f"{ticker} 2026-05-12 单日 +3.50%，收盘 730.00 美元。", metadata={"evidence_scope": "best_day", "date": "2026-05-12"})
        low_high = EvidenceItem(f"U-{ticker}-peak", ticker, f"{ticker} 从 2026-04-07 的低点 700.00 美元到 2026-07-10 的高点 760.00 美元上涨 +8.57%。", metadata={"evidence_scope": "peak_trough", "peak_date": "2026-07-10", "trough_date": "2026-04-07"})
        span = EvidenceItem(f"U-{ticker}-span", ticker, f"同期行业基准 SPY 从 2026-04-07 到 2026-07-10 回报 +6.00%，{ticker} 比基准多涨 2.57 个百分点。", metadata={"evidence_scope": "benchmark_span", "benchmark": "SPY"})
        narrative = f"{window.claim.rstrip('。')}[{window.id}]。\n{low_high.claim.rstrip('。')}[{low_high.id}]。\n{span.claim.rstrip('。')}[{span.id}]。\n{ticker} 近 3 月涨幅最大的交易日：2026-05-12 +3.50%[{best.id}]。"
        return ToolEnvelope(
            "market.runup",
            ResultStatus.COMPLETED,
            subject=ticker,
            metrics={"window": "3m", "window_start": "2026-06-10", "window_return": 0.062, "best_days": [{"date": "2026-05-12", "return": 0.035}], "peak": {"date": "2026-07-10", "close": 760.0}, "trough": {"date": "2026-04-07", "close": 700.0}, "runup": 0.0857, "move": 0.0857},
            evidence=[window, low_high, span, best],
            metadata={"narrative": narrative, "require_cited_numbers": True, "direction": "up", "best_dates": ["2026-05-12"], "dates": ["2026-05-12"], "answer_constraints": [{"require_cited": {"metadata": {"evidence_scope": "benchmark_span"}, "warning": f"这段涨幅的回答必须引用同期行业基准对比那条证据 [{span.id}]"}}]},
        )

    registry.register("market.runup", runup)
    registry.register(
        "filings.recent",
        lambda a, c: ToolEnvelope(
            "filings.recent",
            ResultStatus.COMPLETED,
            subject=a["ticker"],
            evidence=[
                EvidenceItem(f"F-{a['ticker']}-0805", a["ticker"], f"{a['ticker']} 于 2026-08-05 向 SEC 提交了 8-K（0001-25-000001）。", metadata={"evidence_scope": "filing", "date": "2026-08-05"}),
                EvidenceItem(f"F-{a['ticker']}-0301", a["ticker"], f"{a['ticker']} 于 2026-03-01 向 SEC 提交了 8-K（0001-25-000000）。", metadata={"evidence_scope": "filing", "date": "2026-03-01"}),
            ],
        ),
    )
    registry.register(
        "filings.read_events",
        lambda a, c: ToolEnvelope(
            "filings.read_events",
            ResultStatus.COMPLETED,
            subject=a["ticker"],
            evidence=[EvidenceItem(f"E-{a['ticker']}-{a['around']}", a["ticker"], f"{a['ticker']} {a['around']}：财报指引低于预期（6-K {a['around']} s2：“guidance below expectations”）。", as_of=a["around"], metadata={"evidence_scope": "filing_event", "date": a["around"]})],
            metadata={"narrative": f"{a['ticker']} 申报中读到的事件：{a['around']} 财报指引低于预期[E-{a['ticker']}-{a['around']}]。", "around": a["around"]},
        ),
    )
    def attribute(a, c):
        ticker, day = a["ticker"], a["date"]
        if day == "2026-05-12":
            # The one best day of the run-up fixture: a rise with no catalyst found.
            price = EvidenceItem(f"AT-{ticker}-{day}-price", ticker, f"{ticker} 在 {day} 收于 730.00 美元，较前一交易日 +3.50%。", as_of=day, metadata={"evidence_scope": "price"})
            assessment = EvidenceItem(f"AT-{ticker}-{day}-assessment", ticker, f"{ticker} {day} 异动归因中有 0 个高置信度直接驱动，0 个候选解释。", as_of=day, metadata={"evidence_scope": "attribution", "claim_role": "attribution_assessment"})
            narrative = f"{price.claim.rstrip('。')}[{price.id}]。\n\n“为什么上涨”目前还不能下定论：暂未找到可核实的同日催化剂，具体触发原因尚未确认[{assessment.id}]。"
            rise_agent = {"name": "move_attributor", "label": "异动归因", "subject": f"{ticker} {day}", "rounds": 2, "llm_calls": 2, "elapsed_ms": 8, "seconds_allowed": 120, "stop_reason": "finished", "calls": {"news": 1 if c.allow_web else 0, "filing_events": 0, "memory": 1}, "yield": {"kept": 0, "dropped": 0, "confirmed": 0}, "reader_runs": []}
            return ToolEnvelope("market.attribute_move", ResultStatus.COMPLETED, subject=ticker, as_of=day, summary=price.claim, metrics={"price_change_pct": 0.035, "confirmed_driver_count": 0}, evidence=[price, assessment], metadata={"narrative": narrative, "narrative_compact": f"{day} {ticker} +3.50%[{price.id}]。暂未找到可核实的同日催化剂[{assessment.id}]。", "require_cited_numbers": True, "date": day, "agent": rise_agent, "trace": [{"round": 1, "action": "memory", "detail": "", "ms": 4}, {"round": 2, "action": "finish", "detail": "reasons=0", "ms": 4}]})
        price = EvidenceItem(f"AT-{ticker}-{day}-price", ticker, f"{ticker} 在 {day} 收于 275.10 美元，较前一交易日 -13.21%。", as_of=day, metadata={"evidence_scope": "price"})
        filing_event = EvidenceItem(f"E-{ticker}-{day}", ticker, f"{ticker} {day}：财报指引低于预期（6-K {day} s2：“guidance below expectations”）。", as_of=day, metadata={"evidence_scope": "filing_event", "date": day})
        reasons = [EvidenceItem(f"AT-{ticker}-{day}-filing", ticker, f"{ticker} {day} 中置信度候选解释：财报指引低于预期（申报：“guidance below expectations”）。", as_of=day, confidence=0.6, metadata={"evidence_scope": "candidate", "claim_role": "candidate_driver", "causal_confidence": "中", "driver_text": "财报指引低于预期", "source_kind": "filing"})]
        if c.allow_web:
            reasons.insert(0, EvidenceItem(f"AT-{ticker}-{day}-news", ticker, f"{ticker} {day} 高置信度归因：财报后指引令市场失望，股价大跌（新闻：“shares slid after the report as guidance disappointed”）。", as_of=day, confidence=0.9, source_url="https://example.com/arm", metadata={"evidence_scope": "driver", "claim_role": "confirmed_driver", "causal_confidence": "高", "driver_text": "财报后指引令市场失望，股价大跌", "source_kind": "news"}))
        assessment = EvidenceItem(f"AT-{ticker}-{day}-assessment", ticker, f"{ticker} {day} 异动归因中有 {1 if c.allow_web else 0} 个高置信度直接驱动，1 个候选解释。", as_of=day, metadata={"evidence_scope": "attribution", "claim_role": "attribution_assessment", "verified": True})
        first = f"{price.claim.rstrip('。')}[{price.id}]。"
        if c.allow_web:
            second = f"能直接支持的高置信度驱动：财报后指引令市场失望，股价大跌[{reasons[0].id}]。最相关的一条候选线索是“财报指引低于预期”，只能作为排查方向[{reasons[-1].id}]。"
        else:
            second = f"“为什么下跌”目前还不能下定论：暂未找到可核实的同日催化剂，具体触发原因尚未确认[{assessment.id}]。最相关的一条候选线索是“财报指引低于预期”，只能作为排查方向[{reasons[-1].id}]。"
        confirmed = 1 if c.allow_web else 0
        agent = {"name": "move_attributor", "label": "异动归因", "subject": f"{ticker} {day}", "rounds": 3 if c.allow_web else 2, "llm_calls": 3 if c.allow_web else 2, "elapsed_ms": 12, "seconds_allowed": 120, "stop_reason": "finished", "calls": {"news": 1 if c.allow_web else 0, "filing_events": 1, "memory": 1}, "yield": {"kept": len(reasons), "dropped": 0, "confirmed": confirmed}, "reader_runs": [{"filings": 1, "sections_read": 2, "events": 1, "rounds": 2, "stop_reason": "finished", "elapsed_ms": 5, "trace": []}]}
        return ToolEnvelope(
            "market.attribute_move",
            ResultStatus.COMPLETED,
            subject=ticker,
            as_of=day,
            summary=price.claim,
            metrics={"price_change_pct": -0.1321, "confirmed_driver_count": confirmed},
            evidence=[price, *reasons, filing_event, assessment],
            limitations=[] if c.allow_web else ["用户未授权网页搜索，归因未使用新闻。"],
            metadata={"narrative": first + "\n\n" + second, "require_cited_numbers": True, "date": day, "agent": agent, "trace": [{"round": 1, "action": "filing_events", "detail": "", "ms": 5}, {"round": 2, "action": "finish", "detail": f"reasons={len(reasons)}", "ms": 3}]},
        )

    registry.register("market.attribute_move", attribute)
    registry.register("market.anomaly_history", lambda a, c: ToolEnvelope("market.anomaly_history", ResultStatus.COMPLETED, subject=a["ticker"], evidence=[EvidenceItem(f"A-{a['ticker']}-0805", a["ticker"], f"{a['ticker']} 2026-08-05 盯盘记录：volume_spike,gap_down；财报后跳空低开。", metadata={"evidence_scope": "anomaly", "date": "2026-08-05"})]))
    registry.register("web.research", lambda a, c: ToolEnvelope("web.research", ResultStatus.COMPLETED, subject=a["ticker"], evidence=[EvidenceItem(f"WEB-{a['ticker']}", a["ticker"], f"{a['ticker']} shares slid after the 2026-08-05 report as guidance disappointed.", as_of="2026-08-06", source_title="Arm slides on soft guidance", source_url="https://example.com/arm", metadata={"evidence_type": "search_snippet"})]))
    return registry




@dataclass(frozen=True)
class ScenarioCase:
    id: str
    description: str
    turns: tuple[str, ...]
    #: Capabilities the last turn's plan must contain, and must not.
    expected_capabilities: frozenset[str]
    forbidden_capabilities: frozenset[str] = frozenset()
    #: Phrases the last answer must and must not contain.
    expected_phrases: tuple[str, ...] = ()
    forbidden_phrases: tuple[str, ...] = ()
    allow_web: bool = False
    #: Whether the last turn must have been rewritten from the conversation.
    expect_rewritten: bool | None = None
    expect_verified: bool = True
    #: Sub-agents that must have run (name → minimum runs), each finishing on its own.
    expected_agents: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ScenarioScore:
    case_id: str
    passed: bool
    capabilities_ok: bool
    discipline_ok: bool
    phrases_ok: bool
    rewritten_ok: bool
    verified_ok: bool
    called: tuple[str, ...] = ()
    missing_phrases: tuple[str, ...] = ()
    #: Sub-agents seen: "move_attributor×3 finished" per name; and whether they met the case's expectation.
    agents: tuple[str, ...] = ()
    agents_ok: bool = True


_DRAWDOWN_PHRASES = (
    "你问的是 ARM 买入以来的浮动盈亏：-32.22%，成本价 $389.52",
    "今日盘中为上涨（+0.94%），与买入以来的浮动盈亏是不同区间",
    "这段跌幅大部分落在近 3 月内",
    "近 1 年 +89.85% 而该持仓仍在浮亏，说明买入点在这轮上涨之后的高位",
    "ARM 期间跌幅最大的交易日：2026-08-05 -13.21%",
    "ARM 这段下跌期间 1 份申报：2026-08-05 8-K[F-ARM-0805]。",
    "ARM 在 2026-08-05 收于 275.10 美元，较前一交易日 -13.21%[AT-ARM-2026-08-05-price]。",
    "最相关的一条候选线索是“财报指引低于预期”，只能作为排查方向[AT-ARM-2026-08-05-filing]。",
)
_DRAWDOWN_TOOLS = frozenset({"account.portfolio", "market.performance", "market.drawdown", "filings.recent", "market.anomaly_history", "market.attribute_move"})

SCENARIO_CASES: tuple[ScenarioCase, ...] = (
    ScenarioCase(
        "sc_drawdown_why",
        "哪只跌得最多 → 为什么跌这么狠：追问带着买入以来的浮亏框架，走回撤归因计划",
        ("我的仓库里哪只跌的最多?", "为什么跌这么狠?"),
        _DRAWDOWN_TOOLS,
        frozenset({"market.explain_move", "web.research", "filings.read_events"}),
        _DRAWDOWN_PHRASES + ("“为什么下跌”目前还不能下定论：暂未找到可核实的同日催化剂",),
        ("组合价值", "2026-03-01", "Revenue growth", "AT-ARM-2026-08-05-news"),
        expect_rewritten=True,
        expected_agents={"move_attributor": 1},
    ),
    ScenarioCase(
        "sc_drawdown_why_web",
        "同上，用户授权网页搜索：归因者可以用新闻，给出高置信度驱动",
        ("我的仓库里哪只跌的最多?", "为什么跌这么多?"),
        _DRAWDOWN_TOOLS,
        frozenset({"market.explain_move", "web.research", "filings.read_events"}),
        _DRAWDOWN_PHRASES + ("能直接支持的高置信度驱动：财报后指引令市场失望，股价大跌[AT-ARM-2026-08-05-news]。",),
        ("组合价值", "尚未确认"),
        allow_web=True,
        expect_rewritten=True,
        expected_agents={"move_attributor": 1},
    ),
    ScenarioCase(
        "sc_direct_drawdown_question",
        "单独一句「ARM 从 6 月高点为什么跌了这么多」：措辞本身建框架，不依赖前一轮",
        ("ARM 从 6 月高点为什么跌了这么多?",),
        frozenset({"account.portfolio", "market.performance", "market.drawdown", "filings.recent", "market.anomaly_history", "market.attribute_move"}),
        frozenset({"market.explain_move", "research.stock"}),
        (
            "你问的是 ARM 这段跌幅：ARM 从 2026-06-18 的高点 439.46 美元到 2026-08-05 的低点 275.10 美元回撤 -37.40%[D-ARM-peak]。",
            "你持有该股，买入以来的浮动盈亏 -32.22%，成本价 $389.52[legacy-",
            "今日盘中为上涨（+0.94%），与这段跌幅是不同区间",
            "这段跌幅大部分落在近 3 月内",
            "ARM 期间跌幅最大的交易日：2026-08-05 -13.21%",
            "最相关的一条候选线索是“财报指引低于预期”",
        ),
        ("组合价值", "IVV 70 sh"),
        expect_rewritten=False,
        expected_agents={"move_attributor": 1},
    ),
    ScenarioCase(
        "sc_today_is_not_a_stretch",
        "「NVDA 今天为什么跌这么多」带「今天」：仍是当日异动解释",
        ("NVDA 今天为什么跌这么多?",),
        frozenset({"market.explain_move"}),
        frozenset({"market.drawdown", "market.attribute_move"}),
        expect_rewritten=False,
    ),
    ScenarioCase(
        "sc_pronoun_follow_up",
        "哪只跌得最多 → 它财报怎么样：代词接到答案点名的那只股票",
        ("我的仓库里哪只跌的最多?", "它财报怎么样"),
        frozenset({"research.stock"}),
        frozenset({"account.portfolio", "market.drawdown"}),
        ("ARM",),
        expect_rewritten=True,
    ),
    ScenarioCase(
        "sc_rise_is_not_a_drawdown",
        "哪只涨得最多 → 为什么涨这么多：浮盈持仓的追问走上涨归因链，不是回撤，也不是今日涨跌",
        ("我的仓库里哪只涨的最多?", "为什么涨这么多?"),
        frozenset({"account.portfolio", "market.performance", "market.runup", "filings.recent", "market.anomaly_history", "market.attribute_move"}),
        frozenset({"market.drawdown", "market.explain_move", "filings.read_events"}),
        (
            "你问的是 IVV 买入以来的浮动盈亏：+2.03%，成本价 $755.41",
            "这段涨幅大部分落在",
            "IVV 从 2026-04-07 的低点 700.00 美元到 2026-07-10 的高点 760.00 美元上涨 +8.57%",
            "同期行业基准 SPY 从 2026-04-07 到 2026-07-10 回报 +6.00%，IVV 比基准多涨 2.57 个百分点",
            "IVV 期间涨幅最大的交易日：2026-05-12 +3.50%",
            "“为什么上涨”目前还不能下定论",
        ),
        ("为什么下跌", "组合价值", "回撤"),
        expect_rewritten=True,
        expected_agents={"move_attributor": 1},
    ),
    ScenarioCase(
        "sc_rise_after_loss_ranking_is_todays_move",
        "哪只跌得最多 → 为什么涨：对浮亏持仓问上涨，指今日涨跌，不走回撤归因",
        ("我的仓库里哪只跌的最多?", "为什么涨"),
        frozenset({"market.explain_move"}),
        frozenset({"market.drawdown", "market.runup", "filings.read_events"}),
        expect_rewritten=True,
    ),
    ScenarioCase(
        "sc_fresh_session_has_no_referent",
        "没有上文的追问不被改写，也不假装知道在问谁",
        ("什么原因跌这么多?",),
        frozenset(),
        frozenset({"market.drawdown", "filings.read_events"}),
        forbidden_phrases=("你问的是 ARM",),
        expect_rewritten=False,
    ),
)


def run_scenario(case: ScenarioCase):
    """Play the scenario's turns through one agent and session; return the last result."""

    memory = ShortTermSession()
    agent = AgentV2(catalog=default_catalog(), registry=build_drawdown_registry(), session=memory, config=AgentV2Config(enable_web_fallback=case.allow_web, record_sub_agents=False))
    result = None
    for turn in case.turns:
        result = agent.run(turn, session_id=f"scenario-{case.id}", allow_web=case.allow_web)
    assert result is not None
    return result


def score_scenario(case: ScenarioCase) -> ScenarioScore:
    result = run_scenario(case)
    called = tuple(dict.fromkeys(task.capability for task in result.plan.tasks))
    planned = set(called)
    capabilities_ok = case.expected_capabilities <= planned
    discipline_ok = not (case.forbidden_capabilities & planned)
    missing = tuple(phrase for phrase in case.expected_phrases if phrase not in result.answer)
    leaked = tuple(phrase for phrase in case.forbidden_phrases if phrase in result.answer)
    phrases_ok = not missing and not leaked
    rewritten = bool(result.request.metadata.get("rewritten"))
    rewritten_ok = case.expect_rewritten is None or rewritten == case.expect_rewritten
    verified_ok = result.verification.ok == case.expect_verified
    from collections import Counter

    from v2.agent_v2.models import sub_agent_summaries

    summaries = sub_agent_summaries(result.results)
    runs = Counter(entry["name"] for entry in summaries)
    stops = {entry["name"]: Counter() for entry in summaries}
    for entry in summaries:
        stops[entry["name"]][entry["stop_reason"] or "?"] += 1
    agents = tuple(f"{name}×{count} " + "/".join(f"{stop} {n}" for stop, n in sorted(stops[name].items())) for name, count in sorted(runs.items()))
    finished = all(entry["stop_reason"] in {"finished", "no_filings"} for entry in summaries)
    agents_ok = all(runs.get(name, 0) >= minimum for name, minimum in case.expected_agents.items()) and finished
    passed = capabilities_ok and discipline_ok and phrases_ok and rewritten_ok and verified_ok and agents_ok
    return ScenarioScore(case.id, passed, capabilities_ok, discipline_ok, phrases_ok, rewritten_ok, verified_ok, called, missing + tuple(f"leaked: {phrase}" for phrase in leaked), agents, agents_ok)
