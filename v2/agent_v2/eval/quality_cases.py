"""Graded questions: what a good answer must say, what it must not, and where it must come from.

Each case is a real question with a rubric.  ``criteria`` are sentences a
model judge checks against the answer (met or not, with the sentence that
meets it); ``forbidden`` are assertions the answer must not make; the
deterministic checks are the route, the sub-agents expected to run and the
source kinds that must be cited.  The quality run scores every case and
keeps the record, so a change can be read as "pass rate went from X to Y"
instead of one person reading Telegram.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from v2.agent_v2.models import RouteKind


@dataclass(frozen=True)
class QualityCase:
    id: str
    question: str
    #: Sentences the answer must satisfy; the judge decides each one.
    criteria: tuple[str, ...]
    #: Assertions the answer must not make; the judge decides each one.
    forbidden: tuple[str, ...] = ()
    #: Source kinds (``EvidenceItem.source_id`` prefixes) at least one cited item must come from.
    must_cite: tuple[str, ...] = ()
    expected_route: RouteKind | None = None
    #: Sub-agent names that must have run (``move_attributor``, ``news_checker``, ``filing_reader``, ``debater``).
    expected_agents: tuple[str, ...] = ()
    allow_web: bool = True
    #: Where the case came from: ``seed`` or ``feedback`` (a user's correction turned into a case).
    origin: str = "seed"
    note: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)
    #: Turns sent in the same session before the graded question (a first question the
    #: graded one refers back to, or a "记住，…" preference the answer must follow).
    preceding: tuple[str, ...] = ()
    #: Upper bound on the answer's length in characters (0: no bound); a deterministic check.
    max_chars: int = 0
    #: ``dev`` cases are looked at while iterating; ``holdout`` cases are run, never read.
    set: str = "dev"


QUALITY_CASES: tuple[QualityCase, ...] = (
    # -- attribution: today ------------------------------------------------------
    QualityCase(
        "q_today_attribution",
        "AAPL今天为什么涨？",
        criteria=(
            "给出了当日涨跌幅，并说明是盘中还是收盘口径",
            "给出了行业基准（如 XLK）同日回报和相对表现的对照",
            "把没有直接证据的解释标为候选或可能相关，而不是已确认原因",
            "说明了接下来值得观察什么",
        ),
        forbidden=("把候选解释说成已确认的原因", "声称申报的正文没有被读取"),
        must_cite=("market_data",),
        expected_route=RouteKind.RESEARCH,
        expected_agents=("move_attributor",),
        tags=("attribution", "today", "quick"),
    ),
    QualityCase(
        "q_english_attribution",
        "why did AMD drop today",
        criteria=("用中文回答", "给出了当日涨跌幅和口径", "给出了行业基准对照"),
        forbidden=("把候选解释说成已确认的原因",),
        must_cite=("market_data",),
        expected_route=RouteKind.RESEARCH,
        expected_agents=("move_attributor",),
        tags=("attribution", "english"),
    ),
    QualityCase(
        "q_attribution_colloquial",
        "特斯拉咋回事，今天怎么跌成这样",
        criteria=("把特斯拉识别为 TSLA 并围绕它回答", "给出了当日涨跌幅和口径", "给出了行业或大盘基准对照"),
        forbidden=("把候选解释说成已确认的原因",),
        must_cite=("market_data",),
        expected_route=RouteKind.RESEARCH,
        expected_agents=("move_attributor",),
        tags=("attribution", "paraphrase"),
    ),
    QualityCase(
        "q_attribution_yesterday",
        "NVDA昨天为什么跌？",
        criteria=("回答的是上一个交易日而不是今天", "给出了那天的涨跌幅并说明口径", "对没有确认证据的解释加了限定"),
        forbidden=("把候选解释说成已确认的原因",),
        must_cite=("market_data",),
        expected_agents=("move_attributor",),
        tags=("attribution", "yesterday"),
    ),
    # -- a stretch of decline or gains ------------------------------------------
    QualityCase(
        "q_drawdown_chain",
        "ARM买入以来跌了这么多，是什么原因？",
        criteria=(
            "先说明买入以来的浮亏和这段跌幅落在哪个区间，再说单日涨跌只是旁注",
            "给出了从高点到低点的回撤幅度和同期行业基准的对照",
            "按日期列出了跌幅最大的交易日，并对每一天分别说有没有确认的原因",
            "把已读取的申报事件与对应日期对应起来，或明说该日没有对应事件",
        ),
        forbidden=("声称申报的正文或内容没有被读取", "用今天的涨跌解释买入以来的亏损"),
        must_cite=("market_data",),
        expected_agents=("move_attributor", "filing_reader"),
        tags=("drawdown",),
    ),
    QualityCase(
        "q_drawdown_from_high",
        "特斯拉从高点跌下来了多少？为什么？",
        criteria=("给出了高点日期、低点或当前价和回撤幅度", "给出了同期行业或大盘基准的对照", "对回撤期间的主要下跌日分别说了有没有确认的原因"),
        forbidden=("用今天的涨跌解释整段回撤",),
        must_cite=("market_data",),
        tags=("drawdown", "window"),
    ),
    QualityCase(
        "q_runup",
        "AMD这一个月涨了多少？涨的原因是什么？",
        criteria=("给出了近一个月的区间回报和起止日期", "给出了同期行业基准对照", "对涨幅最大的交易日说了有没有确认的原因"),
        forbidden=("把候选解释说成已确认的原因",),
        must_cite=("market_data",),
        tags=("runup", "window"),
    ),
    # -- news, filings, earnings of one stock ---------------------------------------
    QualityCase(
        "q_news",
        "ARM最近有哪些新闻？",
        criteria=(
            "按日期列出了近两周可核实的事件，每条有来源",
            "分别交代了网页新闻、SEC 申报和盯盘记录三类来源各有什么或没有什么",
            "指出了主要风险和数据缺口",
        ),
        forbidden=("把没有日期或来源的传闻当作事件",),
        must_cite=("web",),
        expected_agents=("news_checker",),
        tags=("news", "quick"),
    ),
    QualityCase(
        "q_news_paraphrase",
        "英特尔最近有什么动静？",
        criteria=("围绕英特尔（INTC）这家公司回答，没有和别的公司混淆；写不写代码 INTC 都算", "分别交代了网页新闻、SEC 申报和盯盘记录三类来源的结果"),
        expected_agents=("news_checker",),
        tags=("news", "paraphrase"),
    ),
    QualityCase(
        "q_filings",
        "MU最近有什么SEC申报？",
        criteria=("列出了近期申报的类型和日期，或明说该窗口内没有申报", "对读到的申报说了主要内容，而不是只给表格类型"),
        forbidden=("声称申报的正文没有被读取",),
        must_cite=("sec_",),
        tags=("filings",),
    ),
    QualityCase(
        "q_earnings_one",
        "NVDA下次财报是什么时候？上次财报表现如何？",
        criteria=("给出了下次财报日期或明说日历里没有", "给出了上次财报的营收、每股收益或与预期的对比"),
        forbidden=("把历史业绩写成对下次财报的保证",),
        tags=("earnings",),
    ),
    QualityCase(
        "q_insiders",
        "AMD最近有内部人买卖吗？",
        criteria=("给出了近期内部人交易的方向和金额，或明说没有已申报的交易", "说明了内部人数据的口径和局限"),
        forbidden=("把没有申报交易说成内部人看空或看多",),
        tags=("ownership",),
    ),
    # -- research and comparison -------------------------------------------------
    QualityCase(
        "q_valuation",
        "分析NVDA估值",
        criteria=(
            "给出了滚动市盈率等估值倍数并引用来源",
            "把估值和增长、盈利能力放在一起判断，而不是只报倍数",
            "明确指出前瞻口径缺失或其他数据缺口",
            "对异常高的比率（如 ROIC 接近 100%）提示口径依赖",
        ),
        forbidden=("把历史数据写成未来收益保证",),
        must_cite=("fd_", "sec_"),
        expected_route=RouteKind.RESEARCH,
        expected_agents=("debater",),
        tags=("research", "valuation"),
    ),
    QualityCase(
        "q_risk_one",
        "QCOM有什么风险？",
        criteria=("列出了两三条有证据支撑的具体风险", "区分了公司自身风险和行业或宏观风险", "指出了数据缺口"),
        forbidden=("给出无条件的买卖建议",),
        expected_route=RouteKind.RESEARCH,
        tags=("research", "risk"),
    ),
    QualityCase(
        "q_compare",
        "MU和SNDK哪个更值得购买？",
        criteria=(
            "对两只股票用同口径的数字比较（增长、估值、盈利兑现）",
            "明确说现有证据不足以无条件判定谁更值得买，或给出有条件的结论",
            "指出两边的数据缺口",
        ),
        forbidden=("给出无条件的买入建议",),
        expected_route=RouteKind.RESEARCH,
        tags=("compare",),
    ),
    QualityCase(
        "q_compare_three",
        "NVDA、AMD、AVGO三个里面估值谁最贵？",
        criteria=("三只都点名并给出了同口径的估值倍数", "说明了谁最贵以及这个比较的前提或缺口"),
        forbidden=("只比较了其中两只而没有说明第三只为什么缺席",),
        expected_route=RouteKind.RESEARCH,
        tags=("compare", "valuation"),
    ),
    # -- portfolio and watchlist ------------------------------------------------------
    QualityCase(
        "q_watchlist_volume",
        "我关注的股票里有没有最近在放量的？",
        criteria=("对关注列表里的每只股票给出成交量相对均量的倍数", "说明了成交量的口径：是盘中累计进度（不能据此判定放量或缩量）还是已收盘的完整日成交量", "指出哪几只相对靠前"),
        forbidden=("说没有成交量数据",),
        must_cite=("market_data",),
        tags=("watchlist", "performance", "quick"),
    ),
    QualityCase(
        "q_portfolio_ranking",
        "我的持仓里哪只跌的最惨？",
        criteria=("点名跌得最多的那只并给出浮亏百分比", "提到紧随其后的一两只", "把单只和组合放在一起看：给出它占组合的比重或组合整体的盈亏"),
        must_cite=("account.portfolio",),
        tags=("portfolio", "ranking"),
    ),
    QualityCase(
        "q_portfolio_best",
        "持仓里这周谁涨得最好？",
        criteria=("点名本周涨幅最大的那只并给出数字", "说明是周口径而不是买入以来", "提到其余表现靠前的一两只"),
        tags=("portfolio", "ranking", "window"),
    ),
    QualityCase(
        "q_portfolio_pnl",
        "我今天赚了还是亏了？这个月呢？",
        criteria=("给出了当日盈亏的金额和百分比", "给出了本月盈亏", "说明这是模拟或实盘账户的口径"),
        must_cite=("account.",),
        tags=("portfolio", "pnl"),
    ),
    QualityCase(
        "q_portfolio_risk",
        "我的组合现在最大的风险是什么？",
        criteria=("点出了集中度或单一持仓占比这类组合层面的风险并给出数字", "提到了回撤或波动的数字", "说明了大盘 ETF 内部分散和单票集中的区别（如果持有大盘 ETF）"),
        tags=("portfolio", "risk"),
    ),
    QualityCase(
        "q_earnings_calendar",
        "接下来两周我的持仓里谁要出财报？",
        criteria=("直接回答未来两周持仓里有没有财报安排", "如果没有，说明这是日历口径，不等于确定没有"),
        tags=("portfolio", "earnings"),
    ),
    QualityCase(
        "q_watchlist_view",
        "我的关注列表里有哪些股票？",
        criteria=("原样列出了关注列表里的全部代码",),
        forbidden=("逐只分析每只股票的行情",),
        expected_route=RouteKind.FAST_LOOKUP,
        allow_web=False,
        tags=("watchlist", "lookup", "quick"),
    ),
    # -- market and macro ----------------------------------------------------------
    QualityCase(
        "q_briefing",
        "美股今天有啥注意的？",
        criteria=("点出当天或近几天的宏观数据和事件（如 CPI、FOMC）", "给出组合的当日盈亏、集中度或回撤", "说明未来两周有没有财报安排"),
        forbidden=("逐只解释持仓当天的涨跌原因",),
        tags=("briefing",),
    ),
    QualityCase(
        "q_market",
        "今天美股行情如何？",
        criteria=("给出了三大指数或对应 ETF（SPY、QQQ、DIA）的当日涨跌幅和口径", "给出了近 5 日或近 1 月的走势对照", "提到了当天或近期的宏观事件"),
        forbidden=("用用户账户的盈亏代替大盘行情",),
        must_cite=("market_data",),
        tags=("market", "quick"),
    ),
    QualityCase(
        "q_macro_release",
        "最近一次CPI数据怎么样？",
        criteria=("给出了最新一次 CPI 的数值和发布日期", "说明了与预期或前值的对比，或明说没有对比数据"),
        forbidden=("编造尚未发布的数据",),
        allow_web=False,
        tags=("macro", "release"),
    ),
    QualityCase(
        "q_macro_rates",
        "现在美债收益率和VIX是多少？",
        criteria=("给出了 10 年期和 2 年期收益率的数字", "给出了 VIX 的数字", "说明了数据的时间点"),
        allow_web=False,
        tags=("macro",),
    ),
    # -- institutions --------------------------------------------------------------
    QualityCase(
        "q_guru",
        "巴菲特最新的13F持仓有什么变化？",
        criteria=("给出了最新 13F 的报告期", "列出了主要持仓或本期增减仓，或明说没有变化数据", "说明 13F 有滞后"),
        allow_web=False,
        tags=("guru",),
    ),
    QualityCase(
        "q_ark",
        "ARKK最近在买什么？",
        criteria=("列出了近期的买入或卖出及日期", "说明了数据口径和时间范围"),
        allow_web=False,
        tags=("ark",),
    ),
    # -- knowledge and help ----------------------------------------------------------
    QualityCase(
        "q_knowledge",
        "市盈率和市销率有什么区别？",
        criteria=("给出两者的定义和分母的差别", "说明各自适用的场景和局限", "声明这是通用知识，未使用实时数据"),
        forbidden=("引用某只具体股票当前的市盈率数字",),
        expected_route=RouteKind.GENERAL_KNOWLEDGE,
        allow_web=False,
        tags=("knowledge", "quick"),
    ),
    QualityCase(
        "q_help",
        "你能做什么？",
        criteria=("列出了能做的几类事（行情归因、新闻、申报、持仓、宏观、提醒）", "没有编造不存在的功能"),
        allow_web=False,
        tags=("help",),
    ),
    # -- commands --------------------------------------------------------------------
    QualityCase(
        "q_command",
        "NVDA涨到240时提醒我",
        criteria=("说明将要执行的写操作并等待确认，没有直接执行",),
        expected_route=RouteKind.COMMAND,
        allow_web=False,
        tags=("command",),
    ),
    QualityCase(
        "q_command_watchlist",
        "把 AVGO 加到关注列表",
        criteria=("说明将把 AVGO 加入关注列表并等待确认，没有直接执行",),
        expected_route=RouteKind.COMMAND,
        allow_web=False,
        tags=("command",),
    ),
    QualityCase(
        "q_command_ambiguous",
        "帮我盯着点特斯拉",
        criteria=("要么理解为加入关注列表并等待确认，要么反问是加关注还是设提醒；两种都算", "没有直接执行写操作"),
        allow_web=False,
        tags=("command", "ambiguous"),
    ),
    # -- multi-turn ------------------------------------------------------------------
    QualityCase(
        "q_clarify_alert",
        "跌到150的时候",
        criteria=("把这句理解为上一轮 AMD 提醒的补充：AMD 跌到 150 时提醒，说明将要执行的写操作并等待确认，没有直接执行",),
        preceding=("给AMD设个提醒",),
        expected_route=RouteKind.COMMAND,
        allow_web=False,
        tags=("multi_turn", "command", "clarification"),
    ),
    QualityCase(
        "q_followup_pronoun",
        "那它最近有什么新闻？",
        criteria=("把“它”理解为上一轮的 AAPL 并围绕 AAPL 回答", "按日期列出了近两周可核实的事件"),
        preceding=("AAPL今天为什么涨？",),
        tags=("multi_turn", "news"),
    ),
    QualityCase(
        "q_followup_other",
        "那SNDK的风险呢？",
        criteria=("围绕 SNDK 回答而不是 MU", "列出了有证据支撑的具体风险"),
        preceding=("MU和SNDK哪个更值得购买？",),
        expected_route=RouteKind.RESEARCH,
        tags=("multi_turn", "risk"),
    ),
    QualityCase(
        "q_followup_expand",
        "上面第二点展开讲",
        criteria=("围绕上一条回答里的第二点展开，而不是把整条回答重说一遍", "展开的内容有证据引用"),
        forbidden=("把上一条回答里没有的新事实当成证据",),
        preceding=("QCOM有什么风险？",),
        tags=("multi_turn", "follow_up"),
    ),
    QualityCase(
        "q_followup_why",
        "为什么这么说？",
        criteria=("解释上一条估值结论的依据，引用支持它的证据", "没有另起炉灶回答一个新问题"),
        preceding=("分析NVDA估值",),
        tags=("multi_turn", "follow_up"),
    ),
    QualityCase(
        "q_preference_short",
        "分析AMD估值",
        criteria=("给出了估值倍数并引用来源", "没有展开成多段长文，只有两三条要点、一个风险和一个观察点"),
        preceding=("记住，回答短一点",),
        max_chars=500,
        expected_route=RouteKind.RESEARCH,
        tags=("multi_turn", "preference"),
    ),
    # -- investigator: the three jobs ----------------------------------------------
    QualityCase(
        "q_investigate_event",
        "英特尔被美国政府入股那件事的来龙去脉是什么？",
        criteria=(
            "按时间顺序交代了事件的起因、经过和最新进展，至少有两个带日期的节点",
            "每个节点都注明出处（报道或申报）并带有原文引文",
            "没有找到出处的环节明说没有找到，而不是用常识补全",
        ),
        forbidden=("把没有出处的说法写成已确认的事实",),
        expected_route=RouteKind.RESEARCH,
        expected_agents=("investigator",),
        tags=("investigate", "event_story"),
    ),
    QualityCase(
        "q_investigate_filing_terms",
        "特斯拉最新的 10-K 里对 FSD 自动驾驶的风险具体是怎么写的？",
        criteria=(
            "说明了是哪份申报（10-K）的哪个章节",
            "引用了申报原文的措辞，而不是转述常识",
            "最后有一句概括，且没有超出原文的推断",
        ),
        forbidden=("把申报里没有的表述写成申报原文",),
        expected_route=RouteKind.RESEARCH,
        expected_agents=("investigator",),
        allow_web=False,
        tags=("investigate", "filing_terms"),
    ),
    QualityCase(
        "q_investigate_claim",
        "有说法称英伟达要把 H20 在华收入的 15% 交给美国政府，这个说法有出处吗？",
        criteria=(
            "先给结论：有出处、没有找到出处、或来源之间有出入",
            "列出了出处（报道、申报或公司声明）并引用原文",
            "区分了报道与公司或政府的正式确认",
        ),
        forbidden=("在没有出处的情况下把说法写成事实",),
        expected_route=RouteKind.RESEARCH,
        expected_agents=("investigator",),
        tags=("investigate", "claim_source"),
    ),
)


def from_feedback(rows: list[dict[str, Any]], *, limit: int = 20) -> tuple[QualityCase, ...]:
    """Turn negative feedback into cases: the question, with the user's correction as the criterion."""

    cases: list[QualityCase] = []
    seen: set[str] = set()
    for row in reversed(rows):
        if str(row.get("verdict") or "") != "bad":
            continue
        question = " ".join(str(row.get("question") or "").split())
        if not question or question in seen:
            continue
        seen.add(question)
        note = " ".join(str(row.get("note") or "").split())
        criterion = f"回答不再出现用户指出的问题：{note}" if note else "回答与用户上次指出有误的回答不同，并且有证据支持"
        cases.append(QualityCase(f"fb_{len(cases) + 1}", question, criteria=(criterion,), origin="feedback", note=note, tags=("feedback",)))
        if len(cases) >= limit:
            break
    return tuple(cases)
