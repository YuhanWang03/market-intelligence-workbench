"""A deterministic planner keyed on the question's intent; an LLM planner can refine it.

The intent (``intent.Intent``) says what the question wants: its kind, the
time it is about, a direction, the topics (``wants``), the tickers, whether
it is about the user's account or watchlist, and the structured details a
command or a lab run needs.  The planner turns those fields into tasks.
It never reads the wording: translating words into fields is the model's
job (or the recorded labels', offline), and the templates below are the
part that must stay deterministic — the drawdown chain with its extreme
days and sector benchmark, the news plan, the account and watchlist fan-outs.
"""

from __future__ import annotations

from typing import Any

from v2.agent_v2.intent import FOCUS_OF_WANT, MARKET_TICKERS, Intent, default_intent
from v2.agent_v2.models import (
    AnswerMode,
    BudgetClass,
    ExecutionPlan,
    NormalizedRequest,
    PlanTask,
    RouteDecision,
    RouteKind,
)

_HELP_ANSWER = (
    "我可以：查看持仓、盈亏和组合风险；研究单只股票的财报、内部人交易、SEC 申报、资金流、估值和产业链；"
    "解释个股异动并对比行业基准；查询宏观数据、知名基金经理的 13F 和 ARK ETF 动向；"
    "对持仓或关注列表逐只排查；运行筛选、回测和事件研究；管理关注列表和价格提醒。"
)

#: Wants that ask for research-engine modules (as opposed to market data, account facts or state).
#: The investigator's brief, tools and look-back per job type (``Intent.investigation``).
INVESTIGATION_JOBS: dict[str, dict] = {
    "event_story": {
        "brief": "梳理这件事的来龙去脉，按日期找出起因、经过和最新进展，每个节点给出处和原文引文：",
        "tools": ("search_news", "read_page", "list_filings", "read_filing", "recall_memory"),
        "days": 120,
        "purpose": "the dated story of the event from news pages, filings and the monitor's memory, each node quoted",
        "answer": "按时间顺序写事件的起因、经过和最新进展；",
    },
    "filing_terms": {
        "brief": "在该公司的 SEC 申报里找到相关章节，原样引用具体条款和措辞，不要转述常识。list_filings 用 forms 指定题目问的表格（年报 [\"10-K\"]、季报 [\"10-Q\"]、临时报告 [\"8-K\"]），read_filing 用 find 给关键词（中英文都写，如产品名、条款名）定位段落，先读 Risk Factors 一类的相关章节：",
        "tools": ("list_filings", "read_filing"),
        "days": 400,
        "purpose": "the filing's own words on the point asked, section by section",
        "answer": "先说是哪份申报的哪个章节，再引用原文条款，最后一句概括；",
    },
    "claim_source": {
        "brief": "核实这个说法有没有出处：找到最早或最权威的来源（报道、申报、公司声明），引用原文，并说明来源之间是否一致：",
        "tools": ("search_news", "read_page", "list_filings", "read_filing"),
        "days": 180,
        "purpose": "whether the claim has a source, which, and what it says in its own words",
        "answer": "先给结论（有出处/没有找到出处/来源之间有出入），再列出处和引文；",
    },
}
_RESEARCH_WANTS = ("valuation", "earnings", "filings", "ownership", "supply_chain", "catalysts", "risk", "full", "overview")


def intent_of(request: NormalizedRequest, route: RouteDecision) -> Intent:
    """The intent the route carries, or the default for a route built without one."""

    intent = getattr(route, "intent", None)
    return intent if isinstance(intent, Intent) else default_intent(request)


def focuses_of(intent: Intent) -> list[str]:
    """Research focuses: the explicit ones, else the research wants; three or more (or ``full``) collapse to ``full``."""

    found = list(intent.focus)
    for want in intent.wants:
        focus = FOCUS_OF_WANT.get(want)
        if focus and focus not in found and want in _RESEARCH_WANTS:
            found.append(focus)
    if "full" in found or len(found) > 2:
        return ["full"]
    return found


def mutation_task(intent: Intent, entities: tuple[str, ...]) -> tuple[PlanTask | None, str]:
    """Map a command intent to one ``state.mutate`` task, or explain what is missing."""

    command = dict(intent.command or {})
    operation = str(command.get("operation") or "")
    ticker = str(command.get("ticker") or (intent.tickers[0] if intent.tickers else "") or (entities[0] if entities else "")).upper()
    if operation == "alert.remove":
        alert_id = command.get("alert_id")
        if not isinstance(alert_id, int):
            return None, "取消提醒需要提醒编号（可先查看提醒列表）。"
        return PlanTask("mutation", "state.mutate", {"operation": "alert.remove", "payload": {"alert_id": int(alert_id)}}, purpose=f"取消提醒 #{alert_id}"), ""
    if operation == "alert.add":
        if not ticker:
            return None, "设置提醒需要股票代码。"
        direction, price = str(command.get("direction") or ""), command.get("price")
        if direction not in {"above", "below"} or not isinstance(price, (int, float)) or price <= 0:
            return None, f"为 {ticker} 设置提醒需要方向（涨到/跌到）和目标价。"
        label = "涨到" if direction == "above" else "跌到"
        return PlanTask("mutation", "state.mutate", {"operation": "alert.add", "payload": {"ticker": ticker, "direction": direction, "target_price": float(price)}}, purpose=f"当 {ticker} {label} {float(price):g} 美元时提醒"), ""
    if not ticker:
        return None, "关注列表操作需要股票代码。"
    if operation == "watchlist.remove":
        return PlanTask("mutation", "state.mutate", {"operation": "watchlist.remove", "payload": {"ticker": ticker}}, purpose=f"将 {ticker} 移出关注列表"), ""
    return PlanTask("mutation", "state.mutate", {"operation": "watchlist.add", "payload": {"ticker": ticker}}, purpose=f"将 {ticker} 加入关注列表"), ""


def _is_heavy(capability: str) -> bool:
    """A research-engine run or a sub-agent: never a 30-second lookup."""

    from v2.agent_v2.catalog import default_catalog

    spec = default_catalog().get(capability)
    return capability.startswith("research.") or bool(spec is not None and spec.long_running)


def _budget(tasks: list[PlanTask]) -> BudgetClass:
    if any(task.fan_out for task in tasks):
        return BudgetClass.PORTFOLIO
    if any(task.capability == "research.compare" for task in tasks):
        # Two or more stocks researched cold in parallel do not fit the
        # focused 60 s; the run was timing out and falling back to the web.
        return BudgetClass.COMPARISON
    if any(task.capability == "research.stock" for task in tasks):
        # A cold research run of one stock has overrun 60 s as well.
        return BudgetClass.STANDARD
    count = len(tasks)
    if count <= 1:
        return BudgetClass.FOCUSED if any(_is_heavy(task.capability) for task in tasks) else BudgetClass.DIRECT
    if count <= 2:
        return BudgetClass.FOCUSED
    if count <= 5:
        return BudgetClass.STANDARD
    if count <= 7:
        return BudgetClass.PORTFOLIO
    return BudgetClass.DEEP


def portfolio_ranking(intent: Intent) -> bool:
    """Whether the question ranks the user's holdings by a direction ("哪只跌得最多")."""

    return fan_out_rank(intent) is not None


def fan_out_rank(intent: Intent) -> dict | None:
    """Order a portfolio fan-out by P/L when the question ranks holdings by direction."""

    if "ranking" not in intent.wants:
        return None
    rank = intent.rank or ("low" if intent.direction == "down" else "high" if intent.direction == "up" else "")
    if not rank:
        return None
    return {"field": "positions", "key": "pl_pct", "descending": rank == "high"}


class IntentPlanner:
    """Produces conservative plans from the intent; works without an LLM or API key given recorded labels."""

    def plan(self, request: NormalizedRequest, route: RouteDecision) -> ExecutionPlan:
        text, entities = request.text, request.entities
        intent = intent_of(request, route)
        if route.kind == RouteKind.GENERAL_KNOWLEDGE:
            return ExecutionPlan(objective=text, route=route.kind, answer_mode=AnswerMode.GENERAL_KNOWLEDGE, budget=BudgetClass.DIRECT)
        if route.kind == RouteKind.COMMAND:
            task, problem = mutation_task(intent, entities)
            return ExecutionPlan(
                objective=text,
                route=route.kind,
                tasks=(task,) if task else (),
                answer_mode=AnswerMode.TOOL_GROUNDED,
                budget=BudgetClass.DIRECT,
                requires_confirmation=task is not None,
                direct_answer=problem,
                assumptions=("No mutation is executed until the user confirms the exact operation.",),
            )
        if route.kind in {RouteKind.LAB, RouteKind.ASYNC}:
            return self._lab(text, intent, entities, route)
        if intent.kind == "help":
            return ExecutionPlan(objective=text, route=route.kind, answer_mode=AnswerMode.GENERAL_KNOWLEDGE, budget=BudgetClass.DIRECT, direct_answer=_HELP_ANSWER)

        tickers = tuple(ticker for ticker in (intent.tickers or entities) if not ticker.startswith("ARK") or ticker in entities and ticker not in intent.ark_etfs)
        tickers = tuple(ticker for ticker in tickers if ticker not in intent.ark_etfs)
        frame = request.metadata.get("context_frame")
        notes: list[str] = []
        if intent.source == "default":
            notes.append(f"intent: {intent.note}")

        if intent.investigation:
            return self._investigate(text, intent, tickers, route, request, notes)

        if len(tickers) == 1:
            stretch = self._stretch_direction(intent, frame if isinstance(frame, dict) else None)
            if stretch:
                direction, framing = stretch
                return self._stretch(text, tickers[0], framing, route, request, direction, notes)
            if intent.wants_any("news") and not intent.wants_any("attribution"):
                return self._news(text, tickers[0], route, request, notes)
            if intent.wants_any("attribution"):
                tasks = [PlanTask("market-move", "market.explain_move", {"ticker": tickers[0]}, purpose="separate confirmed market facts from candidate move drivers")]
                for index, focus in enumerate(focuses_of(intent), 1):
                    tasks.append(PlanTask(f"research-{index}", "research.stock", {"ticker": tickers[0], "focus": focus}, purpose=f"collect {focus} evidence"))
                return ExecutionPlan(objective=text, route=route.kind, tasks=tuple(tasks[:5]), answer_mode=AnswerMode.RESEARCH_GROUNDED, budget=_budget(tasks[:5]), web_fallback_allowed=request.allow_web, assumptions=tuple(notes))
            if intent.wants_any("performance") and not any(want in _RESEARCH_WANTS for want in intent.wants) and not intent.focus:
                return ExecutionPlan(
                    objective=text,
                    route=route.kind,
                    tasks=(PlanTask("market-performance", "market.performance", {"ticker": tickers[0]}, purpose="measure recent returns, volume and benchmark-relative performance"),),
                    answer_mode=AnswerMode.TOOL_GROUNDED,
                    budget=BudgetClass.FOCUSED,
                    assumptions=tuple(notes),
                )
            if intent.wants_any("research_changes"):
                return ExecutionPlan(objective=text, route=route.kind, tasks=(PlanTask("research-change", "research.changes", {"ticker": tickers[0]}, purpose="compare stored research snapshots"),), answer_mode=AnswerMode.RESEARCH_GROUNDED, budget=BudgetClass.DIRECT, assumptions=tuple(notes))

        tasks = self._compose(intent, tickers)
        grounded = any(task.capability.startswith(("research.", "market.")) for task in tasks)
        return ExecutionPlan(
            objective=text,
            route=route.kind,
            tasks=tuple(tasks),
            answer_mode=AnswerMode.RESEARCH_GROUNDED if grounded else AnswerMode.TOOL_GROUNDED,
            budget=_budget(tasks),
            web_fallback_allowed=request.allow_web,
            assumptions=tuple(notes),
        )

    # -- pieces ---------------------------------------------------------------

    @staticmethod
    def _stretch_direction(intent: Intent, frame: dict | None) -> tuple[str, dict] | None:
        """Whether the question is about a stretch (a loss since purchase, a fall from the high) and which way.

        A framed follow-up ("为什么跌这么狠" after the position with a loss was
        named) goes by the position's sign unless the question's direction
        contradicts it, which makes it about today's move instead.
        """

        if not intent.wants_any("attribution", "drawdown", "runup") or intent.scope == "today":
            return None
        if frame and frame.get("kind") == "position" and frame.get("field") == "pl_pct" and isinstance(frame.get("value"), (int, float)):
            loss = frame["value"] < 0
            if loss and intent.direction != "up" and intent.wants_any("attribution", "drawdown"):
                return "down", frame
            if not loss and frame["value"] > 0 and intent.direction != "down" and intent.wants_any("attribution", "runup"):
                return "up", frame
            return None
        if intent.wants_any("drawdown") or (intent.direction == "down" and intent.scope in {"window", "since_purchase"}):
            return "down", {"kind": "drawdown", "ticker": intent.tickers[0] if intent.tickers else "", "label": "这段跌幅", "window": intent.window}
        if intent.wants_any("runup") or (intent.direction == "up" and intent.scope in {"window", "since_purchase"}):
            return "up", {"kind": "runup", "ticker": intent.tickers[0] if intent.tickers else "", "label": "这段涨幅", "window": intent.window}
        return None

    @staticmethod
    def _investigate(text: str, intent: Intent, tickers: tuple[str, ...], route: RouteDecision, request: NormalizedRequest, notes: list[str]) -> ExecutionPlan:
        """The investigator's three jobs, each with its own brief, tools and look-back.

        The story of an event is read from news pages, filings and the
        monitor's memory; a filing's terms only from the filing itself; a
        claim's source from the news and the filings.  Every finding it
        reports carries a located quote, so the answer is a dated list of
        what the sources say and an honest "not found" otherwise.
        """

        from datetime import date, timedelta

        ticker = tickers[0] if tickers else ""
        job = INVESTIGATION_JOBS[intent.investigation]
        tasks = [PlanTask("investigate", "agent.investigate", {"task": job["brief"] + text, **({"ticker": ticker} if ticker else {}), "tools": list(job["tools"]), "recency_days": job["days"]}, purpose=job["purpose"])]
        if ticker and intent.investigation == "event_story":
            tasks.append(PlanTask("anomaly-history", "market.anomaly_history", {"ticker": ticker, "lookback_days": job["days"]}, purpose="what the monitor recorded over the period", required=False))
        if ticker and intent.investigation == "filing_terms":
            tasks.append(PlanTask("filings-recent", "filings.recent", {"ticker": ticker, "since": (date.today() - timedelta(days=job["days"])).isoformat()}, purpose="the dated list of filings the terms may sit in", required=False))
        web_state = "网页已授权。" if request.allow_web else "网页未授权，调查员只读申报和盯盘记录，回答里说明未查新闻。"
        note = (
            f"investigation: {job['answer']}调查员的每条发现单独成一行，格式「日期：事实（来源名：“原文引文”）[id]」——日期用证据开头的那个日期，"
            "引文照抄证据里引号内的原文（英文原文不翻译、不改写、不省略），一条都不能合并或省掉；同一天的几条发现各占一行。"
            f"调查员没找到的环节就明说没找到，不得用常识补全故事或条款。{web_state}"
        )
        return ExecutionPlan(objective=text, route=route.kind, tasks=tuple(tasks), answer_mode=AnswerMode.RESEARCH_GROUNDED, budget=BudgetClass.STANDARD, web_fallback_allowed=False, assumptions=(note, *notes))

    @staticmethod
    def _news(text: str, ticker: str, route: RouteDecision, request: NormalizedRequest, notes: list[str]) -> ExecutionPlan:
        """"What's the news on X": dated events from the web (with consent), recent filings and the monitor's memory."""

        from datetime import date, timedelta

        since = (date.today() - timedelta(days=14)).isoformat()
        tasks = [
            PlanTask("web-news", "web.research", {"query": f"{ticker} stock news latest two weeks", "topic": "company_event", "ticker": ticker, "recency_days": 14, "min_searches": 2}, purpose="dated events from news pages, quotes located in the text", required=False),
            PlanTask("filings-recent", "filings.recent", {"ticker": ticker, "since": since, "forms": ["8-K", "6-K", "4", "424B5"]}, purpose="what the company and its insiders filed in the last two weeks (an offering is a 424B5 prospectus)", required=False),
            PlanTask("anomaly-history", "market.anomaly_history", {"ticker": ticker, "lookback_days": 30}, purpose="what the monitor recorded in the last month", required=False),
        ]
        web_state = "网页已授权并已搜索，不得写成未授权或无法访问网页。" if request.allow_web else "网页未授权，只列申报和盯盘记录，并说明未使用网页。"
        note = f"news: 按日期列出近两周的事件（网页、申报含 Form 4 内幕交易、盯盘记录各注明来源），没有事件的来源明说。{web_state}"
        return ExecutionPlan(objective=text, route=route.kind, tasks=tuple(tasks), answer_mode=AnswerMode.RESEARCH_GROUNDED, budget=_budget(tasks), web_fallback_allowed=request.allow_web, assumptions=(note, *notes))

    @staticmethod
    def _stretch(text: str, ticker: str, frame: dict, route: RouteDecision, request: NormalizedRequest, direction: str, notes: list[str]) -> ExecutionPlan:
        """Explain a loss (or gain) since purchase: cost basis, where in time the stretch sits, events over the period; today's move only as an aside."""

        up = direction == "up"
        value_text = frame.get("value_text") or frame.get("value")
        entry = frame.get("avg_entry_price")
        entry_text = f"（成本价 ${float(entry):.2f}）" if isinstance(entry, (int, float)) else ""
        word, extreme, days = ("涨幅", "低点到高点的涨幅", "上涨日") if up else ("跌幅", "高点到低点的回撤", "下跌日")
        subject = f"{ticker} {frame.get('label') or '买入以来的浮动盈亏'} {value_text}{entry_text}" if value_text is not None else f"{ticker} 从{'低' if up else '高'}点以来或这段时间的{word}"
        note = (
            f"context_frame: 用户{'追' if value_text is not None else ''}问的是 {subject}，不是今日涨跌。"
            f"先回答这段{word}落在哪个区间（对照 5 日、1 月、3 月、1 年回报窗口）、区间{extreme}和同期行业基准对比，再按日期列出{word}最大的交易日，"
            f"把日期相同或相邻的 SEC 申报（含从申报原文摘出的事件）、盯盘记录和新闻与这些{days}对应起来，每个{days}的高置信度驱动和候选解释要分开说；"
            f"今日涨跌只用区间回报里的单日数字作一句旁注，并点明它与买入以来的{word}是不同区间；某个{days}找不到对应事件就明说，不得用当日归因冒充。"
        )
        loss = frame.get("value")
        tasks = [
            PlanTask("account-portfolio", "account.portfolio", purpose="restate the position's cost basis and unrealized P/L"),
            PlanTask("market-performance", "market.performance", {"ticker": ticker}, purpose=f"locate the {'rise' if up else 'decline'} across return windows"),
            PlanTask(
                "market-stretch",
                "market.runup" if up else "market.drawdown",
                {"ticker": ticker, **({("gain_pct" if up else "loss_pct"): float(loss)} if isinstance(loss, (int, float)) else {}), **({"window": str(frame["window"])} if frame.get("window") else {}), "top": 3},
                purpose="low-to-high and the best trading days in the window" if up else "peak-to-trough and the worst trading days in the window",
            ),
            PlanTask("filings-recent", "filings.recent", {"ticker": ticker}, purpose="dated SEC filings (8-K, or 6-K for a foreign issuer) over the past year", required=False),
            PlanTask("anomaly-history", "market.anomaly_history", {"ticker": ticker, "lookback_days": 365}, purpose=f"what the monitor recorded on the {'best' if up else 'worst'} days", required=False),
            PlanTask(
                "attribute-best-days" if up else "attribute-worst-days",
                "market.attribute_move",
                {"ticker": ticker},
                purpose=f"explain each of the {'best' if up else 'worst'} days with verifiable sources",
                depends_on=("market-stretch",),
                required=False,
                fan_out={"from": "market-stretch", "field": "best_dates" if up else "worst_dates", "argument": "date", "max": 3},
            ),
        ]
        return ExecutionPlan(
            objective=text,
            route=route.kind,
            tasks=tuple(tasks),
            answer_mode=AnswerMode.RESEARCH_GROUNDED,
            budget=BudgetClass.PORTFOLIO,
            web_fallback_allowed=request.allow_web,
            assumptions=(note, *notes),
            frame={key: value for key, value in frame.items() if key != "ticker" or value} | ({"ticker": ticker} if not frame.get("ticker") else {}),
        )

    @staticmethod
    def _lab(text: str, intent: Intent, entities: tuple[str, ...], route: RouteDecision) -> ExecutionPlan:
        capability = f"lab.{intent.lab or 'screen'}"
        tickers = list(intent.tickers or entities)
        args: dict = {"tickers": tickers} if tickers else {}
        if capability == "lab.backtest":
            args["strategy"] = intent.strategy or "momentum"
        return ExecutionPlan(
            objective=text,
            route=route.kind,
            tasks=(PlanTask("lab-1", capability, args, purpose="requested quantitative experiment"),),
            answer_mode=AnswerMode.TOOL_GROUNDED,
            budget=BudgetClass.DEEP if route.kind == RouteKind.ASYNC else BudgetClass.LAB,
            assumptions=("Missing experiment parameters must be disclosed before execution.",),
        )

    def _compose(self, intent: Intent, tickers: tuple[str, ...]) -> list[PlanTask]:
        tasks: list[PlanTask] = []
        ids: set[str] = set()

        def add(task_id: str, capability: str, arguments: dict | None = None, *, purpose: str = "", depends_on: tuple[str, ...] = (), fan_out: dict | None = None, required: bool = True) -> None:
            if task_id in ids:
                return
            ids.add(task_id)
            tasks.append(PlanTask(task_id, capability, dict(arguments or {}), depends_on=depends_on, purpose=purpose, fan_out=fan_out, required=required))

        wants = set(intent.wants)
        managers, ark_etfs = list(intent.managers[:2]), list(intent.ark_etfs[:2])
        watchlist_scope = intent.watchlist_scope or "watchlist" in wants
        account_scope = intent.portfolio_scope
        explicit_portfolio = "portfolio" in wants
        list_scope = account_scope or watchlist_scope
        #: Wants answered at the account level here; they are not research focuses below.
        account_topics: set[str] = set()

        # account-level topics
        # A ranking over the holdings ("这周谁涨得最好") needs each holding's return, not the account's P&L.
        if "performance" in wants and not tickers and not watchlist_scope and not managers and "market" not in wants and not ("ranking" in wants and intent.scope in {"today", "recent", "window"}):
            for period in intent.periods or ("day",):
                add(f"account-performance-{period}", "account.performance", {"period": period}, purpose=f"account P&L for the {period}")
            account_topics.add("performance")
        if "risk" in wants and (account_scope or explicit_portfolio or not tickers) and not managers:
            add("account-risk", "account.risk", purpose="collect portfolio-level risk")
            account_topics.add("risk")
        if "earnings" in wants and not tickers and intent.kind != "research":
            # The calendar lists every holding's date; "各自的财报日期" (each=True) used
            # to fan the research engine over twelve holdings and time out at 240 s.
            add("account-earnings", "account.earnings_schedule", {"days": 14}, purpose="upcoming earnings across holdings and watchlist")
            account_topics.add("earnings")
        if "briefing" in wants and not tickers:
            # What to watch today: the backdrop, the calendar, the book's risk
            # and the watchlist; no per-holding attribution (that is a
            # different question and three sub-agents dearer).
            add("macro-overview", "macro.overview", purpose="market backdrop")
            add("account-earnings", "account.earnings_schedule", {"days": 14}, purpose="upcoming earnings across holdings and watchlist")
            add("account-risk", "account.risk", purpose="today's P/L, drawdown and concentration")
            add("state-watchlist", "state.read", {"section": "watchlist"}, purpose="the watchlist to keep an eye on")
        if "positioning" in wants and not tickers:
            add("macro-overview", "macro.overview", purpose="market backdrop before changing exposure")
            add("account-risk", "account.risk", purpose="collect portfolio-level risk")
            account_topics.add("risk")
            if account_scope or explicit_portfolio:
                add("account-portfolio", "account.portfolio", purpose="identify positions and weights")

        # user state
        if "alerts" in wants:
            add("state-alerts", "state.read", {"section": "alerts"}, purpose="read configured alerts")
        elif "settings" in wants:
            add("state-settings", "state.read", {"section": "settings"}, purpose="read user settings")
        if watchlist_scope:
            add("state-watchlist", "state.read", {"section": "watchlist"}, purpose="read the watchlist")

        # the market as a whole: the macro board plus the index ETFs' price action
        if "market" in wants and not tickers and not list_scope:
            add("macro-overview", "macro.overview", purpose="market backdrop: VIX, yields, releases")
            for symbol in MARKET_TICKERS:
                add(f"performance-{symbol}", "market.performance", {"ticker": symbol}, purpose=f"{symbol} as the market's price action")
            account_topics.add("performance")
        # macro, managers, ARK
        if "macro" in wants:
            add("macro-overview", "macro.overview", purpose="macro dashboard")
        if intent.release or "macro_release" in wants:
            release = intent.release or "fomc"
            add(f"macro-{release}", "macro.release", {"release_type": release}, purpose=f"latest {release.upper()} release")
        for manager in managers:
            add(f"manager-{manager}", "institutional.manager_portfolio", {"manager": manager}, purpose=f"latest 13F for {manager}")
        for symbol in ark_etfs:
            add(f"etf-{symbol}", "etf.ark_activity", {"symbol": symbol}, purpose=f"{symbol} holdings and activity")

        # "MU 最近有什么申报": the dated list of filings, not the engine's filings module
        if "filings" in wants and tickers and intent.kind != "research":
            from datetime import date, timedelta

            since = (date.today() - timedelta(days=45)).isoformat()
            for ticker in tickers[:4]:
                add(f"filings-recent-{ticker}", "filings.recent", {"ticker": ticker, "since": since, "forms": ["8-K", "6-K", "4", "424B5", "10-Q", "10-K"]}, purpose=f"dated filings for {ticker} over the last 45 days")
                # What the filings say: the reader quotes the latest two; the engine's filings module (focus=filings) still runs below.
                add(f"filings-read-{ticker}", "filings.read_events", {"ticker": ticker, "since": since, "max_filings": 2}, purpose=f"what {ticker}'s latest filings disclosed", depends_on=(f"filings-recent-{ticker}",), required=False)
        # per-ticker topics
        explain = "attribution" in wants
        # A want answered at the account level is not a research focus, except
        # "risk" when the holdings are ranked by it ("哪只风险最高" looks at each).
        focuses = [focus for focus in focuses_of(intent) if focus not in account_topics or (focus == "risk" and "ranking" in wants)]
        if tickers:
            if not explain and not focuses:
                focuses = ["market"] if "performance" in wants else ["overview"]
            if len(tickers) >= 2 and (focuses or "compare" in wants) and not explain:
                add("research-compare", "research.compare", {"tickers": list(tickers[:4]), "dimensions": (focuses or ["overview"])[:2]}, purpose="compare identical dimensions")
            else:
                for ticker in tickers[:4]:
                    if explain:
                        add(f"move-{ticker}", "market.explain_move", {"ticker": ticker}, purpose=f"explain {ticker}'s move")
                    for focus in focuses[:2]:
                        add(f"research-{ticker}-{focus}", "research.stock", {"ticker": ticker, "focus": focus}, purpose=f"{focus} research for {ticker}")
        elif list_scope:
            # Account-level wording ("组合风险", "快发财报") is answered by the
            # account capabilities; per-ticker fan-out needs a per-ticker ask.
            if "risk" in focuses and "ranking" not in wants:
                focuses = [focus for focus in focuses if focus != "risk"]
            if "earnings" in focuses and "earnings" in account_topics:
                focuses = [focus for focus in focuses if focus != "earnings"]
            source = "state-watchlist" if watchlist_scope and not explicit_portfolio else "account-portfolio"
            rank = fan_out_rank(intent) if source == "account-portfolio" else None
            # "哪只跌得最多" is answered by the position card's own P/L column;
            # only a time frame or a "why" needs the per-holding market look.
            card_answers = rank is not None and intent.scope in {"none", "since_purchase"} and not explain
            # A performance question over the list ("有没有在放量的") is per name unless the account card already answered it.
            # "这周谁涨得最好" ranks by a return the card does not carry, whether or not the classifier said "performance".
            time_ranked = rank is not None and intent.scope in {"today", "recent", "window"} and not explain
            per_ticker = explain or bool(focuses) or time_ranked or (intent.each and "earnings" not in account_topics) or ("performance" in wants and "performance" not in account_topics and not card_answers)
            if intent.each and "earnings" not in focuses and "earnings" not in account_topics:
                focuses = [*focuses, "earnings"]
            if source == "account-portfolio" and (explicit_portfolio or per_ticker or not tasks):
                add("account-portfolio", "account.portfolio", purpose="identify positions and weights")
            if per_ticker:
                fan_out = {"from": source, "field": "tickers", "argument": "ticker", "max": 8}
                if rank is not None and (explain or intent.scope in {"none", "since_purchase"}):
                    fan_out["rank"] = rank
                elif rank is not None:
                    # "这周谁涨得最好" ranks by the week's return, which the card does not carry: look at every holding.
                    fan_out["max"] = 12
                if explain:
                    add("move-each", "market.explain_move", {}, purpose="explain each holding's recent move", depends_on=(source,), fan_out=dict(fan_out))
                elif ("performance" in wants or time_ranked) and not focuses:
                    add("performance-each", "market.performance", {}, purpose="recent returns and volume for each name", depends_on=(source,), fan_out=dict(fan_out))
                for focus in focuses[:2]:
                    add(f"research-each-{focus}", "research.stock", {"focus": focus}, purpose=f"{focus} research for each holding", depends_on=(source,), fan_out=dict(fan_out))
        return tasks[:7]


#: The name the rest of the code base and the tests grew up with.
RulePlanner = IntentPlanner
