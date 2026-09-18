"""Where a classified question goes: one table for the intent, one for the plan.

Both used to be chains of ``if`` statements in ``graph._classify`` and
``brain.plan``. Each new branch was written for the question that exposed
it, and an earlier, broader branch kept absorbing questions that had a
dedicated tool ("未来两周谁发财报" answered from the portfolio overview). The
tables make the order and the conditions visible, give every rule a name
that is recorded on the run, and let one test walk the whole grid of
intents instead of one question at a time.

``NORMALIZERS`` rewrite the classifier's intent into the shape the planners
expect; ``RULES`` choose the tasks. The first matching rule wins. A rule
whose ``build`` returns ``None`` keeps the shared deterministic plan: that
is a decision too, and it is named like the others.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from typing import Any, Callable

from v2.agent_v2.models import ExecutionPlan, PlanTask

#: Wants that have their own tool in the shared planner. A portfolio-scoped
#: question carrying one of them is about that topic, not a request for the
#: whole-portfolio overview.
DEDICATED_WANTS = frozenset({"earnings", "macro", "macro_release", "guru", "ark", "briefing", "positioning", "watchlist", "alerts", "settings"})
#: Wants the overview cannot answer: they need a per-position or per-day tool.
NOT_OVERVIEW_WANTS = frozenset({"ranking", "attribution", "drawdown", "runup", "news"})
MOVE_WANTS = frozenset({"attribution", "drawdown", "runup"})
INDEX_PROXIES = ("SPY", "QQQ", "DIA")


# --- intent normalizers -----------------------------------------------------------------

@dataclass(frozen=True)
class IntentContext:
    holding: dict = field(default_factory=dict)  # history.portfolio_context: the position the last turn selected
    today: date = field(default_factory=date.today)


Normalizer = Callable[[Any, IntentContext], Any]


def broad_market(intent, context):
    """美股大盘: the index ETFs plus the macro board. A macro reading or a briefing is not a quote, so those keep their own plan."""
    if intent.market_scope != "us_broad" or set(intent.wants) & {"macro", "briefing", "macro_release", "positioning"}:
        return intent
    return intent.model_copy(update={"tickers": list(INDEX_PROXIES), "wants": ["performance", "macro"], "kind": "lookup", "portfolio_scope": False, "portfolio_followup": "", "portfolio_metric": "", "analysis_scope": "focused", "refers_back": False, "clarification": ""})


def explain_selected_position(intent, context):
    """“它为什么亏这么多”: the position the last turn ranked, over the holding period."""
    if intent.portfolio_followup != "explain_position":
        return intent
    ticker = next(iter(intent.tickers), None) or context.holding.get("ticker")
    if not ticker:
        return intent  # the graph asks which position; nothing to rewrite
    return intent.model_copy(update={"tickers": [ticker], "scope": "since_purchase", "wants": ["attribution"], "portfolio_scope": True, "date_window": None, "refers_back": False, "portfolio_metric": context.holding.get("metric", "unrealized_percent")})


def rerank_portfolio(intent, context):
    if intent.portfolio_followup != "rerank":
        return intent
    return intent.model_copy(update={"tickers": [], "portfolio_scope": True, "wants": ["ranking"], "refers_back": False})


def lab_request(intent, context):
    if not intent.lab:
        return intent
    return intent.model_copy(update={"kind": "lab", "analysis_scope": "focused", "refers_back": False})


def open_company_question(intent, context):
    """“NVDA 怎么样”: a whole-company read, not whatever the previous turn was about."""
    if intent.analysis_scope != "company" or intent.kind == "lab" or "compare" in intent.wants:
        return intent
    return intent.model_copy(update={"kind": "research", "wants": ["overview", "performance", "valuation", "risk"], "refers_back": False, "investigation": ""})


def news_only(intent, context):
    if "news" in intent.wants and set(intent.wants) <= {"news", "catalysts"}:
        return intent.model_copy(update={"wants": ["news"]})
    return intent


def default_date_window(intent, context):
    """News and recent moves get a window when the user gave none: 14 days of publication, 30 days of events."""
    explicit = intent.date_window and not (intent.scope == "recent" and not intent.date_window_explicit)
    wants = set(intent.wants)
    if explicit or not ("news" in wants or (intent.scope == "recent" and wants & MOVE_WANTS)):
        return intent
    from v2.agent_v3.contracts import DateWindow

    move = bool(wants & MOVE_WANTS)
    days = 30 if move else 14
    return intent.model_copy(update={"date_window": DateWindow(start=(context.today - timedelta(days=days - 1)).isoformat(), end=context.today.isoformat(), basis="event" if move else "publication")})


NORMALIZERS: tuple[Normalizer, ...] = (broad_market, explain_selected_position, rerank_portfolio, lab_request, open_company_question, news_only, default_date_window)


def normalize_intent(intent, context: IntentContext | None = None):
    """Apply every normalizer in order; returns (intent, names of the ones that changed it)."""
    context = context or IntentContext()
    applied: list[str] = []
    for step in NORMALIZERS:
        changed = step(intent, context)
        if changed != intent:
            applied.append(step.__name__)
        intent = changed
    return intent, applied


# --- plan rules -------------------------------------------------------------------------------

@dataclass(frozen=True)
class Facts:
    """Everything a rule may look at, so a rule never reaches into the request or the registry itself."""

    intent: Any
    text: str = ""
    metadata: dict = field(default_factory=dict)
    registered: Callable[[str], bool] = lambda name: True

    @property
    def wants(self) -> frozenset:
        return frozenset(self.intent.wants)

    @property
    def tickers(self) -> tuple[str, ...]:
        return tuple(self.intent.tickers)

    @property
    def whole_portfolio(self) -> bool:
        """The user's own book, no single name picked out, no per-holding fan-out asked for."""
        return bool(self.intent.portfolio_scope) and not self.tickers and not self.intent.each

    @property
    def page_record(self) -> bool:
        return bool((self.metadata.get("page_context", {}).get("selection") or {}).get("record_id"))


@dataclass(frozen=True)
class Rule:
    name: str
    question: str  # the kind of question, in the user's words
    when: Callable[[Facts], bool]
    build: Callable[[Facts], "tuple[PlanTask, ...] | None"]
    needs: tuple[str, ...] = ()  # skipped unless all are registered
    frame: "Callable[[Facts], dict] | None" = None
    keep_plan_flags: bool = False  # True: leave assumptions / web fallback as the shared planner set them


def _prices(tickers) -> tuple[PlanTask, ...]:
    return tuple(PlanTask(f"price-{ticker}", "market.performance", {"ticker": ticker}, purpose="dated market observations") for ticker in tickers[:4])


def _compare(f: Facts):
    return (PlanTask("compare", "research.compare", {"tickers": list(f.tickers[:4]), "dimensions": ["overview"]}), *(PlanTask(f"price-{t}", "market.performance", {"ticker": t}) for t in f.tickers[:4]))


def _ranking(f: Facts):
    metric = f.metadata.get("portfolio_metric") or ("daily_return" if f.intent.scope == "today" else "unrealized_percent")
    return (PlanTask("portfolio-ranking", "account.ranking", {"metric": metric, "direction": "high" if f.intent.rank == "high" else "low"}),)


def _pnl(f: Facts):
    return tuple(PlanTask(f"account-performance-{period}", "account.performance", {"period": period}, purpose=f"account P&L for the {period}") for period in (tuple(f.intent.periods) or ("day",)))


def _overview(f: Facts):
    tasks = [PlanTask("portfolio-overview", "account.overview", purpose="Calculate the whole portfolio before any optional investigation")]
    if "risk" in f.wants and f.registered("account.risk"):
        # Concentration comes from the overview; drawdown and the period P/L only from the risk tool.
        tasks.append(PlanTask("account-risk", "account.risk", required=False, purpose="drawdown, period P/L and concentration flags"))
    if f.registered("market.performance"):
        tasks.append(PlanTask("priority-market", "market.performance", {}, depends_on=("portfolio-overview",), required=False, fan_out={"from": "portfolio-overview", "field": "priority_tickers", "argument": "ticker", "max": 3}, purpose="Optional market context for largest exposure and loss contributors; valid for stocks and ETFs"))
    return tuple(tasks)


def _company(f: Facts):
    tasks: list[PlanTask] = []
    for ticker in f.tickers[:2]:
        tasks += [PlanTask(f"market-{ticker}", "market.performance", {"ticker": ticker}), PlanTask(f"research-{ticker}", "research.stock", {"ticker": ticker, "focus": "overview"})]
    return tuple(tasks)


def _news(f: Facts):
    # A news question is answered from three sources: the web, the filings and the
    # desk anomaly log, so the answer can say what each had.
    tasks: list[PlanTask] = []
    for ticker in f.tickers[:4]:
        tasks.append(PlanTask(f"news-{ticker}", "web.research", {"query": f.text, "topic": "company_event", "ticker": ticker, "recency_days": 14}, purpose="Read and verify news originals"))
        if f.registered("filings.recent"):
            tasks.append(PlanTask(f"filings-{ticker}", "filings.recent", {"ticker": ticker}, required=False, purpose="Filings in the same window"))
        if f.registered("market.anomaly_history"):
            tasks.append(PlanTask(f"anomalies-{ticker}", "market.anomaly_history", {"ticker": ticker, "lookback_days": 14}, required=False, purpose="Desk anomaly records in the same window"))
    return tuple(tasks)


def _quote(f: Facts):
    backdrop = (PlanTask("macro-overview", "macro.overview", purpose="market backdrop: VIX, yields, releases"),) if "macro" in f.wants and f.registered("macro.overview") else ()
    return (*_prices(f.tickers), *backdrop)


RULES: tuple[Rule, ...] = (
    Rule("compare", "A 和 B 哪个更好", lambda f: len(f.tickers) >= 2 and "compare" in f.wants, _compare),
    Rule("position_since_purchase", "我买的 X 为什么亏这么多", lambda f: f.intent.scope == "since_purchase" and bool(f.tickers) and bool(f.wants & MOVE_WANTS),
         lambda f: (PlanTask("position-analysis", "account.position_analysis", {"ticker": f.tickers[0]}),), needs=("account.position_analysis",), frame=lambda f: {"kind": "holding", "ticker": f.tickers[0]}),
    Rule("portfolio_ranking", "持仓里谁跌得最多", lambda f: bool(f.intent.portfolio_scope) and ("ranking" in f.wants or bool(f.metadata.get("portfolio_metric"))), _ranking, needs=("account.ranking",)),
    Rule("etf_holdings", "SPY 的前五大持仓", lambda f: f.metadata.get("data_target") == "etf_holdings" and bool(f.tickers),
         lambda f: tuple(PlanTask(f"holdings-{t}", "etf.holdings", {"ticker": t, "top": f.metadata.get("holdings_top", 5)}) for t in f.tickers[:4]), needs=("etf.holdings",)),
    Rule("account_earnings", "我的持仓里谁要发财报", lambda f: f.whole_portfolio and "earnings" in f.wants,
         lambda f: (PlanTask("account-earnings", "account.earnings_schedule", {"days": 14}, purpose="upcoming earnings across holdings and watchlist"),), needs=("account.earnings_schedule",)),
    Rule("account_pnl", "我今天赚了还是亏了", lambda f: f.whole_portfolio and not (f.wants & ({"ranking", "risk"} | DEDICATED_WANTS)) and (bool(f.intent.periods) or f.wants == {"performance"}), _pnl, needs=("account.performance",)),
    Rule("dedicated_topic", "带了“我的”的宏观 / 大师持仓 / 关注列表 / 简报问题", lambda f: f.whole_portfolio and bool(f.wants & DEDICATED_WANTS), lambda f: None, keep_plan_flags=True),
    Rule("portfolio_overview", "我的组合怎么样 / 最大的风险是什么", lambda f: f.whole_portfolio and f.intent.kind in {"research", "lookup"} and not (f.wants & NOT_OVERVIEW_WANTS), _overview, needs=("account.overview",)),
    Rule("company_overview", "NVDA 怎么样", lambda f: bool(f.tickers) and bool(f.wants & {"overview", "full"}) and f.intent.kind == "research", _company),
    Rule("ticker_news", "ARM 最近有什么新闻", lambda f: bool(f.tickers) and f.wants == {"news"}, _news),
    Rule("move_explanation", "AMD 今天为什么涨", lambda f: bool(f.tickers) and bool(f.wants & MOVE_WANTS) and f.intent.scope != "since_purchase" and not f.page_record,
         lambda f: tuple(PlanTask(f"move-{t}", "market.explain_move", {"ticker": t}, purpose="Verify market move before researching candidate causes") for t in f.tickers[:4])),
    # A quote is a structured market read; the shared template would use a research card.
    Rule("quote", "NVDA 现在多少钱 / 今天大盘怎么样", lambda f: f.intent.kind == "lookup" and bool(f.tickers) and f.wants in ({"performance"}, {"performance", "macro"}), _quote, keep_plan_flags=True),
)


def select_rule(facts: Facts, rules: tuple[Rule, ...] = RULES) -> Rule | None:
    for rule in rules:
        if all(facts.registered(name) for name in rule.needs) and rule.when(facts):
            return rule
    return None


def apply_rule(deterministic: ExecutionPlan, rule: Rule, facts: Facts) -> ExecutionPlan:
    """The shared plan with this rule's tasks; the rule's name travels in ``frame`` so a run shows which one fired."""
    tasks = rule.build(facts)
    frame = {**deterministic.frame, **(rule.frame(facts) if rule.frame else {}), "route_rule": rule.name}
    if tasks is None:
        return replace(deterministic, frame=frame)
    if rule.keep_plan_flags:
        return replace(deterministic, tasks=tasks, frame=frame)
    return replace(deterministic, tasks=tasks, assumptions=(), web_fallback_allowed=False, frame=frame)


def routed_plan(request, decision, registered: Callable[[str], bool]) -> tuple[ExecutionPlan, bool]:
    """The shared deterministic plan with the matching rule applied.

    Returns (plan, final). ``final`` is False when no rule chose the tasks
    (none matched, or the matching rule kept the shared plan), so the caller
    may still let the model refine a research plan.
    """
    from v2.agent_v2.planning import IntentPlanner

    deterministic = IntentPlanner().plan(request, decision)
    facts = Facts(decision.intent, request.text, request.metadata, registered)
    rule = select_rule(facts)
    if rule is None:
        return replace(deterministic, frame={**deterministic.frame, "route_rule": "shared_plan"}), False
    planned = apply_rule(deterministic, rule, facts)
    return planned, planned.tasks is not deterministic.tasks
