"""The routing tables, walked as a grid instead of one question at a time.

Every V3 routing bug the bench found had one shape: the classifier read the
question correctly and a broader branch downstream took it anyway. These
tests state the property those bugs violated, over every want, so the next
variant of the same question is covered before anyone asks it.
"""
from dataclasses import replace
from datetime import date

import pytest

from v2.agent_v2.intent import WANTS
from v2.agent_v2.models import NormalizedRequest
from v2.agent_v2.routing import route
from v2.agent_v3.contracts import SemanticIntent
from v2.agent_v3.routing import DEDICATED_WANTS, NORMALIZERS, RULES, Facts, IntentContext, normalize_intent, routed_plan, select_rule

EVERYTHING = lambda name: True  # noqa: E731 — a registry with every capability


def _plan(registered=EVERYTHING, metadata=None, **intent):
    semantic = SemanticIntent.model_validate({"kind": "research", **intent})
    request = NormalizedRequest("q", "q", session_id="s", entities=tuple(semantic.tickers), allow_web=True,
                                metadata={"data_target": semantic.data_target, "holdings_top": semantic.holdings_top, "portfolio_metric": semantic.portfolio_metric, **(metadata or {})})
    decision = route(request, intent=semantic.domain())
    plan, final = routed_plan(request, decision, registered)
    return plan, final


def _capabilities(plan) -> list[str]:
    return [task.capability for task in plan.tasks]


#: The tool each dedicated want must reach when the question is about the user's own book.
DEDICATED_TOOL = {
    "earnings": "account.earnings_schedule", "macro": "macro.overview", "macro_release": "macro.release", "guru": "institutional.manager_portfolio",
    "ark": "etf.ark_activity", "briefing": "account.earnings_schedule", "positioning": "account.risk", "watchlist": "state.read", "alerts": "state.read", "settings": "state.read",
}
_EXTRA = {"macro_release": {"release": "cpi"}, "guru": {"managers": ["buffett"]}, "ark": {"ark_etfs": ["ARKK"]}}


def test_every_dedicated_want_has_a_tool_listed_here():
    assert set(DEDICATED_TOOL) == set(DEDICATED_WANTS) and DEDICATED_WANTS <= set(WANTS)


@pytest.mark.parametrize("kind", ["lookup", "research"])
@pytest.mark.parametrize("want", sorted(DEDICATED_WANTS))
def test_a_portfolio_scoped_question_with_a_dedicated_want_reaches_its_tool_not_the_overview(want, kind):
    # "我的持仓里谁要发财报" / "帮我看看我关注的宏观" : portfolio_scope is set, the want names the topic.
    plan, _ = _plan(kind=kind, wants=[want], portfolio_scope=True, **_EXTRA.get(want, {}))
    names = _capabilities(plan)
    assert DEDICATED_TOOL[want] in names, (want, names)
    assert "account.overview" not in names, (want, names)


@pytest.mark.parametrize("want", sorted(set(WANTS) - {"committee", "backtest", "sweep", "event_study", "screen", "momentum", "insider", "pead"}))
@pytest.mark.parametrize("portfolio, tickers", [(True, []), (False, []), (False, ["NVDA"]), (True, ["NVDA"])])
def test_the_whole_grid_plans_without_error_and_names_its_rule(want, portfolio, tickers):
    plan, _ = _plan(kind="lookup", wants=[want], portfolio_scope=portfolio, tickers=tickers, **_EXTRA.get(want, {}))
    assert plan.frame.get("route_rule"), (want, portfolio, tickers)


@pytest.mark.parametrize("intent, rule, capabilities", [
    ({"wants": ["compare"], "tickers": ["MU", "SNDK"]}, "compare", ["research.compare", "market.performance", "market.performance"]),
    ({"wants": ["attribution"], "tickers": ["ARM"], "scope": "since_purchase", "portfolio_scope": True}, "position_since_purchase", ["account.position_analysis"]),
    ({"wants": ["ranking"], "portfolio_scope": True}, "portfolio_ranking", ["account.ranking"]),
    ({"wants": ["ranking"], "portfolio_scope": True, "tickers": ["ARM"]}, "portfolio_ranking", ["account.ranking"]),
    ({"wants": ["portfolio"], "tickers": ["SPY"], "data_target": "etf_holdings"}, "etf_holdings", ["etf.holdings"]),
    ({"wants": ["earnings"], "portfolio_scope": True, "scope": "window"}, "account_earnings", ["account.earnings_schedule"]),
    ({"wants": ["portfolio", "performance"], "portfolio_scope": True, "periods": ["today", "month_to_date"]}, "account_pnl", ["account.performance", "account.performance"]),
    ({"wants": ["portfolio"], "portfolio_scope": True, "periods": ["day"], "kind": "lookup"}, "account_pnl", ["account.performance"]),
    ({"wants": ["performance"], "portfolio_scope": True}, "account_pnl", ["account.performance"]),
    ({"wants": ["portfolio", "risk"], "portfolio_scope": True}, "portfolio_overview", ["account.overview", "account.risk", "market.performance"]),
    ({"wants": ["overview"], "portfolio_scope": True}, "portfolio_overview", ["account.overview", "market.performance"]),
    ({"wants": ["overview", "performance", "valuation", "risk"], "tickers": ["NVDA"]}, "company_overview", ["market.performance", "research.stock"]),
    ({"wants": ["news"], "tickers": ["ARM"]}, "ticker_news", ["web.research", "filings.recent", "market.anomaly_history"]),
    ({"wants": ["attribution"], "tickers": ["AMD"], "scope": "today"}, "move_explanation", ["market.explain_move"]),
    ({"wants": ["runup", "attribution"], "tickers": ["AMD"], "scope": "recent"}, "move_explanation", ["market.explain_move"]),
    ({"kind": "lookup", "wants": ["performance"], "tickers": ["NVDA"]}, "quote", ["market.performance"]),
    ({"kind": "lookup", "wants": ["performance", "macro"], "tickers": ["SPY", "QQQ", "DIA"]}, "quote", ["market.performance"] * 3 + ["macro.overview"]),
])
def test_each_rule_fires_for_its_question(intent, rule, capabilities):
    plan, final = _plan(**intent)
    assert plan.frame["route_rule"] == rule and final
    assert _capabilities(plan) == capabilities


def test_a_rule_whose_tool_is_not_registered_is_skipped_not_failed():
    without = lambda name: name not in {"account.earnings_schedule", "account.overview"}  # noqa: E731
    plan, _ = _plan(registered=without, wants=["earnings"], portfolio_scope=True)
    assert plan.frame["route_rule"] != "account_earnings"
    assert select_rule(Facts(SemanticIntent.model_validate({"kind": "research", "wants": ["overview"], "portfolio_scope": True}).domain(), registered=without)) is None


def test_a_question_no_rule_claims_keeps_the_shared_plan_and_may_be_refined():
    plan, final = _plan(wants=["valuation"], tickers=["NVDA"])
    assert plan.frame["route_rule"] == "shared_plan" and not final and "research.stock" in _capabilities(plan)


def test_rule_names_are_unique_and_every_rule_says_what_it_is_for():
    names = [rule.name for rule in RULES]
    assert len(names) == len(set(names)) and all(rule.question for rule in RULES)
    assert len({step.__name__ for step in NORMALIZERS}) == len(NORMALIZERS)


# --- normalizers ------------------------------------------------------------------------------

def _intent(**values):
    return SemanticIntent.model_validate({"kind": "lookup", **values})


def test_the_broad_market_becomes_index_quotes_with_the_macro_board():
    out, applied = normalize_intent(_intent(wants=["market"], market_scope="us_broad", scope="today", portfolio_scope=True))
    assert applied == ["broad_market"] and out.tickers == ["SPY", "QQQ", "DIA"] and out.wants == ["performance", "macro"] and not out.portfolio_scope


@pytest.mark.parametrize("wants", [["macro"], ["market", "briefing"], ["macro_release"], ["positioning", "market"]])
def test_a_macro_reading_or_a_briefing_is_not_reduced_to_index_quotes(wants):
    out, applied = normalize_intent(_intent(wants=wants, market_scope="us_broad"))
    assert "broad_market" not in applied and out.wants == wants and not out.tickers


def test_explaining_the_ranked_position_inherits_it_over_the_holding_period():
    out, applied = normalize_intent(_intent(kind="research", wants=["attribution"], portfolio_followup="explain_position"), IntentContext(holding={"ticker": "ARM", "metric": "unrealized_amount"}))
    assert "explain_selected_position" in applied and out.tickers == ["ARM"] and out.scope == "since_purchase" and out.portfolio_metric == "unrealized_amount"
    unchanged, applied = normalize_intent(_intent(kind="research", wants=["attribution"], portfolio_followup="explain_position"))
    assert "explain_selected_position" not in applied and not unchanged.tickers


def test_news_gets_a_fortnight_and_a_recent_move_gets_a_month():
    today = date(2026, 9, 18)
    news, applied = normalize_intent(_intent(wants=["news", "catalysts"], tickers=["ARM"]), IntentContext(today=today))
    assert applied == ["news_only", "default_date_window"] and news.wants == ["news"]
    assert (news.date_window.start, news.date_window.end, news.date_window.basis) == ("2026-09-05", "2026-09-18", "publication")
    move, _ = normalize_intent(_intent(wants=["runup"], tickers=["AMD"], scope="recent"), IntentContext(today=today))
    assert (move.date_window.start, move.date_window.basis) == ("2026-08-20", "event")


def test_an_open_company_question_is_a_full_read_whatever_the_last_turn_was():
    out, applied = normalize_intent(_intent(wants=["news"], tickers=["NVDA"], analysis_scope="company", refers_back=True))
    assert applied[0] == "open_company_question" and out.kind == "research" and out.wants == ["overview", "performance", "valuation", "risk"] and not out.refers_back


def test_an_untouched_intent_reports_no_normalizers():
    intent = _intent(wants=["performance"], tickers=["NVDA"])
    out, applied = normalize_intent(intent)
    assert out == intent and applied == []
