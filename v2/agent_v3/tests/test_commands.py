"""A state command names one of four operations; anything else is never turned into a write."""
import pytest
from pydantic import ValidationError

from v2.agent_v2.intent import Intent
from v2.agent_v2.planning import mutation_task
from v2.agent_v3.contracts import SemanticIntent


def test_unknown_operation_fails_validation_instead_of_becoming_a_watchlist_add():
    with pytest.raises(ValidationError):
        SemanticIntent.model_validate({"kind": "command", "tickers": ["NVDA"], "command": {"operation": "price_alert", "ticker": "NVDA", "price": 150}})


def test_price_alert_keeps_direction_and_price_through_to_the_task():
    intent = SemanticIntent.model_validate({"kind": "command", "tickers": ["NVDA"], "command": {"operation": "alert.add", "ticker": "NVDA", "direction": "below", "price": 150}}).domain()
    task, problem = mutation_task(intent, ("NVDA",))
    assert not problem
    assert task.arguments == {"operation": "alert.add", "payload": {"ticker": "NVDA", "direction": "below", "target_price": 150.0}}


@pytest.mark.parametrize("operation", ["", "set_alert", "buy"])
def test_planner_refuses_an_operation_it_does_not_know(operation):
    task, problem = mutation_task(Intent(kind="command", tickers=("NVDA",), command={"operation": operation}), ("NVDA",))
    assert task is None and "关注列表" in problem


def test_account_periods_are_normalised_to_the_tool_enum():
    intent = SemanticIntent.model_validate({"kind": "research", "portfolio_scope": True, "wants": ["performance"], "periods": ["today", "month_to_date", "day", "decade"]})
    assert intent.periods == ["day", "month"]


def test_wants_synonyms_fold_onto_known_objectives():
    intent = SemanticIntent.model_validate({"kind": "research", "tickers": ["NVDA", "AMD"], "wants": ["compare", "fundamentals", "made_up"]})
    assert intent.wants == ["compare", "overview"]


def test_wants_with_nothing_recognisable_still_fail():
    with pytest.raises(ValidationError):
        SemanticIntent.model_validate({"kind": "research", "tickers": ["NVDA"], "wants": ["made_up"]})


def test_a_confirmed_write_answers_with_what_changed_not_the_evidence_fallback():
    from types import SimpleNamespace
    from v2.agent_v3.adapters import register_user_state
    from v2.agent_v3.tools import Registry
    from v2.agent_v2.catalog import default_catalog

    store = SimpleNamespace(watchlist_list=lambda: [], alert_list=lambda fired: [], settings_all=lambda: {},
                            watchlist_add=lambda ticker, note="": ticker == "AMD", watchlist_remove=lambda ticker: False,
                            alert_add=lambda ticker, direction, price: 7, alert_remove=lambda alert_id: False)
    registry = Registry(default_catalog())
    register_user_state(registry, state_source=store, enable_mutations=True)
    mutate = registry.handlers["state.mutate"]
    ctx = SimpleNamespace(run_id="test-run")
    added = mutate({"operation": "watchlist.add", "payload": {"ticker": "AMD"}}, ctx)
    assert added.metadata["deterministic_answer"].startswith("已将 AMD 加入关注列表。 [v3-business-")
    present = mutate({"operation": "watchlist.add", "payload": {"ticker": "NVDA"}}, ctx)
    assert present.metadata["deterministic_answer"].startswith("NVDA 已在关注列表中，无需重复添加。")
    alert = mutate({"operation": "alert.add", "payload": {"ticker": "NVDA", "direction": "below", "target_price": 150.0}}, ctx)
    assert alert.metadata["deterministic_answer"].startswith("已为 NVDA 设置价格提醒：跌到 150.0 美元（提醒编号 7）。")
