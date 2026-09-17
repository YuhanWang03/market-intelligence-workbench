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
