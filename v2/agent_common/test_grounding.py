"""Figure extraction shared by both agents' verifiers: what is not a quantity."""
from v2.agent_common import grounding


def test_date_range_shorthand_is_not_a_figure():
    assert grounding.check("以下为截至2026-09-15/16的已核实证据。", "").total == 0
    assert grounding.check("窗口 09-15/16 内无事件。", "").total == 0


def test_plain_dates_and_real_figures_still_behave():
    assert grounding.check("截至 2026-09-15 的数据。", "").total == 0
    checked = grounding.check("ROIC 超过 100% 属异常高比率。", "")
    assert checked.total == 1 and checked.ungrounded == ["100"]
