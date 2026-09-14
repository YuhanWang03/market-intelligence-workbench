"""Lab questions: the long-running experiments, graded on the shape of their result.

They take minutes each and depend on the Web Lab backend, so they are their
own set (``--set lab``), never part of the smoke subset or the development
set.  Run before a release:

    python -m v2.agent_v2.eval.quality run --set lab --label lab1

The criteria name the fields a sound experiment answer must carry: the period
and the assumptions (costs, holding period), the return and the drawdown, the
parameter table and the best parameters, the event count and window and the
average abnormal return — and every one must say a backtest is not a forecast.
"""

from __future__ import annotations

from v2.agent_v2.eval.quality_cases import QualityCase
from v2.agent_v2.models import RouteKind

LAB_CASES: tuple[QualityCase, ...] = (
    QualityCase(
        "lab_backtest_momentum",
        "回测 NVDA 动量策略，持有21天，成本10bp",
        criteria=(
            "说明了回测区间和假设：持有期、交易成本、起始资金或每笔金额",
            "给出了策略的总收益或年化收益，以及最大回撤",
            "说明了这是历史回测，不代表未来收益",
        ),
        forbidden=("把回测收益写成未来收益的预期或保证",),
        expected_route=RouteKind.LAB,
        allow_web=False,
        set="lab",
        tags=("lab", "backtest"),
    ),
    QualityCase(
        "lab_sweep_holding",
        "对 NVDA 做动量策略参数扫描，持有期分别用 10、21、42 天",
        criteria=(
            "给出了参数表：每个持有期对应的收益或回撤",
            "点名了表现最好的参数组合，并说明差距有多大",
            "说明了这是历史回测，参数扫描有过拟合风险",
        ),
        forbidden=("把某个参数组合写成确定有效的选择",),
        expected_route=RouteKind.LAB,
        allow_web=False,
        set="lab",
        tags=("lab", "sweep"),
    ),
    QualityCase(
        "lab_event_study_earnings",
        "对 AAPL 和 MSFT 做财报事件研究",
        criteria=(
            "给出了事件数量和事件窗口（财报日前后几天）",
            "给出了平均异常收益或超额收益，并说明置信度或样本量的限制",
            "分别交代了两只股票的结果",
        ),
        forbidden=("把历史事件反应写成下次财报的预测",),
        expected_route=RouteKind.LAB,
        allow_web=False,
        set="lab",
        tags=("lab", "event_study"),
    ),
)
