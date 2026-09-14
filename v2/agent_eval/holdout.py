"""Held-out cases — questions that were never used to tune anything.

The 89 cases in ``cases.py`` are a development set: the router's signal table
was written against them and the checks were fixed against their failures.
98% routing accuracy and 95% pass rate on that set say how well the system
fits those 89 phrasings, not how it will do on the next question a user types.

These come from a different source: the questions actually sent to the
Telegram bot during the live test rounds (groups A–F, and what the user then
typed on their own), written to exercise the bot, not the eval. They are
labelled here once, from intent and from the fixture cards, without opening
``router.py``. The protocol:

* Run once, with no code change in between, and report that number.
* A failure here is either the system generalising badly or a label being
  wrong. Both are findings. A label fixed *after* looking at the run is noted
  as such in ``note``; the case is not silently repaired.
* Nothing in ``router.py`` or the checks is changed to make one of these pass.
  Once that happens the set is a development set again and this docstring is
  a lie.

Left out, with reasons:

* Exact duplicates of a development case (「我持仓里哪只最危险」= r01,
  「今天有什么值得注意的」= d01, 「为什么我的半导体仓位表现分化这么大」= c07,
  「我持仓里的半导体股票下次财报都是什么时候」= m11, and three of group A's
  regression guards: 「我的持仓」= s06, 「组合风险怎么样」= s08, 「巴菲特最新
  持仓」= s13 — a test caught those three after they were first written in).
  Re-running a case under a new id measures nothing.
* The two write requests (「把 AMD 加入我的关注列表」…). The eval registry has
  no fixtures for mutating tools; that path is covered by ``test_router.py``
  and ``test_agent.py`` directly.
* The follow-up turns (「那它财报呢」「为什么」「为什么不是 IVV」). The runner
  is single-turn; session resolution is tested in ``test_agent.py``.

Run: ``python -m v2.agent_v2.run_benchmark --holdout --modes v2_rules``
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from v2.agent_eval.cases import EvalCase

C = EvalCase

_GAP = ("没有", "未记录", "缺失", "无数据", "查不到", "未覆盖", "不可用", "数据不足",
        "无财报记录", "无该周期", "取不到")

HOLDOUT: tuple[EvalCase, ...] = (
    # --- group A: the regression guards. Three of the four are verbatim
    # development cases (s06, s08, s13) and are left out; this one survives
    # only as the user's own spelling of s01.
    C("x01", "NVDA为什么涨？", "single_lookup", "explain_move", "NVDA",
      ("explain_move",), ("portfolio_view", "risk_view"),
      (("3.85%", "3.85"),), expected_path="single_hop", max_tool_calls=2,
      note="s01 去掉空格、加问号——用户实际的打法"),
    # --- group B: one per routing signal -----------------------------------
    C("x02", "我这个月比上个月表现好还是差？", "honesty", "pnl_period", "",
      ("pnl_period",), (), (("6,120.70", "6120.70"),),
      behaviors=(_GAP,), max_tool_calls=4,
      note="只有本月卡，没有上个月；合格答案给出本月并承认无法比较。"
           "c08 的教训：不能把不存在的周期标成事实"),
    C("x03", "NVDA 和 AMD 谁更值得买？", "ranking", "unknown", "",
      ("explain_move",), (), (("3.85%", "3.85"), ("1.32%", "1.32")),
      max_tool_calls=8,
      note="「值得买」没有唯一的卡；标 explain_move 是因为两只的当日走势与相对强度"
           "是这套工具里最直接的对比依据。这是一个判断，跑完要复核"),
    C("x04", "我持仓里有没有内部人在集中卖出的？", "multi_hop", "unknown", "",
      ("portfolio_view", "insider_view"), (),
      (("CRWD",), ("3 次", "三次", "集群")), max_tool_calls=12,
      note="只有 CRWD 达到集群卖出阈值；要逐仓查 insider_view 才知道"),
    C("x05", "NVDA 的供应链上下游里，有哪几家我也持有？", "multi_hop", "chain", "NVDA",
      ("chain", "portfolio_view"), (), (("AMD",), ("MSFT",)), max_tool_calls=4,
      note="产业链卡：下游 MSFT/META，同业 AMD/AVGO；持仓里有 MSFT 和 AMD"),

    # --- group C: aimed at grounding and attribution -----------------------
    C("x06", "QCOM 亏这么多，是资金在流出还是只是估值回调？", "honesty",
      "moneyflow_view", "QCOM", ("moneyflow_view",), (), (),
      behaviors=(_GAP,), forbidden=("-0.31", "-0.18", "-0.21", "28.4"),
      expected_path="single_hop", max_tool_calls=4,
      note="QCOM 不在 fixture 里；问题里的「亏这么多」是用户的前提，不是数据"),
    C("x07", "把我持仓按风险从高到低排个序，说明理由", "ranking", "unknown", "",
      ("portfolio_view",), (), (("SMCI",), ("CRWD",)), max_tool_calls=14,
      note="排序本身没有唯一答案；断言只到「SMCI 与 CRWD 必须被点名」——"
           "一个已回撤 -21.5% 且上季 miss，一个集中度超标且财报 D-3"),
    C("x08", "NVDA、AMD、MU 这三家谁的财报最近，各自上次表现如何？", "ranking",
      "earnings_view", "", ("earnings_view",), (),
      (("2026-11-03", "11-03", "11 月 3"), ("5.6%", "5.6"), ("2.9%", "2.9")),
      behaviors=(_GAP,), max_tool_calls=6,
      note="AMD 11-03 最近（NVDA 11-19）；MU 无记录，必须说明"),

    # --- group F: errors and edges ------------------------------------------
    C("x09", "ZZZZ 怎么样？", "honesty", "summary", "ZZZZ",
      ("summary",), (), (), behaviors=(_GAP,),
      forbidden=("448.77", "226.66", "75.48"),
      expected_path="single_hop", max_tool_calls=4,
      note="不存在的 ticker；不能拿任何一只真实持仓的价格顶上"),

    # --- the two the router left on the fast path in the dry run ------------
    C("x10", "木头姐最近在买什么，和我持仓有重叠吗？", "multi_hop", "etf_view", "",
      ("etf_view", "portfolio_view"), (), (("PLTR",), ("TSLA",)), max_tool_calls=6,
      note="「最近在买」= ARKK 当日买入 PLTR；重叠 = TSLA（ARKK 第一大、持仓 9.7%）。"
           "预演时路由把它留在了快路径"),
    C("x11", "CPI 和 NFP 最近怎么样，对我持仓有什么影响？", "compound", "release_check", "",
      ("release_check",), (), (("2.70%", "2.7%"), ("14.2", "142")),
      max_tool_calls=8,
      note="两个序列各一张卡；「影响」部分不设断言，fixture 里没有可引用的因果"),

    # --- what the user then typed on their own ------------------------------
    C("x12", "我的持仓最近整体在跌还是涨？", "single_lookup", "pnl_view", "",
      (), (),
      (("3,880.12", "3880.12", "1.93%", "1,204.33", "1204.33", "0.65%"),),
      expected_path="single_hop", max_tool_calls=4,
      note="「最近」既可读作今日也可读作本周；两张卡的数字任一个都算。"
           "must_call 留空，因为两条路都对"),
    C("x13", "我的持仓最近整体在跌还是张？", "single_lookup", "pnl_view", "",
      (), (),
      (("3,880.12", "3880.12", "1.93%", "1,204.33", "1204.33", "0.65%"),),
      expected_path="single_hop", max_tool_calls=4,
      note="x12 带错别字（张/涨），用户真的这么打过"),
    C("x14", "谁的财报最近？", "cost_trap", "earnings_calendar", "",
      ("earnings_calendar",), ("portfolio_view", "earnings_view"),
      (("CRWD",), ("2026-09-06", "9-06", "9 月 6", "D-3")),
      expected_path="single_hop", max_tool_calls=2),
    C("x15", "NVDA怎么样", "single_lookup", "summary", "NVDA",
      ("summary",), ("portfolio_view",),
      (("226.66", "3.85%", "3.85"),), expected_path="single_hop", max_tool_calls=3),
)

HOLDOUT_EXTRAS: dict[str, dict[str, Any]] = {
    "x02": {"period": "month"},
    "x10": {"etf": "ARKK"},
    "x11": {"release_type": "cpi"},
}

HOLDOUT = tuple(
    replace(case, extra=HOLDOUT_EXTRAS[case.id]) if case.id in HOLDOUT_EXTRAS else case
    for case in HOLDOUT
)
