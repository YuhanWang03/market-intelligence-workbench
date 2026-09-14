"""The labelled evaluation set.

Each case says three things about one question: which tools the answer *needs*,
which facts must survive into the reply, and how much it may cost. That triple is
what turns "the answer looks good" into a number.

Labelling decisions worth knowing before reading the scores:

* ``must_call`` is the **minimal** set. A run that calls extra tools still passes
  the recall check — over-calling is charged against the cost metrics instead,
  which is where it belongs.
* ``must_not_call`` marks tools that would be actively wrong (a portfolio tool on
  a question about a stock the user does not own) or pure waste on a question one
  card already answers. These are the cost traps.
* ``facts`` are tuples of *acceptable surface forms* for one fact, because a
  model may write 2026-09-06 or 9月6日 and both are correct. Exact-substring
  scoring without this measures phrasing, not correctness.
* ``expected_path`` is what the router should choose, so routing accuracy is
  scored on the same set as answer quality rather than on its own toy list.

Cases the current system is expected to fail are included on purpose — a suite
tuned to pass is a suite that cannot detect a regression or justify a change.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


@dataclass(frozen=True)
class EvalCase:
    id: str
    query: str
    category: str
    intent: str = "unknown"                 # what v2/bot/intent.py returns
    ticker: str = ""
    must_call: tuple[str, ...] = ()
    #: Tools that are merely unnecessary for this question. Counted as **cost,
    #: never as a failure** — the same rule ``must_call`` already follows
    #: ("extra tools are charged to the cost metrics, not to correctness").
    #: Scoring them as errors double-counted waste: t02 failed 3/3 for calling
    #: one redundant tool while producing an entirely correct answer.
    wasteful_tools: tuple[str, ...] = ()
    #: Assertions about *data*: each must be quotable from the recorded
    #: fixtures, and a test enforces that.
    facts: tuple[tuple[str, ...], ...] = ()
    #: Assertions about *behaviour* — that the answer acknowledged a gap, named
    #: its capabilities, refused to guess. These deliberately do not exist in
    #: any fixture, so they are exempt from the answer-key check while still
    #: gating the case the same way facts do.
    behaviors: tuple[tuple[str, ...], ...] = ()
    forbidden: tuple[str, ...] = ()
    expected_path: str = "agent"
    max_tool_calls: int = 8
    note: str = ""
    #: Extra fields the real classifier would emit for this query (manager, etf,
    #: release_type, period …). Without them the baseline is handicapped by the
    #: harness rather than by its own design, and the comparison lies.
    extra: dict[str, Any] = field(default_factory=dict)
    #: Tools whose use would make the answer *wrong*. Reserved for genuine
    #: correctness breaks; in this tool set reading an extra card never makes an
    #: answer wrong, so it stays empty and wrongness is caught by ``forbidden``.
    must_not_call: tuple[str, ...] = ()


C = EvalCase

CASES: tuple[EvalCase, ...] = (
    # =======================================================================
    # 1. single_lookup — one card is the whole answer (18)
    # =======================================================================
    C("s01", "NVDA 为什么涨", "single_lookup", "explain_move", "NVDA",
      ("explain_move",), ("portfolio_view", "risk_view"),
      (("3.85%", "3.85"), ("SMH",)), expected_path="single_hop", max_tool_calls=2),
    C("s02", "TSLA 什么时候发财报", "single_lookup", "earnings_view", "TSLA",
      ("earnings_view",), ("portfolio_view",),
      (("2026-10-21", "10-21", "10 月 21"),), expected_path="single_hop", max_tool_calls=2),
    C("s03", "MSFT 资金流怎么样", "single_lookup", "moneyflow_view", "MSFT",
      ("moneyflow_view",), (), (("0.14",), ("58.4",)),
      expected_path="single_hop", max_tool_calls=2),
    C("s04", "NVDA 内部人交易", "single_lookup", "insider_view", "NVDA",
      ("insider_view",), (), (("10b5-1", "12,000", "12000"),),
      expected_path="single_hop", max_tool_calls=2),
    C("s05", "AVGO 最近有什么 8-K", "single_lookup", "eight_k_view", "AVGO",
      ("eight_k_view",), (), (("4.10B", "4.1B", "41 亿"),),
      expected_path="single_hop", max_tool_calls=2),
    C("s06", "我的持仓", "single_lookup", "portfolio_view", "",
      ("portfolio_view",), (), (("CRWD",), ("184,320.55", "184320.55")),
      expected_path="single_hop", max_tool_calls=2),
    C("s07", "我的当日盈亏", "single_lookup", "pnl_view", "",
      ("pnl_view",), (), (("1,204.33", "1204.33"),),
      expected_path="single_hop", max_tool_calls=2),
    C("s08", "组合风险怎么样", "single_lookup", "risk_view", "",
      ("risk_view",), (), (("54.7%", "54.7"), ("22.4%", "22.4")),
      expected_path="single_hop", max_tool_calls=2),
    C("s09", "宏观怎么样", "single_lookup", "macro_view", "",
      ("macro_view",), (), (("18.40", "18.4"), ("4.18%", "4.18")),
      expected_path="single_hop", max_tool_calls=2),
    C("s10", "最近 CPI", "single_lookup", "release_check", "",
      ("release_check",), (), (("2.70%", "2.7%"), ("3.10%", "3.1%")),
      expected_path="single_hop", max_tool_calls=2),
    C("s11", "上次 FOMC 说了什么", "single_lookup", "release_check", "",
      ("release_check",), (), (("4.25", ), ("2026-09-17", "9-17", "9 月 17")),
      expected_path="single_hop", max_tool_calls=2),
    C("s12", "我关注了哪些股票", "single_lookup", "watchlist_view", "",
      ("watchlist_view",), ("portfolio_view",),
      (("ARM",), ("PLTR",), ("AVGO",)),
      expected_path="single_hop", max_tool_calls=2),
    C("s13", "巴菲特最新持仓", "single_lookup", "thirteen_f", "",
      ("institutional_13f",), (),
      (("57.80B", "57.8B", "578 亿", "578亿"), ("AAPL",)),
      expected_path="single_hop", max_tool_calls=2,
      note="模型会把 $57.80B 写成「578 亿」；只列英文单位形式等于在测措辞而非正确性"),
    C("s14", "谁在持有 NVDA", "single_lookup", "holders_view", "NVDA",
      ("holders",), (), (("Vanguard",), ("8.94%", "8.94")),
      expected_path="single_hop", max_tool_calls=2),
    C("s15", "ARKK 今天买了什么", "single_lookup", "etf_view", "",
      ("etf_view",), (), (("PLTR",), ("42,000", "42000")),
      expected_path="single_hop", max_tool_calls=2),
    C("s16", "NVDA 的产业链", "single_lookup", "chain", "NVDA",
      ("chain",), (), (("TSM",), ("ASML",)),
      expected_path="single_hop", max_tool_calls=2),
    C("s17", "我的价格提醒有哪些", "single_lookup", "alert_list", "",
      ("alert_list",), (), (("无未触发", "没有", "无提醒", "暂无"),),
      expected_path="single_hop", max_tool_calls=2),
    C("s18", "推送阈值是多少", "single_lookup", "settings", "",
      ("settings_view",), (), (("3.0%", "3%"), ("2.5", )),
      expected_path="single_hop", max_tool_calls=2),

    # =======================================================================
    # 2. multi_hop_portfolio — enumerate the book, then look each one up (14)
    # =======================================================================
    C("m01", "我持仓里有哪些快发财报了", "multi_hop", "earnings_calendar", "",
      ("earnings_calendar",), (),
      (("CRWD",), ("SMCI",), ("2026-09-06", "9-06", "9 月 6")),
      expected_path="single_hop", max_tool_calls=2,
      note="财报日历卡本身就带持仓标注 —— 初版要求先调 portfolio_view 是我把它"
           "想复杂了，一张卡就够"),
    C("m02", "我持仓里有没有内部人在卖", "multi_hop", "insider_view", "",
      ("portfolio_view", "insider_view"), (),
      (("CRWD",), ("3 次", "三次", "3次")), max_tool_calls=10),
    C("m03", "我的持仓最近有什么 SEC 申报", "multi_hop", "eight_k_view", "",
      ("portfolio_view", "eight_k_view"), (),
      (("CRWD",), ("CFO",)), max_tool_calls=10),
    C("m04", "我持仓里哪些在亏钱", "multi_hop", "portfolio_view", "",
      ("portfolio_view",), (),
      (("SMCI",), ("TSLA",), ("AMD",)),
      expected_path="single_hop", max_tool_calls=2),
    C("m05", "我的组合里半导体占多少，风险大吗", "multi_hop", "risk_view", "",
      ("risk_view",), (), (("38.1%", "38.1"),),
      expected_path="single_hop", max_tool_calls=2),
    C("m06", "我持仓里资金在流出的有哪些", "multi_hop", "portfolio_view", "",
      ("portfolio_view", "moneyflow_view"), (),
      (("SMCI",), ("TSLA",)), max_tool_calls=12),
    C("m07", "关注列表里那几只最近怎么样", "multi_hop", "watchlist_view", "",
      ("watchlist_view", "explain_move"), (),
      (("ARM",), ("7.42%", "7.42")), max_tool_calls=8),
    C("m08", "我持仓里跌得最狠的那只是什么原因", "multi_hop", "portfolio_view", "",
      ("portfolio_view", "explain_move"), (),
      (("SMCI",), ("5.40%", "5.4%")), max_tool_calls=6),
    C("m09", "我的持仓有没有踩到 going concern 或者高管离职的", "multi_hop", "unknown", "",
      ("portfolio_view", "eight_k_view"), (),
      (("CRWD",), ("CFO",), ("2026-10-01", "10-01", "10 月 1")), max_tool_calls=10),
    C("m10", "帮我看看这周该关注持仓里的谁", "multi_hop", "unknown", "",
      ("portfolio_view",), (), (("CRWD",),), max_tool_calls=10),
    C("m11", "我持仓里的半导体股票下次财报都是什么时候", "multi_hop", "earnings_calendar", "",
      ("portfolio_view", "earnings_view"), (),
      (("2026-11-19", "11-19"), ("2026-11-03", "11-03")), max_tool_calls=10),
    C("m12", "关注列表里有没有快发财报的", "multi_hop", "earnings_calendar", "",
      ("earnings_calendar",), (),
      (("AVGO",), ("2026-09-11", "9-11", "9 月 11")),
      expected_path="single_hop", max_tool_calls=3,
      note="财报日历卡把关注列表成员也标注了，一张卡即可回答 —— must_call 应当是"
           "最小必要集合，我原先写的 watchlist_view + earnings_view 不是最小的"),
    C("m13", "我持仓里有几只这周有事件", "multi_hop", "unknown", "",
      ("portfolio_view",), (), (("CRWD",),), max_tool_calls=10),
    C("m14", "我的组合和 ARKK 有重叠吗", "multi_hop", "unknown", "",
      ("portfolio_view", "etf_view"), (), (("TSLA",),), max_tool_calls=6),

    # =======================================================================
    # 3. ranking — a verdict across several subjects (10)
    # =======================================================================
    C("r01", "我持仓里哪只最危险", "ranking", "risk_view", "",
      ("portfolio_view", "earnings_view"), (),
      (("CRWD",), ("22.4%", "22.4")), max_tool_calls=10),
    C("r02", "我持仓里哪只跌得最多", "ranking", "portfolio_view", "",
      ("portfolio_view",), (), (("SMCI",), ("21.5%", "21.5")),
      expected_path="single_hop", max_tool_calls=2),
    C("r03", "我持仓里哪只占比最高", "ranking", "portfolio_view", "",
      ("portfolio_view",), (), (("CRWD",), ("22.4%", "22.4")),
      expected_path="single_hop", max_tool_calls=2),
    C("r04", "NVDA 和 AMD 谁的财报更好", "ranking", "earnings_view", "NVDA",
      ("earnings_view",), ("portfolio_view",),
      (("5.6%", "5.6"), ("2.9%", "2.9")), max_tool_calls=4,
      note="必须同时给出两家的 surprise，否则只查一家也能蒙过"),
    C("r05", "AAPL 和 MSFT 谁的资金流更强", "ranking", "moneyflow_view", "AAPL",
      ("moneyflow_view",), (), (("MSFT",), ("0.14",)), max_tool_calls=4),
    C("r06", "CRWD 和 SMCI 哪个财报风险更大", "ranking", "earnings_view", "CRWD",
      ("earnings_view",), (), (("SMCI",), ("23.6%", "23.6")), max_tool_calls=4),
    C("r07", "watchlist 里哪只最值得关注", "ranking", "watchlist_view", "",
      ("watchlist_view", "explain_move"), (), (("ARM",), ("7.42%", "7.42", "2.40B", "2.4B")),
      max_tool_calls=8,
      note="要给出选它的理由，只念一遍关注列表不算回答"),
    C("r08", "我持仓里哪只的内部人卖得最凶", "ranking", "insider_view", "",
      ("portfolio_view", "insider_view"), (),
      (("CRWD",), ("10.97M", "10,97")), max_tool_calls=12),
    C("r09", "TSLA 和 PLTR 哪个逆势更严重", "ranking", "explain_move", "TSLA",
      ("explain_move",), (), (("PLTR",), ("5.58", )), max_tool_calls=4),
    C("r10", "我持仓里哪只离财报最近", "ranking", "earnings_calendar", "",
      ("earnings_calendar",), (), (("CRWD",), ("2026-09-06", "9-06", "9 月 6")),
      expected_path="single_hop", max_tool_calls=2),

    # =======================================================================
    # 4. causal — why, spanning the book (8)
    # =======================================================================
    C("c01", "我这周为什么亏钱", "causal", "pnl_period", "",
      ("pnl_period",), (), (("SMCI",), ("2,940.55", "2940.55")),
      expected_path="single_hop", max_tool_calls=3),
    C("c02", "我这个月亏了多少，主要是谁拖的", "causal", "pnl_period", "",
      ("pnl_period",), (), (("6,120.70", "6120.70"),),
      expected_path="single_hop", max_tool_calls=3,
      note="月度卡只有总额，fixture 里没有月度归因 —— 断言只到总额为止"),
    C("c03", "组合回撤是怎么来的", "causal", "risk_view", "",
      ("risk_view",), (), (("4.20%", "4.2%"), ("SMCI",)),
      expected_path="single_hop", max_tool_calls=3),
    C("c04", "SMCI 最近出什么事了", "causal", "explain_move", "SMCI",
      ("explain_move",), (), (("5.40%", "5.4%"),),
      expected_path="single_hop", max_tool_calls=3),
    C("c05", "CRWD 为什么跌，是基本面问题吗", "causal", "explain_move", "CRWD",
      ("explain_move",), (), (("2.10%", "2.1%"),),
      expected_path="single_hop", max_tool_calls=3),
    C("c06", "ARM 今天为什么大涨", "causal", "explain_move", "ARM",
      ("explain_move",), (), (("7.42%", "7.42"),),
      expected_path="single_hop", max_tool_calls=3),
    C("c07", "为什么我的半导体仓位表现分化这么大", "causal", "unknown", "",
      ("portfolio_view",), (), (("SMCI",), ("NVDA",)), max_tool_calls=10),
    C("c08", "我这周亏的钱今天补回来了吗", "causal", "pnl_period", "",
      ("pnl_period", "pnl_view"), (),
      (("3,880.12", "3880.12"), ("1,204.33", "1204.33")), max_tool_calls=6,
      note="要对比两个周期，一张卡答不了 —— 初版把两个数写成同一条事实的备选，"
           "被单跳蒙混过关。问句原本写的是「上周…这周」,而 pnl_period 只有 "
           "day/week/month,根本取不到上周:断言要的 3,880.12 是本周、1,204.33 "
           "是今日,正确回答「没有上周的数据」反而会挂掉两条事实。0/3 稳定失败,"
           "查下来又是标注和 fixture 对不上,不是模型不会规划。"),

    # =======================================================================
    # 5. compound — two asks in one message (8)
    # =======================================================================
    C("p01", "AAPL 财报怎么样，另外内部人有没有在卖", "compound", "earnings_view", "AAPL",
      ("earnings_view", "insider_view"), (),
      (("2026-10-29", "10-29"), ("COO", "41,000", "9.19")), max_tool_calls=4),
    C("p02", "NVDA 涨了吗？资金流呢？", "compound", "explain_move", "NVDA",
      ("explain_move", "moneyflow_view"), (), (("3.85%", "3.85"), ("0.26",)),
      max_tool_calls=4),
    C("p03", "CRWD 什么时候财报，8-K 有什么", "compound", "earnings_view", "CRWD",
      ("earnings_view", "eight_k_view"), (),
      (("2026-09-06", "9-06"), ("CFO",)), max_tool_calls=4),
    C("p04", "我的当日盈亏和组合风险", "compound", "pnl_view", "",
      ("pnl_view", "risk_view"), (), (("1,204.33", "1204.33"), ("54.7%", "54.7")),
      max_tool_calls=4),
    C("p05", "宏观怎么样，还有最近 CPI", "compound", "macro_view", "",
      ("macro_view", "release_check"), (), (("18.40", "18.4"), ("2.70%", "2.7%")),
      max_tool_calls=4),
    C("p06", "TSLA 为什么跌，内部人在卖吗，财报什么时候", "compound", "explain_move", "TSLA",
      ("explain_move", "insider_view", "earnings_view"), (),
      (("2.74%", "2.74"), ("2026-10-21", "10-21")), max_tool_calls=6),
    C("p07", "AMD 的资金流和 8-K", "compound", "moneyflow_view", "AMD",
      ("moneyflow_view", "eight_k_view"), (), (("0.09",),), max_tool_calls=4),
    C("p08", "巴菲特买了什么，ARKK 又买了什么", "compound", "thirteen_f", "",
      ("institutional_13f", "etf_view"), (), (("AAPL",), ("PLTR",)), max_tool_calls=4),

    # =======================================================================
    # 6. recovery — a data source is down; route around it (5)
    # =======================================================================
    C("v01", "SMCI 最近有什么 8-K", "recovery", "eight_k_view", "SMCI",
      ("eight_k_view",), (), (),
      behaviors=(("超时", "失败", "无法", "未能", "取不到", "不可用", "数据缺", "没能"),),
      expected_path="single_hop", max_tool_calls=6,
      note="8-K 源必然超时；合格答案必须承认数据缺口，而不是编造申报。"
           "路由期望是 single_hop：工具会不会超时是运行时的事，"
           "先验的路由决策无从预知，要求它预知是不公平的标注"),
    C("v02", "SMCI 最近出什么事了，SEC 有申报吗", "recovery", "eight_k_view", "SMCI",
      ("eight_k_view", "explain_move"), (), (("5.40%", "5.4%"),), max_tool_calls=6,
      note="8-K 失败后应改用异动归因补上"),
    C("v03", "我持仓里最近有 SEC 重大事项的是哪几只", "recovery", "unknown", "",
      ("portfolio_view", "eight_k_view"), (), (("CRWD",),), max_tool_calls=12,
      note="批量查 8-K 时 SMCI 会失败，其余结果不能被一条失败带走"),
    C("v04", "SMCI 值不值得继续拿", "recovery", "unknown", "SMCI",
      ("earnings_view",), (), (("SMCI",),), max_tool_calls=8),
    C("v05", "SMCI 的申报和财报都看一下", "recovery", "unknown", "SMCI",
      ("eight_k_view", "earnings_view"), (), (("2026-09-09", "9-09"),), max_tool_calls=6),

    # =======================================================================
    # 7. honesty — the data isn't there; say so instead of inventing (7)
    # =======================================================================
    C("h01", "GOOGL 的资金流怎么样", "honesty", "moneyflow_view", "GOOGL",
      ("moneyflow_view",), (), (), forbidden=("CMF(20) -0.31", "CMF(20) +0.26"),
      expected_path="single_hop", max_tool_calls=3,
      note="fixture 未记录 GOOGL 资金流；不能把别的 ticker 的数字搬过来"),
    C("h02", "AVGO 的资金流和 RSI", "honesty", "moneyflow_view", "AVGO",
      ("moneyflow_view",), (), (), forbidden=("RSI(14) 63.7", "RSI(14) 28.4"),
      expected_path="single_hop", max_tool_calls=3),
    C("h03", "Burry 最新持仓", "honesty", "thirteen_f", "",
      ("institutional_13f",), (), (("BABA", "JD", "9 个", "84.30"),),
      expected_path="single_hop", max_tool_calls=3),
    C("h04", "ARKQ 的持仓", "honesty", "etf_view", "",
      ("etf_view",), (), (),
      behaviors=(("没有", "未记录", "缺失", "无数据", "查不到", "未覆盖", "不可用"),),
      expected_path="single_hop", max_tool_calls=3,
      note="只记录了 ARKK。判据是「有没有承认 ARKQ 缺数据」；"
           "「有没有把 ARKK 的数字当成 ARKQ 的」交给 attribution 检查 —— "
           "用禁止字符串表达后者会误伤「ARKQ 无数据，可参考 ARKK…」这种正确写法"),
    C("h05", "TSLA 的产业链", "honesty", "chain", "TSLA",
      ("chain",), (), (), forbidden=("TSM", "ASML"),
      expected_path="single_hop", max_tool_calls=3),
    C("h06", "PLTR 的综合分析", "honesty", "summary", "PLTR",
      ("summary",), (), (), forbidden=("448.77", "226.66"),
      expected_path="single_hop", max_tool_calls=4),
    C("h07", "我持仓里每只的机构持股比例", "honesty", "unknown", "",
      ("portfolio_view", "holders"), (), (),
      behaviors=(("没有", "未记录", "缺失", "无数据", "查不到", "未覆盖", "不可用"),),
      max_tool_calls=12,
      note="只有 NVDA/CRWD 有机构数据，其余必须说明缺失。"
           "原判据是禁止字符串 Vanguard 8.94% —— 但那正是 NVDA 的真实数据，"
           "任何正确回答都会包含它，这条 case 因此从构造上不可能通过。"
           "我为此改了两轮 prompt、怪了两轮模型"),

    # =======================================================================
    # 8. dead_end — the classifier gives up; today the user gets "没听懂" (7)
    # =======================================================================
    C("d01", "今天有什么值得注意的", "dead_end", "unknown", "",
      ("portfolio_view",), (), (("CRWD", "SMCI", "NVDA", "AMD", "TSLA"),),
      max_tool_calls=10, note="必须落到具体持仓，不能只给一段泛泛的市场感想"),
    C("d02", "帮我看看要不要减仓", "dead_end", "unknown", "",
      ("portfolio_view",), (), (("CRWD",),), max_tool_calls=10),
    C("d03", "我的组合健康吗", "dead_end", "unknown", "",
      ("risk_view",), (), (("54.7%", "54.7"),), max_tool_calls=8),
    C("d04", "最近市场怎么了", "dead_end", "unknown", "",
      ("macro_view",), (), (("18.40", "18.4"),), max_tool_calls=6),
    C("d05", "有没有什么我该知道但还不知道的事", "dead_end", "unknown", "",
      (), (), (("CRWD", "SMCI", "财报", "风险", "内部人"),), max_tool_calls=12),
    C("d06", "你能帮我做什么", "dead_end", "unknown", "",
      (), ("portfolio_view",), (("持仓", "财报", "风险", "宏观", "异动"),),
      max_tool_calls=2,
      note="元问题：应当直接说明能力范围，不该为此调用任何数据工具"),
    C("d07", "现在是加仓的好时候吗", "dead_end", "unknown", "",
      ("macro_view",), (), (("VIX", "18.40", "18.4", "宏观"),), max_tool_calls=10),

    # =======================================================================
    # 9. cost_trap — sounds complex, one card answers it (6)
    # =======================================================================
    C("t01", "我的持仓总市值是多少", "cost_trap", "portfolio_view", "",
      ("portfolio_view",),
      ("risk_view", "earnings_view", "insider_view", "eight_k_view"),
      (("184,320.55", "184320.55"),), expected_path="single_hop", max_tool_calls=2),
    C("t02", "我的组合行业暴露", "cost_trap", "risk_view", "",
      ("risk_view",), ("portfolio_view", "earnings_view"),
      (("38.1%", "38.1"), ("36.5%", "36.5")),
      expected_path="single_hop", max_tool_calls=2),
    C("t03", "未来两周谁要发财报", "cost_trap", "earnings_calendar", "",
      ("earnings_calendar",), ("portfolio_view", "earnings_view"),
      (("CRWD",), ("AVGO",)), expected_path="single_hop", max_tool_calls=2),
    C("t04", "CRWD 最近的内部人交易明细", "cost_trap", "insider_view", "CRWD",
      ("insider_view",), ("portfolio_view", "eight_k_view"),
      (("7.96M", "7,96"), ("10.97M", "10,97")),
      expected_path="single_hop", max_tool_calls=2),
    C("t05", "我这个月的盈亏", "cost_trap", "pnl_period", "",
      ("pnl_period",), ("portfolio_view", "risk_view"),
      (("6,120.70", "6120.70"),), expected_path="single_hop", max_tool_calls=2),
    C("t06", "NVDA 的机构持股情况", "cost_trap", "holders_view", "NVDA",
      ("holders",), ("portfolio_view", "institutional_13f"),
      (("Vanguard",), ("8.94%", "8.94")),
      expected_path="single_hop", max_tool_calls=2),

    # =======================================================================
    # 10. checker_stress — answer shapes that broke the checks, not the model (6)
    # =======================================================================
    #
    # Every case here was a production incident. Nine rounds of live testing
    # produced seven attribution false positives, and not one of them was
    # visible to this suite: the cases passed, the ⚠️ banner nobody scored sat
    # on top of a correct answer, and a human found it by reading Telegram.
    #
    # So these do not test the model. They aim questions at the fixture regions
    # whose *shape* defeated the checker — a column of dates, a threshold, a
    # window length, a percentage the answer computes — and the attribution axis
    # scores whatever the model writes about them. A warning raised here on an
    # otherwise-correct answer is a false positive by construction.
    #
    # Phrasing cannot be forced, which is the point: the shapes recur, the
    # wording will not, and a fixed list of remembered sentences would only
    # re-detect the seven bugs already fixed.
    C("k01", "接下来两周谁要发财报，分别是什么时候",
      "checker_stress", "earnings_calendar", "",
      ("earnings_calendar",), (),
      (("09-30", "9-30", "9 月 30"), ("10-21", "10 月 21"), ("MSFT",), ("AMD",)),
      expected_path="single_hop", max_tool_calls=3,
      note="日期列：每个日子都会被数字提取器读成负数，落进上一行标的的窗口"),
    C("k02", "NVDA 和 AMD 上次财报各自超预期多少，谁的幅度更大",
      "checker_stress", "unknown", "",
      ("earnings_view",), (),
      (("5.6%", "5.6"), ("2.9%", "2.9"), ("AMD",)),
      max_tool_calls=4,
      note="「实际 vs 预期（+X%）」：X 是模型当场算的比值，没有卡片拥有它"),
    C("k03", "MSFT 离 52 周高点还有多远",
      "checker_stress", "moneyflow_view", "MSFT",
      ("moneyflow_view",), (),
      (("0.14",), ("58.4",), ("3.10%", "3.1%")),
      expected_path="single_hop", max_tool_calls=3,
      note="「52 周高点」「200 日均线」：52 和 200 是窗口长度，不是谁的量"),
    C("k04", "CRWD 占仓多少，超没超过集中度阈值",
      "checker_stress", "risk_view", "CRWD",
      ("risk_view",), (),
      (("22.4%", "22.4"), ("20%", "20.0%")),
      expected_path="single_hop", max_tool_calls=3,
      note="阈值数字不归任何主体所有，却紧挨着 CRWD 出现"),
    C("k05", "把我持仓里亏损的几只按浮亏排个序，用整数百分比说",
      "checker_stress", "unknown", "",
      ("portfolio_view",), (),
      (("SMCI",), ("TSLA",), ("AMD",)),
      max_tool_calls=4,
      note="要求整数百分比，逼出「-21.5% 写成 21%」这类舍入表述"),
    C("k06", "我组合里半导体和软件各占多少",
      "checker_stress", "risk_view", "",
      ("risk_view",), (),
      (("38.1%", "38.1"), ("36.5%", "36.5")),
      expected_path="single_hop", max_tool_calls=3,
      note="行业标签（BROAD/半导体）不是 ticker，却会被大写词模式收成实体。"
           "另外它意外成了一条工具选择的硬用例：权重恰好凑得出答案"
           "（NVDA 18.2 + AMD 11.3 + SMCI 8.6 = 38.1），但「哪只属于哪个行业」"
           "不在任何工具返回里，只有 risk_view 说得出。模型用 portfolio_view "
           "自己算，数字对、来源不可溯 —— grounding 独立地也拒了它。"),
)


#: What the real classifier would also emit for these queries beyond intent and
#: ticker. Kept as a separate table rather than threaded through 83 literals:
#: the baseline dispatches on these fields, so omitting them would handicap it
#: with a harness artefact instead of measuring its actual design.
CLASSIFIER_EXTRAS: dict[str, dict[str, Any]] = {
    "s10": {"release_type": "cpi"},
    "s11": {"release_type": "fomc"},
    "s13": {"manager": "buffett"},
    "s15": {"etf": "ARKK"},
    "c01": {"period": "week"},
    "c02": {"period": "month"},
    "c08": {"period": "week"},
    "p05": {"release_type": "cpi"},
    "p08": {"manager": "buffett", "etf": "ARKK"},
    "h03": {"manager": "burry"},
    "h04": {"etf": "ARKQ"},
    "t05": {"period": "month"},
}

CASES = tuple(
    replace(case, extra=CLASSIFIER_EXTRAS[case.id]) if case.id in CLASSIFIER_EXTRAS
    else case
    for case in CASES
)


CATEGORIES: tuple[str, ...] = (
    "single_lookup", "multi_hop", "ranking", "causal",
    "compound", "recovery", "honesty", "dead_end", "cost_trap",
    "checker_stress",
)


def by_category() -> dict[str, list[EvalCase]]:
    grouped: dict[str, list[EvalCase]] = {c: [] for c in CATEGORIES}
    for case in CASES:
        grouped.setdefault(case.category, []).append(case)
    return grouped
