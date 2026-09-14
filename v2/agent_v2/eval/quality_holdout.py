"""Hold-out questions: run before a release, never read while iterating.

The development set (``quality_cases.QUALITY_CASES``) is looked at every
time a case fails and the fix is aimed at it; these cases are not.  Their
pass rate is the estimate of how the agent does on wording it has not been
tuned for.  Add cases in your own words; do not open this file to see why
one failed until the release is out (``quality show`` prints the answer).

``python -m v2.agent_v2.eval.quality run --set holdout --repeat 2``
"""

from __future__ import annotations

from v2.agent_v2.eval.quality_cases import QualityCase
from v2.agent_v2.models import RouteKind

HOLDOUT_CASES: tuple[QualityCase, ...] = (
    QualityCase("h_move_slang", "高通今天这是怎么了", criteria=("把高通识别为 QCOM 并说明当日涨跌幅和口径", "给出了行业基准对照", "对没有确认证据的解释加了限定"), forbidden=("把候选解释说成已确认的原因",), must_cite=("market_data",), expected_agents=("move_attributor",), set="holdout", tags=("attribution",)),
    QualityCase("h_move_english_mixed", "PLTR 今天 why so weak", criteria=("用中文回答", "给出了当日涨跌幅和基准对照"), must_cite=("market_data",), expected_agents=("move_attributor",), set="holdout", tags=("attribution", "english")),
    QualityCase("h_loss_since_buy", "我买的PLTR怎么亏成这样了", criteria=("先给出买入以来的浮亏", "给出了这段回撤的幅度和基准对照", "按日期说了主要下跌日有没有确认的原因"), forbidden=("用今天的涨跌解释买入以来的亏损",), must_cite=("market_data",), set="holdout", tags=("drawdown",)),
    QualityCase("h_news_week", "这一周半导体板块有什么大新闻", criteria=("列出了本周有日期有来源的事件", "围绕半导体板块或其主要公司，而不是单一无关公司"), forbidden=("把没有日期或来源的传闻当作事件",), set="holdout", tags=("news",)),
    QualityCase("h_earnings_next", "苹果什么时候出财报", criteria=("给出了 AAPL 下次财报日期或明说日历里没有",), set="holdout", tags=("earnings",)),
    QualityCase("h_compare_casual", "英伟达跟AMD比，现在哪个性价比高", criteria=("对两只用同口径的估值和增长数字比较", "给出有条件的结论或说明证据不足", "指出数据缺口"), forbidden=("给出无条件的买入建议",), expected_route=RouteKind.RESEARCH, set="holdout", tags=("compare",)),
    QualityCase("h_portfolio_worst_week", "这礼拜我的持仓谁拖后腿", criteria=("点名本周跌幅最大的那只并给出数字", "说明是周口径"), set="holdout", tags=("portfolio", "ranking")),
    QualityCase("h_watchlist_moves", "关注列表里今天有大动静的吗", criteria=("对关注列表里的每只给出当日涨跌幅", "点名波动最大的一两只", "说明盘中口径（如果是盘中）"), must_cite=("market_data",), set="holdout", tags=("watchlist", "performance")),
    QualityCase("h_market_open", "今天开盘美股怎么样", criteria=("给出了三大指数或对应 ETF 的当日涨跌幅和口径", "提到了当天的宏观事件或说明没有"), forbidden=("用用户账户的盈亏代替大盘行情",), set="holdout", tags=("market",)),
    QualityCase("h_fomc", "下次议息会议什么时候，市场预期怎么样", criteria=("给出了下次 FOMC 的日期", "给出了市场对利率路径的预期，或明说没有该数据"), forbidden=("编造尚未发布的决议",), allow_web=False, set="holdout", tags=("macro",)),
    QualityCase("h_knowledge_2", "什么是自由现金流收益率，怎么用", criteria=("给出定义和计算方式", "说明适用场景和局限", "声明这是通用知识"), expected_route=RouteKind.GENERAL_KNOWLEDGE, allow_web=False, set="holdout", tags=("knowledge",)),
    QualityCase("h_alert_below", "AMD跌破150的时候叫我", criteria=("说明将设置 AMD 跌破 150 的提醒并等待确认，没有直接执行",), expected_route=RouteKind.COMMAND, allow_web=False, set="holdout", tags=("command",)),
    QualityCase("h_followup_compare", "那换成一年的口径呢", criteria=("把问题理解为上一轮 AMD 近一个月表现的延续，改用一年口径回答", "给出了一年区间回报和基准对照"), preceding=("AMD这一个月涨了多少？",), must_cite=("market_data",), set="holdout", tags=("multi_turn",)),
)
