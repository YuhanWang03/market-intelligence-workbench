"""Version-neutral questions adapted from Agent V2 contract tests.

Rubrics describe observable answers, never a particular implementation's
sub-agent names. Live cases are read-only and do not require private accounts.
"""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EvaluationCase:
    id: str
    category: str
    question: str
    source_test: str
    criteria: tuple[str, ...]
    preceding: tuple[str, ...] = ()
    allow_web: bool = False

    def to_dict(self):
        return asdict(self)


def case(cid, category, question, source, *criteria, preceding=(), web=False):
    return EvaluationCase(cid, category, question, source, tuple(criteria), preceding, web)


CASES = (
    case('knowledge', 'knowledge', '什么是自由现金流？', 'test_router_separates_knowledge_research_lab_and_commands', '解释经营现金流减资本支出的常见口径', '不伪造实时数据或引用'),
    case('performance', 'market', 'AMD最近表现如何？', 'test_recent_stock_performance_uses_market_data_instead_of_fundamentals', '回答价格及区间回报，标明实际日期', '以行情回答，不用基本面评分替代'),
    case('volume', 'market', 'AMD今天成交量是不是低？', 'test_bare_stock_performance_defaults_to_recent_market_data', '提供成交量及可比基准', '盘中累计量不直接对比完整日成交量并下确定结论'),
    case('volatility', 'market', 'AMD最近波动率多高？', 'test_bare_stock_performance_defaults_to_recent_market_data', '给出波动率、样本窗口及年化口径', '缺数时说明，不估填'),
    case('fundamentals', 'financial', 'AMD基本面表现如何？', 'test_non_price_performance_language_stays_with_research', '使用财务证据与实际报告期间', '明确数据缺口，不把股价表现代替经营表现'),
    case('earnings_quality', 'financial', 'AMD最近的收益质量如何？', 'test_recent_earnings_quality_is_not_misrouted_as_price_performance', '讨论利润、现金流及可持续性，引用依据', '不把收益质量解释为股票涨幅'),
    case('comparison', 'comparison', 'MU和SNDK哪个更值得购买？', 'test_router_separates_knowledge_research_lab_and_commands', '覆盖两家公司并解释取舍及风险', '财务期间、单位和会计口径未对齐时不作严格同期优劣判断'),
    case('risk_comparison', 'comparison', '比较 NVDA 和 AMD 的风险', 'test_router_separates_knowledge_research_lab_and_commands', '分别指出两家公司风险及证据', '区分已知事实和推断'),
    case('move', 'attribution', 'AMD今天为什么涨？', 'test_move_explanation_cannot_be_overridden_by_the_llm_planner', '先核查是否上涨及实际交易日', '区分行情、候选驱动、已核实事件和推断', web=True),
    case('news', 'news', 'NVDA最近有哪些新闻？', 'test_a_news_question_plans_web_filings_and_memory_under_a_real_budget', '提供有原文支持且在窗口内的具体事件', '区分转载与独立来源，不能把申报目录当新闻全貌', web=True),
    case('news_no_web', 'permissions', 'NVDA最近有哪些新闻？', 'test_web_fallback_requires_runtime_and_per_request_opt_in', '不执行未授权网页搜索', '准确说明可访问证据与覆盖限制'),
    case('followup', 'conversation', '为什么涨跌？', 'test_short_term_session_resolves_a_follow_up_before_routing', '追问对象仍为 AMD', '解释实际行情而不预设上涨', preceding=('AMD最近表现如何？',), web=True),
    case('switch', 'conversation', '改查 AMD 的最近表现', 'test_short_term_session_resolves_a_follow_up_before_routing', '明确切换到 AMD，不复用 NVDA 数据', preceding=('NVDA最近表现如何？',)),
    case('restate', 'conversation', '将刚才的结果浓缩成一句话，不要重新取数。', 'test_follow_up_about_the_previous_answer_is_written_from_its_evidence_without_new_calls', '复用前轮证据，不调用新的业务工具', '保留股票、时间与核心数值', preceding=('AMD最近表现如何？',)),
    case('clarify', 'conversation', '最近收盘价是多少？', 'test_low_confidence_classification_asks_one_question_and_reads_the_next_message_as_the_answer', '无上下文时询问股票，不猜对象'),
    case('clarify_reply', 'conversation', 'AMD', 'test_low_confidence_classification_asks_one_question_and_reads_the_next_message_as_the_answer', '承接价格问题查询 AMD 收盘价', preceding=('最近收盘价是多少？',)),
    case('market_overview', 'market', '美股大盘最近一周表现如何？', 'test_market_level_questions_second_repair_on_progress_and_quality_show', '回答主要指数或明确标记的 ETF 代理', '提供实际日期窗口，不无谓要求用户提供单只股票'),
    case('filing', 'filing', '阅读 NVDA 最新年报中的供应链风险，引用原文。', 'test_the_investigator_can_list_an_annual_report_and_read_the_passages_around_a_keyword', '引用已读取年报正文及申报日期', '不将目录或摘要伪装成原文', web=True),
)

OFFLINE_ONLY = {
    'account': '持仓排名与买入以来归因需要同一份合成账户和成交记录；继续使用 portfolio_acceptance，不混入公开实时比较。',
    'mutations': '确认、取消及越权写入使用注入工具契约测试，测评不修改真实账户或关注列表。',
    'lab': '实验需相同数据集和资源预算，暂不与普通问答混合评分。',
}
