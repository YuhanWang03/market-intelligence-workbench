"""Non-overlapping, model-facing capability catalog for Agent V2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class CapabilitySpec:
    name: str
    pack: str
    description: str
    input_schema: dict[str, Any]
    mutating: bool = False
    long_running: bool = False
    evidence_required: bool = True
    #: Domain-specific instructions the synthesizer appends when this
    #: capability contributed evidence.  Keeps domain prose out of the core.
    answer_guidance: str = ""


class CapabilityCatalog:
    def __init__(self, specs: Iterable[CapabilitySpec] = ()) -> None:
        self._specs = {spec.name: spec for spec in specs}

    def get(self, name: str) -> CapabilitySpec | None:
        return self._specs.get(name)

    def names(self, packs: Iterable[str] | None = None) -> list[str]:
        allowed = set(packs or ())
        return [name for name, spec in self._specs.items() if not allowed or spec.pack in allowed]

    def specs(self, packs: Iterable[str] | None = None) -> list[CapabilitySpec]:
        return [self._specs[name] for name in self.names(packs)]

    def schemas(self, packs: Iterable[str] | None = None) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": spec.name.replace(".", "__"),
                    "description": spec.description,
                    "parameters": spec.input_schema,
                },
            }
            for spec in self.specs(packs)
        ]


_EMPTY = {"type": "object", "properties": {}, "additionalProperties": False}
_TICKER = {"type": "string", "pattern": "^[A-Za-z][A-Za-z0-9.-]{0,7}$"}
_TICKERS = {"type": "array", "items": _TICKER}
_UNIVERSE = {
    "type": "string",
    "enum": [
        "custom",
        "tech30",
        "sp500",
        "nasdaq100",
        "dow30",
        "holdings",
        "watchlist",
        "holdings_watchlist",
    ],
}
_DATA_SOURCE = {"type": "string", "enum": ["yfinance", "fd"]}


_RESEARCH_GUIDANCE = "stock_research：围绕公司的核心投资矛盾组织答案，不逐项报分。ROIC、ROE、利润率等异常高于 100% 的比率必须提示其依赖数据与计算口径，不能当作无条件质量结论。问风险时分三层写：公司自身（经营、财务、申报里的风险因素）、行业（竞争、周期、供应链）、宏观（利率、政策、汇率），每层至少一条有证据的，没有证据的那层明说。"
_LAB_GUIDANCE = "lab：先交代实验区间、样本和假设（持有期、成本、每笔金额或起始资金），再给结果；结尾必须有一句：这是历史回测（或历史事件统计），不代表未来收益。"
_SWEEP_GUIDANCE = "lab_sweep：逐个参数组合各写一行：参数、总收益、最大回撤、夏普（有就写）、交易笔数，一个都不能漏；然后点名最好的组合并说差距有多大；最后提醒参数扫描有过拟合风险。"
_EVENT_STUDY_GUIDANCE = "lab_event_study：写明事件数量、事件窗口（财报日前后几天）、平均异常收益或超额收益，以及样本量或置信度的限制；几只股票分别交代。"
_COMPARE_GUIDANCE = "stock_compare：先用“同口径对照”那几条证据（每条含所有候选的同一个指标）把增长、估值、盈利兑现摆在一起比，每只都点名、每个指标各引用对应那条对照；缺失的一方明说缺失，不能拿别的指标顶替。然后才是各自的证据。"
_MACRO_GUIDANCE = "macro：每个数字写明它的时间点（数据日期或“截至查询时”），发布类数据写发布日期和对应月份；没有对比值（预期、前值）时明说。"
_THIRTEEN_F_GUIDANCE = "thirteen_f：先写报告期（季度末）和披露的滞后（13F 在季度结束后 45 天内提交，持仓可能已经变化），再说主要持仓和本期增减仓；没有增减仓数据时明说。"


def _object(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


def default_catalog() -> CapabilityCatalog:
    specs = (
        CapabilitySpec("account.portfolio", "account", "Current positions and weights in the user's account.", _EMPTY),
        CapabilitySpec("account.performance", "account", "Account P&L for day, week, or month.", _object({"period": {"type": "string", "enum": ["day", "week", "month"]}}, ["period"])),
        CapabilitySpec("account.risk", "account", "Portfolio concentration, exposure, drawdown, and event risk.", _EMPTY),
        CapabilitySpec("account.earnings_schedule", "account", "Upcoming earnings across holdings and watchlist.", _object({"days": {"type": "integer", "minimum": 1, "maximum": 90}})),
        CapabilitySpec(
            "research.stock",
            "research",
            "Evidence-backed research for one stock and one focus area.",
            _object({"ticker": _TICKER, "focus": {"type": "string", "enum": ["overview", "fundamentals", "valuation", "earnings", "market", "ownership", "catalysts", "filings", "supply_chain", "risk", "full"]}}, ["ticker"]),
            long_running=True,
            answer_guidance=_RESEARCH_GUIDANCE,
        ),
        CapabilitySpec(
            "research.compare",
            "research",
            "Compare two to four stocks on identical research dimensions.",
            _object({"tickers": {"type": "array", "items": _TICKER, "minItems": 2, "maxItems": 4}, "dimensions": {"type": "array", "items": {"type": "string"}}}, ["tickers"]),
            long_running=True,
            answer_guidance=_RESEARCH_GUIDANCE + " " + _COMPARE_GUIDANCE,
        ),
        CapabilitySpec("research.changes", "research", "Compare the latest two stored research snapshots for one stock.", _object({"ticker": _TICKER}, ["ticker"])),
        CapabilitySpec(
            "market.performance",
            "research",
            "Recent stock-price performance across day, week, month, quarter and year, including volume and sector/broad-market benchmarks. Use for recent performance, returns, price trend, or whether a stock is outperforming.",
            _object({"ticker": _TICKER}, ["ticker"]),
            answer_guidance=(
                "recent_performance：先回答最新交易日、近 5 日和近 1 月的价格回报，再说明相对行业或大盘基准的强弱及成交量；不得用营收、毛利率或估值代替价格表现。"
                "若 metrics.is_intraday=true，必须写“截至查询时”或“盘中”，不能写“收盘”；当前累计成交量只能与完整日均量做进度参考，不得据此判断放量、缩量或上涨持续性。数据不足时明确缺少哪个时间窗口。"
                "每一句含数字的行情事实都必须紧跟对应的 [evidence_id]。"
            ),
        ),
        CapabilitySpec(
            "market.drawdown",
            "research",
            "Where in time a stock's decline happened: the return window that holds most of a loss, the peak-to-trough inside it, and the worst single trading days with dates. Use for a loss since purchase or a drawdown, never for today's move.",
            _object({"ticker": _TICKER, "loss_pct": {"type": "number"}, "window": {"type": "string", "enum": ["1m", "3m", "1y"]}, "top": {"type": "integer", "minimum": 1, "maximum": 5}}, ["ticker"]),
            answer_guidance=(
                "drawdown_timing：先说明跌幅落在哪个回报窗口、区间高点到低点的回撤，再按日期列出跌幅最大的交易日；每个日期和幅度紧跟对应的 [evidence_id]。"
                "只能把申报、盯盘记录或新闻与日期相同或相邻的下跌日关联，不得把当日归因推广到整个区间。"
            ),
        ),
        CapabilitySpec(
            "market.runup",
            "research",
            "Where in time a stock's rise happened: the return window that holds most of a gain, the low-to-high inside it, the sector ETF over the same stretch, and the best single trading days with dates. Use for a gain since purchase or a run-up, never for today's move.",
            _object({"ticker": _TICKER, "gain_pct": {"type": "number"}, "window": {"type": "string", "enum": ["1m", "3m", "1y"]}, "top": {"type": "integer", "minimum": 1, "maximum": 5}}, ["ticker"]),
            answer_guidance=(
                "runup_timing：先说明涨幅落在哪个回报窗口、区间低点到高点的涨幅和同期行业基准对比，再按日期列出涨幅最大的交易日；每个日期和幅度紧跟对应的 [evidence_id]。"
                "只能把申报、盯盘记录或新闻与日期相同或相邻的上涨日关联，不得把当日归因推广到整个区间。"
            ),
        ),
        CapabilitySpec(
            "filings.recent",
            "research",
            "Dated SEC filings (8-K by default) for one stock over a date range, from EDGAR. Use to find what the company disclosed around specific dates.",
            _object({"ticker": _TICKER, "since": {"type": "string"}, "until": {"type": "string"}, "forms": {"type": "array", "items": {"type": "string"}}}, ["ticker"]),
            answer_guidance="filings：只陈述申报的日期、表格类型和链接。同一运行里若有 evidence_scope=filing_event 的证据（申报阅读者从原文摘出的事件），申报已被读取，须引用那些事件，不得写“申报内容未读取”；没有这类证据时才说申报内容未读取、不得推测其影响。",
        ),
        CapabilitySpec(
            "filings.read_events",
            "research",
            "Read the SEC filings around a date and report dated events quoted from their text: earnings figures, guidance, executive changes, offerings, litigation. A bounded sub-agent; slower than listing filings. Use when the question is what the company disclosed around a specific date.",
            _object({"ticker": _TICKER, "since": {"type": "string"}, "until": {"type": "string"}, "around": {"type": "string"}, "max_filings": {"type": "integer", "minimum": 1, "maximum": 3}}, ["ticker"]),
            long_running=True,
            answer_guidance="filing_events：每个事件说明日期、表格和引文出处；引文之外的内容不得补充；读不到相关章节时如实说明。",
        ),
        CapabilitySpec(
            "market.attribute_move",
            "research",
            "Explain one day's move for one stock with verifiable sources: news (only with the user's web consent), the filings around that day read by the filing reader, and the monitor's anomaly memory. A bounded sub-agent; its conclusion is remembered for later questions. Use for a specific past day, not for today's move.",
            _object({"ticker": _TICKER, "date": {"type": "string"}}, ["ticker"]),
            long_running=True,
            answer_guidance=(
                "move_attribution：先说那天的涨跌幅、成交量和相对行业的表现，再分开说高置信度驱动和候选解释，每条带来源引文；"
                "没有高置信度驱动时明确写“尚未确认”，候选只能作为排查方向；不得把内部计数直接告诉用户。"
            ),
        ),
        CapabilitySpec(
            "market.anomaly_history",
            "research",
            "Past intraday anomalies the monitor recorded for one stock (date, flags, attribution notes) within a lookback window.",
            _object({"ticker": _TICKER, "lookback_days": {"type": "integer", "minimum": 1, "maximum": 730}, "query": {"type": "string"}}, ["ticker"]),
            answer_guidance="anomaly_history：盯盘记录只说明那一天触发了什么标志和当时的归因备注，日期必须原样给出。",
        ),
        CapabilitySpec(
            "market.explain_move",
            "research",
            "Explain today's (the latest session's) price move: the day's facts against the sector, a filing dated within three days read first, news with the user's web consent, the monitor's memory; confirmed drivers separated from candidates.",
            _object({"ticker": _TICKER}, ["ticker"]),
            long_running=True,
            answer_guidance=(
                "move_explanation：第一句回答是否上涨/下跌、日期、幅度和成交量。把已确认行情事实、高置信度直接驱动、普通候选解释分开。"
                "只有 metadata.claim_role=confirmed_driver 的证据才能写成已确认原因；candidate_driver 必须写成“可能相关”并说明中/低置信度。"
                "如果 confirmed_driver_count 为 0，用自然语言说“暂未找到可核实的同日催化剂，具体触发原因尚未确认”，不要输出“0 个驱动”之类的系统字段。"
                "不能把历史涨幅、机构持仓或时间不匹配的新闻写成当日直接原因。没有直接驱动时最多展示 1 条最相关的候选线索；metadata.citable=false 的线索不要引用。"
                "必须给出行业基准对比；若工具没有基准则说明缺失。若 metrics.is_intraday=true，必须标明盘中口径，且不得用当前累计成交量推断放量/缩量或持续性。"
                "每一句含数字的行情事实都必须紧跟对应的 [evidence_id]。"
            ),
        ),
        CapabilitySpec("institutional.manager_portfolio", "research", "Latest 13F portfolio for a named manager.", _object({"manager": {"type": "string"}}, ["manager"]), answer_guidance=_THIRTEEN_F_GUIDANCE),
        CapabilitySpec("etf.ark_activity", "research", "ARK ETF holdings and recent activity.", _object({"symbol": {"type": "string"}}, ["symbol"])),
        CapabilitySpec("macro.overview", "research", "Macro dashboard: VIX, DXY, WTI, gold, treasury yields and the most recent economic releases.", _EMPTY, answer_guidance=_MACRO_GUIDANCE),
        CapabilitySpec("macro.release", "research", "Latest value and date for a named macro release.", _object({"release_type": {"type": "string", "enum": ["cpi", "pce", "nfp", "gdp", "ppi", "claims", "fomc"]}}, ["release_type"]), answer_guidance=_MACRO_GUIDANCE),
        CapabilitySpec(
            "lab.screen",
            "lab",
            "Run a deterministic stock screen over a selected universe.",
            _object(
                {
                    "universe": _UNIVERSE,
                    "tickers": _TICKERS,
                    "data_source": _DATA_SOURCE,
                    "with_earnings": {"type": "boolean"},
                    "rules": {
                        "type": "array",
                        "maxItems": 24,
                        "items": {
                            "type": "object",
                            "properties": {
                                "field": {"type": "string"},
                                "op": {"type": "string", "enum": ["gte", "lte"]},
                                "value": {"type": "number"},
                            },
                            "required": ["field", "op", "value"],
                            "additionalProperties": False,
                        },
                    },
                    "market_cap_min": {"type": "number", "minimum": 0},
                    "market_cap_max": {"type": "number", "exclusiveMinimum": 0},
                    "revenue_growth_min": {"type": "number"},
                    "gross_margin_min": {"type": "number"},
                    "volatility_max": {"type": "number", "exclusiveMinimum": 0},
                }
            ),
            long_running=True,
        ),
        CapabilitySpec(
            "lab.backtest",
            "lab",
            "Backtest an existing strategy with explicit assumptions and costs.",
            _object(
                {
                    "strategy": {"type": "string", "enum": ["pead", "momentum", "insider", "committee"]},
                    "universe": _UNIVERSE,
                    "tickers": _TICKERS,
                    "data_source": _DATA_SOURCE,
                    "holding_days": {"type": "integer", "minimum": 1, "maximum": 252},
                    "capital": {"type": "number", "exclusiveMinimum": 0},
                    "per_trade": {"type": "number", "exclusiveMinimum": 0},
                    "cost_bps": {"type": "number", "minimum": 0, "maximum": 200},
                    "earnings_limit": {"type": "integer", "minimum": 1, "maximum": 20},
                    "history_days": {"type": "integer", "minimum": 60, "maximum": 3650},
                    "top_n": {"type": "integer", "minimum": 1, "maximum": 60},
                    "lookback_days": {"type": "integer", "minimum": 20, "maximum": 504},
                    "skip_days": {"type": "integer", "minimum": 0, "maximum": 120},
                    "near_high_pct": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
                    "window_days": {"type": "integer", "minimum": 1, "maximum": 180},
                    "min_insiders": {"type": "integer", "minimum": 1, "maximum": 20},
                    "min_value_usd": {"type": "number", "minimum": 0},
                    "min_consensus": {"type": "number", "minimum": -1, "maximum": 1},
                    "min_agreement": {"type": "number", "minimum": 0, "maximum": 1},
                    "personas": {"type": ["array", "null"], "items": {"type": "string"}, "maxItems": 20},
                    "lean": {"type": "boolean"},
                    "filing_lag_days": {"type": "integer", "minimum": 0, "maximum": 120},
                },
                ["strategy"],
            ),
            long_running=True,
            answer_guidance=_LAB_GUIDANCE,
        ),
        CapabilitySpec(
            "lab.sweep",
            "lab",
            "Run a bounded momentum parameter sweep.",
            _object(
                {
                    "universe": _UNIVERSE,
                    "tickers": _TICKERS,
                    "data_source": _DATA_SOURCE,
                    "history_days": {"type": "integer", "minimum": 60, "maximum": 3650},
                    "lookback_days": {"type": "integer", "minimum": 20, "maximum": 504},
                    "skip_days": {"type": "integer", "minimum": 0, "maximum": 120},
                    "capital": {"type": "number", "exclusiveMinimum": 0},
                    "cost_bps": {"type": "number", "minimum": 0, "maximum": 200},
                    "top_ns": {"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 6},
                    "holding_days_list": {"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 6},
                    "near_high_pcts": {"type": "array", "items": {"type": ["number", "null"]}, "minItems": 1, "maxItems": 4},
                }
            ),
            long_running=True,
            answer_guidance=_LAB_GUIDANCE + " " + _SWEEP_GUIDANCE,
        ),
        CapabilitySpec(
            "lab.event_study",
            "lab",
            "Measure abnormal returns around earnings events.",
            _object(
                {
                    "universe": _UNIVERSE,
                    "tickers": _TICKERS,
                    "data_source": _DATA_SOURCE,
                    "earnings_limit": {"type": "integer", "minimum": 1, "maximum": 20},
                    "n_bootstrap": {"type": "integer", "minimum": 100, "maximum": 10000},
                    "require_eps_surprise": {"type": "boolean"},
                    "dedupe": {"type": "boolean"},
                    "group_by": {"type": "string", "enum": ["surprise", "reaction", "source"]},
                }
            ),
            long_running=True,
            answer_guidance=_LAB_GUIDANCE + " " + _EVENT_STUDY_GUIDANCE,
        ),
        CapabilitySpec(
            "lab.committee",
            "lab",
            "Run the deterministic investor-persona committee.",
            _object(
                {
                    "source": {"type": "string", "enum": ["tickers", "holdings", "watchlist", "screening"]},
                    "tickers": _TICKERS,
                    "personas": {"type": ["array", "null"], "items": {"type": "string"}},
                    "as_of": {"type": ["string", "null"], "pattern": "^\\d{4}-\\d{2}-\\d{2}$"},
                    "top_n": {"type": ["integer", "null"], "minimum": 1, "maximum": 60},
                    "use_cache": {"type": "boolean"},
                    "max_weight": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
                    "lean": {"type": "boolean"},
                    "screening": {"type": ["object", "null"]},
                }
            ),
            long_running=True,
        ),
        CapabilitySpec("state.read", "account", "Read watchlist, alerts, or user settings.", _object({"section": {"type": "string", "enum": ["watchlist", "alerts", "settings"]}}, ["section"])),
        CapabilitySpec(
            "state.mutate",
            "command",
            "Apply a previously confirmed watchlist or alert mutation.",
            _object(
                {
                    "operation": {"type": "string", "enum": ["watchlist.add", "watchlist.remove", "alert.add", "alert.remove"]},
                    "payload": {"type": "object"},
                },
                ["operation", "payload"],
            ),
            mutating=True,
        ),
        CapabilitySpec("agent.investigate", "research", "A sub-agent with no fixed role: give it a task, a ticker and the tools it may use (search_news, read_page, list_filings, read_filing, recall_memory); it reports dated findings with quotes it located in what it read.", _object({"task": {"type": "string", "minLength": 4, "maxLength": 400}, "ticker": _TICKER, "tools": {"type": "array", "items": {"type": "string", "enum": ["search_news", "read_page", "list_filings", "read_filing", "recall_memory"]}, "maxItems": 5}, "recency_days": {"type": "integer", "minimum": 1, "maximum": 3650}}, ["task"]), long_running=True),
        CapabilitySpec("web.research", "web", "Search bounded external evidence when internal sources have a documented gap; with a model it reads pages and reports dated events with located quotes.", _object({"query": {"type": "string", "minLength": 1, "maxLength": 500}, "topic": {"type": "string"}, "ticker": _TICKER, "tickers": {**_TICKERS, "maxItems": 4}, "recency_days": {"type": "integer", "minimum": 1, "maximum": 3650}, "min_searches": {"type": "integer", "minimum": 1, "maximum": 3}}, ["query", "topic"]), long_running=True),
    )
    return CapabilityCatalog(specs)
