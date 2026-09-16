"""Validated semantic inputs and JSON-only graph state."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from v2.agent_v2.intent import Intent, WANTS
from v2.agent_v2.models import (
    AnswerMode, BudgetClass, EvidenceItem, ExecutionPlan, PlanTask,
    ResultStatus, RouteKind, ToolEnvelope,
)


class DateWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: str
    end: str
    basis: Literal["event", "filing", "publication"]

    @model_validator(mode="after")
    def ordered_dates(self):
        if any(date.fromisoformat(value).isoformat() != value for value in (self.start,self.end)):
            raise ValueError("Use YYYY-MM-DD dates")
        if date.fromisoformat(self.start) > date.fromisoformat(self.end):
            raise ValueError("Date window start exceeds end")
        return self


class PortfolioObjection(BaseModel):
    quote: str
    reason: str


class PortfolioAudit(BaseModel):
    violations: list[PortfolioObjection] = Field(default_factory=list,max_length=5)


class SemanticIntent(BaseModel):
    """Meaning of the request, never inferred with phrase matching."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["research", "lookup", "command", "knowledge", "lab", "help"]
    scope: Literal["today", "recent", "window", "since_purchase", "none"] = "none"
    direction: Literal["up", "down", "none"] = "none"
    wants: list[str] = Field(default_factory=list, max_length=6, json_schema_extra={"items":{"type":"string","enum":list(WANTS)}}, description="Requested objectives. performance includes stock price, close, returns and volume; overview is a broad company overview; valuation is valuation analysis, not a price quote.")
    tickers: list[str] = Field(default_factory=list, max_length=12)
    portfolio_scope: bool = Field(default=False, description="用户明确指向自己的账户或持仓；缺少股票代码不意味着查询账户")
    market_scope: Literal["", "us_broad"] = Field(default="",description="美国大盘/美股整体表现，无需用户指定股票；指定单个指数或ETF时留空")
    portfolio_metric: Literal["", "unrealized_percent", "unrealized_amount", "daily_return"] = Field(default="", description="持仓排名口径：累计浮亏比例、累计浮亏金额或股票单日涨跌幅；没有指定时持仓跌最多默认浮亏比例")
    portfolio_followup: Literal["", "explain_position", "rerank"] = Field(default="", description="解释上一轮选出的持仓亏损为explain_position；改按今天或金额比较整个持仓为rerank。新股票或独立市场问题留空")
    watchlist_scope: bool = False
    command: dict[str, Any] | None = None
    release: Literal["", "cpi", "pce", "nfp", "gdp", "ppi", "claims", "fomc"] = ""
    data_target: Literal["auto", "etf_holdings"] = "auto"
    holdings_top: int = Field(default=5,ge=1,le=10,description="Requested number of leading fund holdings; full issuer file remains available separately")
    analysis_scope: Literal["focused", "company"] = Field(default="focused", description="company表示对一家公司的开放式整体分析；不能因上一轮问新闻而继承为新闻查询。focused表示当前明确限定某个专题。")
    managers: list[str] = Field(default_factory=list)
    ark_etfs: list[str] = Field(default_factory=list)
    window: str = ""
    focus: list[str] = Field(default_factory=list)
    lab: Literal["", "backtest", "sweep", "event_study", "screen", "committee"] = ""
    strategy: str = ""
    lab_scale: Literal["", "deep"] = ""
    periods: list[str] = Field(default_factory=list)
    each: bool = False
    rank: Literal["", "high", "low"] = ""
    confidence: float = Field(default=1, ge=0, le=1)
    clarification: str = ""
    refers_back: bool = False
    follow_up_mode: Literal["expand", "restate"] = Field(default="expand", description="restate仅压缩或改写上一答已表达的事实，expand才允许展开旧证据")
    use_selected_record: bool = Field(default=False, description="仅询问或复述所选监控记录本身，不要求当前行情、外部核查或因果调查")
    investigation: Literal["", "event_story", "filing_terms", "claim_source"] = ""
    experiment_arguments: dict[str, Any] = Field(default_factory=dict)
    response_style: Literal["brief", "standard", "detailed"] = Field(default="standard", description="User's requested answer depth. brief for a concise answer or short restatement; detailed only when the user asks for detail.")
    date_window: DateWindow | None = Field(default=None, description="Explicit inclusive date bounds derived from the user's request and reference_date. event for news occurrence, filing for when a document was submitted. Do not confuse submission dates with event dates.")
    date_window_explicit: bool = Field(default=False,description="True only if user explicitly gave dates or a duration such as past 7 days/one month. Generic recent/latest alone is false.")
    filing_items: list[str] = Field(default_factory=list,max_length=4,description="Standard SEC Item identifiers requested by meaning, e.g. Item 1A for 10-K Risk Factors. Empty if the question does not specify a section or topic mapping.")
    filing_forms: list[str] = Field(default_factory=list,max_length=4,description="SEC forms explicitly requested, e.g. 10-K for an annual report. Empty if unspecified.")

    @field_validator("wants")
    @classmethod
    def known_wants(cls, values):
        if set(values) - set(WANTS):
            raise ValueError("unknown research objective")
        return values

    @field_validator("release", mode="before")
    @classmethod
    def canonical_release(cls, value):
        # Normalize a stable tool identifier, not user prose.
        return value.lower() if isinstance(value, str) else value

    @field_validator("tickers")
    @classmethod
    def identifiers(cls, values):
        # Stable exchange identifier syntax, not a natural-language parser.
        allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-")
        normalized = [value.upper() for value in values]
        if any(not value or len(value) > 8 or not value[0].isalpha() or set(value) - allowed for value in normalized):
            raise ValueError("invalid ticker identifier")
        return list(dict.fromkeys(normalized))

    def domain(self) -> Intent:
        data = self.model_dump(exclude={"market_scope", "portfolio_metric", "portfolio_followup", "date_window_explicit", "holdings_top", "analysis_scope", "follow_up_mode", "data_target", "use_selected_record", "experiment_arguments", "response_style", "date_window", "filing_items", "filing_forms"})
        for key in ("wants", "tickers", "focus", "periods", "managers", "ark_etfs"):
            data[key] = tuple(data[key])
        return Intent(**data, source="model")


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=100)
    capability: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    required: bool = True
    purpose: str = ""
    fan_out: dict[str, Any] | None = None


class PlannedTasks(BaseModel):
    tasks: list[TaskSpec] = Field(max_length=12)
    assumptions: list[str] = Field(default_factory=list)


class Objection(BaseModel):
    """One reviewer objection, anchored to the answer sentence and the evidence it contradicts.

    ``material``: acting on it would change a conclusion, or the answer states a
    data gap as a confirmed fact (or hides one).  ``minor``: wording precision
    only.  Only material objections trigger a redraft; minor ones are shown to
    the reader as review notes.
    """

    objection: str
    evidence_id: str
    severity: Literal["material", "minor"] = "material"
    claim: str = ""  # the answer sentence objected to, quoted


class Review(BaseModel):
    """Zero objections is a valid, expected review; the list is not to be padded."""

    objections: list[Objection] = Field(default_factory=list, max_length=6)


class ClaimViolation(BaseModel):
    index: int = Field(ge=0)
    quote: str = Field(min_length=1, description="Exact contiguous span from text that asserts the forbidden claim; never copy the rule itself")
    reason: str


class ClaimVerdict(BaseModel):
    violations: list[ClaimViolation] = Field(default_factory=list)


class GraphState(TypedDict, total=False):
    restatement_of: str
    inherited_partial: bool
    run_id: str
    session_id: str
    text: str
    allow_web: bool
    history: dict
    page_context: dict
    monitor_evidence: list[dict]
    record_answer: bool
    intent: dict
    plan: dict
    route: str
    status: str
    answer: str
    results: list[dict]
    evidence: list[dict]
    report: dict
    attempts: list[dict]
    repairs: int
    fallback: bool
    follow_up: bool
    error: str
    stop_reason: str
    approved: bool
    confirmation_expires: float
    objections: list[dict]
    #: Adversarial review record: ran, skipped (reason), objections, revised,
    #: revised_verified, error.  Present on every finished run.
    debate: dict
    trace: list[str]
    usage: list[dict]


def plain(value: Any) -> Any:
    """Enums/tuples become JSON values; no permissive repr serialization."""
    if hasattr(value, "__dataclass_fields__"):
        value = asdict(value)
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def plan_from(data: dict) -> ExecutionPlan:
    return ExecutionPlan(
        objective=data["objective"], route=RouteKind(data["route"]),
        tasks=tuple(PlanTask(**{**row, "depends_on": tuple(row.get("depends_on", []))}) for row in data.get("tasks", [])),
        budget=BudgetClass(data.get("budget", "standard")),
        answer_mode=AnswerMode(data.get("answer_mode", "research_grounded")),
        requires_confirmation=data.get("requires_confirmation", False),
        web_fallback_allowed=data.get("web_fallback_allowed", False),
        assumptions=tuple(data.get("assumptions", [])),
        direct_answer=data.get("direct_answer", ""), frame=data.get("frame", {}),
    )


def envelope_from(data: dict) -> ToolEnvelope:
    return ToolEnvelope(**{**data, "status": ResultStatus(data["status"]), "evidence": [EvidenceItem(**row) for row in data.get("evidence", [])]})


class RestatementVerdict(BaseModel):
    adds_facts: bool = Field(description="改写中是否新增或改变了原回答未表达的事实、数字、比较或结论")
