"""Intent: what a question asks for, as fixed fields instead of matched words.

The router and the planner decide from these fields.  Translating the
user's words into them is the model's job (:class:`IntentClassifier`, one
call per question); which tasks follow from a given intent is the planner's
(``planning.py``), and that part stays deterministic.

Without a model the fields come from a recorded fixture keyed by the exact
text (``eval/recorded_intents.json``: the labels for every question the
tests and the offline evals use), and for a text nobody labelled, from a
minimal default that says so in the plan's assumptions.

Every live decision (the intent, its source, the plan it produced) is
appended to ``data/agent_v2_intents.jsonl``; ``python -m
v2.agent_v2.eval.intent_report`` summarises it.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from v2.agent_v2.models import ExecutionPlan, NormalizedRequest

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PATH = _PROJECT_ROOT / "data" / "agent_v2_intents.jsonl"
_RECORDED_PATH = Path(__file__).resolve().parent / "eval" / "recorded_intents.json"

#: What kind of answer the question wants.
KINDS = ("research", "lookup", "command", "knowledge", "lab", "help")
#: The time the question is about.
SCOPES = ("today", "recent", "window", "since_purchase", "none")
#: Direction of the move the question is about.
DIRECTIONS = ("up", "down", "none")
#: The investigator's three jobs: how an event unfolded, what a filing says in its own words, whether a claim has a source.
INVESTIGATIONS = ("event_story", "filing_terms", "claim_source")
#: What the answer must contain; the planner's templates key on these.
WANTS = (
    "attribution",  # why a stock moved
    "news",  # recent news / events
    "filings",  # SEC filings, insider forms
    "earnings",  # results, EPS, earnings dates
    "valuation",
    "ownership",  # insiders, institutions
    "supply_chain",
    "catalysts",
    "risk",
    "compare",  # several stocks against each other
    "performance",  # returns, volume, price action
    "drawdown",  # a stretch of decline
    "runup",  # a stretch of gains
    "ranking",  # which holding is best / worst
    "portfolio",  # positions, weights, P/L
    "watchlist",
    "alerts",
    "settings",
    "macro",
    "macro_release",  # a specific data release (CPI, FOMC, ...)
    "guru",  # a manager's 13F
    "ark",
    "research_changes",  # what changed since the last research
    "briefing",  # what should I know today
    "positioning",  # add / trim exposure
    "market",  # the market as a whole: indexes, 大盘, 美股今天怎么样
    "full",  # a complete research report
    "overview",
)
#: The index ETFs that stand in for "the market" in a market-level question.
MARKET_TICKERS = ("SPY", "QQQ", "DIA")
RELEASES = ("cpi", "pce", "nfp", "gdp", "ppi", "claims", "fomc")
MANAGERS = ("buffett", "burry", "ackman", "einhorn", "renaissance", "citadel", "coatue", "twosigma", "deshaw", "ark")
LABS = ("backtest", "sweep", "event_study", "screen", "committee")
STRATEGIES = ("momentum", "insider", "pead", "committee")
OPERATIONS = ("watchlist.add", "watchlist.remove", "alert.add", "alert.remove")
#: Research focuses a want maps to (``research.stock`` ``focus`` argument).
FOCUS_OF_WANT = {"valuation": "valuation", "filings": "filings", "earnings": "earnings", "ownership": "ownership", "supply_chain": "supply_chain", "catalysts": "catalysts", "risk": "risk", "full": "full", "overview": "overview", "performance": "market"}

_SYSTEM = """你是投研助手的意图分类器，只输出 JSON，不回答问题。不要输出任何文字或分析过程，直接调用工具给出结果。
把用户的一句话归成固定字段，枚举值只能从给定列表里选，拿不准就选最接近的并降低 confidence：
kind：research（需要研究、分析、比较、判断值不值得买、解释一段时间的涨跌）| lookup（查一个事实：行情、成交量、持仓、财报日期、列表、宏观数据、某人的持仓）| command（改用户状态：加关注、删关注、设提醒、取消提醒）| knowledge（概念解释、术语区别，不涉及具体股票或账户）| lab（回测、参数扫描、事件研究、筛选、委员会投票）| help（问助手能做什么）
scope：today（今天、盘中、昨天）| recent（最近、这周/本周/这礼拜、这个月，没有明确起点；"这周谁涨得最好"是 recent，不是 none）| window（明确区间：今年、一年、从高点/低点以来）| since_purchase（买入以来、建仓以来；口语的"我买的 X 怎么亏成这样/赔了这么多/套住了"也是 since_purchase，direction=down，wants 含 drawdown 和 attribution，portfolio_scope=true）| none
direction：up | down | none（问题针对上涨还是下跌；"跌了这么多"是 down，"涨了多少"是 up）
wants：从列表里选 0 到 4 个，按重要性排序：%s
  说明：attribution=问原因；drawdown/runup=问一段时间的跌幅/涨幅（配合 scope=window/since_purchase）；performance=行情、成交量、走势；ranking=在持仓/关注列表里比出最好最差；briefing=今天/最近有什么值得注意的；positioning=该不该加仓减仓；market=问大盘、指数、美股整体行情（"今天美股行情如何"、"大盘怎么样"，不指向个股或账户）；overview=泛泛的"怎么样"。
tickers：股票代码（大写），公司中文名或英文名转成代码；没有就空数组。常见中文名：英特尔 INTC、英伟达 NVDA、美光 MU、超威/AMD AMD、高通 QCOM、博通 AVGO、台积电 TSM、苹果 AAPL、微软 MSFT、谷歌 GOOGL、亚马逊 AMZN、特斯拉 TSLA、甲骨文 ORCL、安谋/ARM ARM、闪迪 SNDK、Meta META、奈飞 NFLX、Palantir PLTR。
portfolio_scope：是否指向用户自己的持仓/账户（true/false）；watchlist_scope：是否指向用户的关注列表。
command：kind=command 时给 {"operation":"watchlist.add|watchlist.remove|alert.add|alert.remove","ticker":"...","direction":"above|below","price":数字,"alert_id":整数}，缺的字段省略；否则 null。
release：问具体宏观数据时给 %s 之一，否则空字符串。managers：问到的基金经理，从 %s 里选；ark_etfs：问到的 ARK ETF 代码（如 ARKK）。
window：drawdown/runup 有明确窗口时给 "1m"（一个月）或 "1y"（今年、一年、半年），否则空。
focus：kind=research 时想看的研究维度，从 valuation、earnings、filings、ownership、supply_chain、catalysts、risk、market、full、overview 里选 0 到 2 个。
lab：kind=lab 时给 %s 之一；strategy：回测策略 %s 之一；lab_scale："deep"（全市场、标普全部、十年、完整报告）或空。
periods：问账户盈亏时的口径 day/week/month 数组；each：是否要对每只持仓分别回答（"各自的财报日期"）。
rank：ranking 时 "high"（最好、涨最多）或 "low"（最差、跌最多），否则空。
confidence：0 到 1。
recent_turns：同一会话里之前的问答（问题、回答摘要、涉及的股票），按它补全这句里省略的股票、时间段和对象（"那 SNDK 呢"接着上一轮的比较；"换成一年的口径"接着上一轮的问题）。
refers_back：这句是针对上一条回答本身的追问——展开某一点（"第二点展开讲"）、追问理由（"为什么这么说"）、换个说法或口径重述、问上一条里提到的某个数字——时为 true，此时 wants 留空、tickers 填上一轮的股票；问新的事实（新的时间段、新的股票、新的数据）时为 false。
investigation：需要调查员逐条读原文才能回答的题给三类之一，否则空字符串：event_story（某件事的来龙去脉、前因后果、时间线："英特尔被政府入股那件事是怎么回事"）| filing_terms（某份申报里的具体条款、原文怎么写："特斯拉 10-K 里对 FSD 的风险具体怎么写"）| claim_source（某个说法有没有出处、是谁说的："有人说英伟达要把 H20 收入分 15%% 给政府，有出处吗"）；此时 kind=research，tickers 填涉及的股票。普通的"最近有什么新闻""为什么涨跌"不是调查题。
clarification：confidence 低于 0.6、或问题缺了非问不可的信息（哪只股票、什么时间段、加关注还是设提醒、目标价多少）时，给一句简短的反问，问清那一个缺口；其它情况留空字符串。反问要具体（"是想看 TSLA 的行情、新闻还是研究？"），不要泛泛地问"能否详细说明"。
通过 classify 工具返回；无法调用工具时只输出一个 JSON 对象，字段齐全。""" % ("、".join(WANTS), "/".join(RELEASES), "/".join(MANAGERS), "/".join(LABS), "/".join(STRATEGIES))


@dataclass
class Intent:
    kind: str = "lookup"
    scope: str = "none"
    direction: str = "none"
    wants: tuple[str, ...] = ()
    tickers: tuple[str, ...] = ()
    portfolio_scope: bool = False
    watchlist_scope: bool = False
    command: dict[str, Any] | None = None
    release: str = ""
    managers: tuple[str, ...] = ()
    ark_etfs: tuple[str, ...] = ()
    window: str = ""
    focus: tuple[str, ...] = ()
    lab: str = ""
    strategy: str = ""
    lab_scale: str = ""
    periods: tuple[str, ...] = ()
    each: bool = False
    rank: str = ""
    confidence: float | None = None
    #: ``model``, ``recorded`` (the eval fixture) or ``default`` (nobody classified it).
    source: str = "model"
    note: str = ""
    #: The one question the classifier would ask before answering, when the wording leaves a gap it cannot fill.
    clarification: str = ""
    #: The question is about the previous answer itself ("第二点展开讲", "为什么这么说"), not a new lookup.
    refers_back: bool = False
    #: One of ``INVESTIGATIONS`` when the question needs the investigator (read the sources, quote them), else "".
    investigation: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("wants", "tickers", "managers", "ark_etfs", "focus", "periods"):
            data[key] = list(getattr(self, key))
        return data

    def wants_any(self, *names: str) -> bool:
        return any(name in self.wants for name in names)


def _strings(value: Any, allowed: tuple[str, ...] | None, *, upper: bool = False, limit: int = 8) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        value = [value] if value else []
    cleaned = []
    for item in value:
        text = str(item or "").strip()
        text = text.upper() if upper else text.lower()
        if text and (allowed is None or text in allowed) and text not in cleaned:
            cleaned.append(text)
    return tuple(cleaned[:limit])


def parse_intent(raw: Any, *, source: str = "model") -> Intent:
    """Validate a classifier reply against the fixed vocabulary; unknown values are dropped, not guessed."""

    if not isinstance(raw, dict):
        raise ValueError("intent must be an object")
    kind = str(raw.get("kind") or "").strip().lower()
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}")
    scope = str(raw.get("scope") or "none").strip().lower()
    direction = str(raw.get("direction") or "none").strip().lower()
    confidence = raw.get("confidence")
    try:
        confidence = None if confidence is None else max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence = None
    command = None
    raw_command = raw.get("command")
    if isinstance(raw_command, dict) and str(raw_command.get("operation") or "") in OPERATIONS:
        command = {"operation": str(raw_command["operation"])}
        ticker = str(raw_command.get("ticker") or "").strip().upper()
        if re.fullmatch(r"[A-Z][A-Z0-9.-]{0,7}", ticker):
            command["ticker"] = ticker
        if str(raw_command.get("direction") or "").lower() in {"above", "below"}:
            command["direction"] = str(raw_command["direction"]).lower()
        for key, caster in (("price", float), ("alert_id", int)):
            try:
                if raw_command.get(key) is not None:
                    command[key] = caster(raw_command[key])
            except (TypeError, ValueError):
                pass
    tickers = tuple(value for value in _strings(raw.get("tickers"), None, upper=True) if re.fullmatch(r"[A-Z][A-Z0-9.-]{0,7}", value))
    return Intent(
        kind=kind,
        scope=scope if scope in SCOPES else "none",
        direction=direction if direction in DIRECTIONS else "none",
        wants=_strings(raw.get("wants"), WANTS, limit=6),
        tickers=tickers,
        portfolio_scope=bool(raw.get("portfolio_scope")),
        watchlist_scope=bool(raw.get("watchlist_scope")),
        command=command,
        release=str(raw.get("release") or "").lower() if str(raw.get("release") or "").lower() in RELEASES else "",
        managers=_strings(raw.get("managers"), MANAGERS, limit=2),
        ark_etfs=tuple(value for value in _strings(raw.get("ark_etfs"), None, upper=True, limit=2) if re.fullmatch(r"ARK[A-Z]", value)),
        window=str(raw.get("window") or "") if str(raw.get("window") or "") in {"1m", "1y"} else "",
        focus=_strings(raw.get("focus"), tuple(dict.fromkeys(FOCUS_OF_WANT.values())), limit=2),
        lab=str(raw.get("lab") or "").lower() if str(raw.get("lab") or "").lower() in LABS else "",
        strategy=str(raw.get("strategy") or "").lower() if str(raw.get("strategy") or "").lower() in STRATEGIES else "",
        lab_scale="deep" if str(raw.get("lab_scale") or "").lower() == "deep" else "",
        periods=_strings(raw.get("periods"), ("day", "week", "month"), limit=3),
        each=bool(raw.get("each")),
        rank=str(raw.get("rank") or "").lower() if str(raw.get("rank") or "").lower() in {"high", "low"} else "",
        confidence=confidence,
        source=source,
        note=str(raw.get("note") or "")[:120],
        clarification=" ".join(str(raw.get("clarification") or "").split())[:160],
        refers_back=bool(raw.get("refers_back")),
        investigation=str(raw.get("investigation") or "").lower() if str(raw.get("investigation") or "").lower() in INVESTIGATIONS else "",
    )


def intent_from_dict(data: dict[str, Any], *, source: str) -> Intent:
    return parse_intent(data, source=source)


# -- the model ----------------------------------------------------------------

INTENT_TOOL = {
    "type": "function",
    "function": {
        "name": "classify",
        "description": "把用户的问题归成固定字段。",
        "parameters": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": list(KINDS)},
                "scope": {"type": "string", "enum": list(SCOPES)},
                "direction": {"type": "string", "enum": list(DIRECTIONS)},
                "wants": {"type": "array", "items": {"type": "string", "enum": list(WANTS)}, "maxItems": 4},
                "tickers": {"type": "array", "items": {"type": "string"}},
                "portfolio_scope": {"type": "boolean"},
                "watchlist_scope": {"type": "boolean"},
                "command": {"type": ["object", "null"], "properties": {"operation": {"type": "string", "enum": list(OPERATIONS)}, "ticker": {"type": "string"}, "direction": {"type": "string", "enum": ["above", "below"]}, "price": {"type": "number"}, "alert_id": {"type": "integer"}}},
                "release": {"type": "string"},
                "managers": {"type": "array", "items": {"type": "string", "enum": list(MANAGERS)}},
                "ark_etfs": {"type": "array", "items": {"type": "string"}},
                "window": {"type": "string", "enum": ["", "1m", "1y"]},
                "focus": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
                "lab": {"type": "string"},
                "strategy": {"type": "string"},
                "lab_scale": {"type": "string"},
                "periods": {"type": "array", "items": {"type": "string", "enum": ["day", "week", "month"]}},
                "each": {"type": "boolean"},
                "rank": {"type": "string", "enum": ["", "high", "low"]},
                "confidence": {"type": "number"},
                "clarification": {"type": "string"},
                "refers_back": {"type": "boolean"},
                "investigation": {"type": "string", "enum": ["", *INVESTIGATIONS]},
            },
            "required": ["kind", "scope", "direction", "wants", "tickers", "portfolio_scope", "confidence"],
        },
    },
}


class IntentClassifier:
    """One model call that turns the question into an :class:`Intent`; ``None`` when the model fails or answers off-vocabulary."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def classify(self, request: NormalizedRequest) -> Intent | None:
        if self.llm is None:
            return None
        from v2.agent_v2.agents.base import structured_call
        from v2.usage_context import usage_source

        payload = {"text": request.text, "entities_detected": list(request.entities)}
        recent = request.metadata.get("recent_turns") if isinstance(request.metadata, dict) else None
        if recent:
            payload["recent_turns"] = recent
        try:
            with usage_source("agent_v2.intent"):
                raw = structured_call(self.llm, _SYSTEM, payload, INTENT_TOOL)
            return parse_intent(raw)
        except Exception as exc:  # noqa: BLE001 — a failed classification falls back to the default intent
            logger.warning("intent classifier failed: %s: %s", type(exc).__name__, exc)
            return None


# -- recorded labels and the default -----------------------------------------


class RecordedIntents:
    """Labels keyed by exact text: the offline stand-in for the model (tests, evals, no API key)."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or _RECORDED_PATH
        self._rows: dict[str, dict[str, Any]] | None = None

    def _load(self) -> dict[str, dict[str, Any]]:
        if self._rows is None:
            try:
                self._rows = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self._rows = {}
        return self._rows

    def get(self, text: str) -> Intent | None:
        row = self._load().get(_key(text))
        if not isinstance(row, dict):
            return None
        try:
            return intent_from_dict(row.get("intent") or row, source="recorded")
        except ValueError:
            return None


def _key(text: str) -> str:
    return " ".join((text or "").split())


_RECORDED = RecordedIntents()


def default_intent(request: NormalizedRequest) -> Intent:
    """When nobody classified the text: a lookup of the tickers it names, or a research overview of one."""

    tickers = tuple(request.entities)
    return Intent(kind="research" if tickers else "lookup", wants=("overview",) if tickers else (), tickers=tickers, source="default", note="未分类，按默认处理")


def resolve_intent(request: NormalizedRequest, *, classifier: IntentClassifier | None = None, recorded: RecordedIntents | None = None) -> Intent:
    """The model when there is one, the recorded label otherwise, the default last."""

    if classifier is not None:
        intent = classifier.classify(request)
        if intent is not None:
            return intent
    recorded_intent = (recorded or _RECORDED).get(request.text)
    if recorded_intent is not None:
        return recorded_intent
    return default_intent(request)


# -- decision ledger and report -------------------------------------------------


def ledger_path() -> Path:
    return Path(os.environ.get("AGENT_V2_INTENT_LEDGER") or _DEFAULT_PATH)


def decision_row(request: NormalizedRequest, intent: Intent, plan: ExecutionPlan, *, run_id: str, elapsed_ms: int = 0, channel: str = "") -> dict[str, Any]:
    return {
        "at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "run_id": run_id,
        "channel": channel,
        "text": (request.original_text or request.text or "")[:200],
        "intent": intent.to_dict(),
        "route": plan.route.value,
        "capabilities": [task.capability for task in plan.tasks],
        "budget": plan.budget.value,
        "elapsed_ms": elapsed_ms,
    }


def record_decision(row: dict[str, Any], path: Path | None = None) -> bool:
    target = path or ledger_path()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("intent ledger not written (%s): %s", target, exc)
        return False
    return True


def read_decisions(path: Path | None = None, *, since_days: int | None = None) -> list[dict[str, Any]]:
    target = path or ledger_path()
    if not target.exists():
        return []
    cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=since_days)).isoformat() if since_days else ""
    rows: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("intent") and (not cutoff or str(row.get("at") or "") >= cutoff):
                rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Where the intents came from, what kinds and wants were seen, and the questions the model was unsure about."""

    from collections import Counter

    sources = Counter(str((row.get("intent") or {}).get("source") or "?") for row in rows)
    kinds = Counter(str((row.get("intent") or {}).get("kind") or "?") for row in rows)
    wants: Counter = Counter()
    for row in rows:
        for want in (row.get("intent") or {}).get("wants") or []:
            wants[want] += 1
    confidences = [float(c) for c in ((row.get("intent") or {}).get("confidence") for row in rows) if isinstance(c, (int, float))]
    unsure = sorted((row for row in rows if isinstance((row.get("intent") or {}).get("confidence"), (int, float)) and (row["intent"]["confidence"] or 0) < 0.7), key=lambda row: row["intent"]["confidence"])
    unclassified = [row for row in rows if (row.get("intent") or {}).get("source") == "default"]
    return {
        "rows": len(rows),
        "sources": dict(sources),
        "kinds": dict(kinds),
        "wants": dict(wants.most_common()),
        "confidence_avg": round(sum(confidences) / len(confidences), 3) if confidences else None,
        "elapsed_ms_avg": round(sum(int(row.get("elapsed_ms") or 0) for row in rows) / len(rows)) if rows else 0,
        "unsure": [{"text": row.get("text"), "confidence": row["intent"]["confidence"], "intent": _short(row["intent"]), "capabilities": row.get("capabilities")} for row in unsure[:20]],
        "unclassified": [{"text": row.get("text"), "capabilities": row.get("capabilities")} for row in unclassified[:20]],
    }


def _short(intent: dict[str, Any] | None) -> str:
    if not intent:
        return "—"
    parts = [str(intent.get("kind")), str(intent.get("scope")), str(intent.get("direction")), "+".join(intent.get("wants") or []) or "∅", ",".join(intent.get("tickers") or []) or "∅"]
    if intent.get("portfolio_scope"):
        parts.append("账户")
    if intent.get("watchlist_scope"):
        parts.append("关注")
    return " / ".join(parts)


def render(summary: dict[str, Any], *, since_days: int | None) -> str:
    lines = [f"# 意图分类报告{f'（近 {since_days} 天）' if since_days else ''}", ""]
    if not summary["rows"]:
        lines.append("账本里还没有意图记录。")
        return "\n".join(lines)
    sources = "、".join(f"{key} {value}" for key, value in sorted(summary["sources"].items()))
    lines.append(f"问题 {summary['rows']} 个，来源：{sources}；平均置信度 {summary['confidence_avg'] if summary['confidence_avg'] is not None else '—'}，平均 {summary['elapsed_ms_avg']} ms。")
    lines.append("")
    lines.append("| kind | 问题数 |")
    lines.append("|---|---|")
    for key, value in sorted(summary["kinds"].items(), key=lambda item: -item[1]):
        lines.append(f"| {key} | {value} |")
    lines.append("")
    lines.append("| wants | 次数 |")
    lines.append("|---|---|")
    for key, value in summary["wants"].items():
        lines.append(f"| {key} | {value} |")
    if summary["unsure"]:
        lines.append("")
        lines.append("| 模型拿不准的问题（置信度 < 0.7） | 置信度 | 意图 | 计划 |")
        lines.append("|---|---|---|---|")
        for entry in summary["unsure"]:
            lines.append(f"| {entry['text']} | {entry['confidence']:.2f} | {entry['intent']} | {'、'.join(entry['capabilities'] or []) or '—'} |")
    if summary["unclassified"]:
        lines.append("")
        lines.append("| 未分类、按默认处理的问题 | 计划 |")
        lines.append("|---|---|")
        for entry in summary["unclassified"]:
            lines.append(f"| {entry['text']} | {'、'.join(entry['capabilities'] or []) or '—'} |")
    lines.append("")
    lines.append("意图列：kind / scope / direction / wants / tickers。拿不准的问题是改分类提示或补例子的依据。")
    return "\n".join(lines)


def classify_and_time(classifier: IntentClassifier | None, request: NormalizedRequest) -> tuple[Intent, int]:
    started = time.monotonic()
    intent = resolve_intent(request, classifier=classifier)
    return intent, int((time.monotonic() - started) * 1000)
