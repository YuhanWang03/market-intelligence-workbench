"""A run ledger for the sub-agents, and the report that reads it.

Every live run appends one JSON line per sub-agent run (the reader, the
attributor with its nested reader runs, the news checker): rounds, calls,
elapsed time, stop reason and yield (what it reported versus what the
verifier dropped).  Token use per sub-agent comes from the usage ledger,
where each provider call is attributed to ``agent_v2.<name>`` while the
sub-agent's loop runs.  ``python -m v2.agent_v2.eval.subagent_report``
aggregates both.

Cost is measured in tokens, not money: the provider's price changes with the
time of day, so the report folds uncached input, cached input and output
into one price-invariant *standard token equivalent* (see ``TOKEN_WEIGHTS``).
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from v2.agent_v2.models import AgentResult, sub_agent_summaries

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PATH = _PROJECT_ROOT / "data" / "agent_v2_subagents.jsonl"


def ledger_path() -> Path:
    return Path(os.environ.get("AGENT_V2_SUBAGENT_LEDGER") or _DEFAULT_PATH)


def rows_for(result: AgentResult, *, channel: str = "") -> list[dict[str, Any]]:
    """One ledger row per sub-agent run in ``result`` (nested reader runs included, flagged as such)."""

    at = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    question = (result.request.original_text or result.request.text or "")[:80]
    rows: list[dict[str, Any]] = []
    for entry in sub_agent_summaries(result.results):
        base = {"at": at, "run_id": result.run_id, "channel": channel or str(result.request.metadata.get("channel") or ""), "question": question, "status": result.status.value}
        rows.append({**base, "agent": entry["name"], "capability": entry["capability"], "subject": entry["subject"], "rounds": entry["rounds"], "llm_calls": entry["llm_calls"], "elapsed_ms": entry["elapsed_ms"], "seconds_allowed": entry["seconds_allowed"], "stop_reason": entry["stop_reason"], "calls": entry["calls"], "yield": entry["yield"], "intraday": entry["intraday"], "nested": False})
        for nested in entry["nested"]:
            rows.append({**base, "agent": "filing_reader", "capability": entry["capability"], "subject": entry["subject"], "rounds": nested["rounds"], "llm_calls": None, "elapsed_ms": nested["elapsed_ms"], "seconds_allowed": None, "stop_reason": nested["stop_reason"], "calls": nested["calls"], "yield": {"kept": nested["calls"].get("events"), "dropped": None}, "intraday": entry["intraday"], "nested": True})
    return rows


def record_runs(result: AgentResult, *, channel: str = "", path: Path | None = None) -> int:
    """Append the run's sub-agent rows; best effort, never raises. Returns the rows written."""

    rows = rows_for(result, channel=channel)
    if not rows:
        return 0
    target = path or ledger_path()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("sub-agent ledger not written (%s): %s", target, exc)
        return 0
    return len(rows)


def read_rows(path: Path | None = None, *, since_days: int | None = None) -> list[dict[str, Any]]:
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
            if isinstance(row, dict) and (not cutoff or str(row.get("at") or "") >= cutoff):
                rows.append(row)
    return rows


def _percentile(values: list[float], share: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(share * (len(ordered) - 1)))))
    return ordered[index]


def aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per agent: runs, rounds, time, stop reasons, calls, yield and drop rate."""

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("agent") or "?")].append(row)
    summary: dict[str, dict[str, Any]] = {}
    for agent, items in sorted(groups.items()):
        rounds = [float(item.get("rounds") or 0) for item in items]
        elapsed = [float(item.get("elapsed_ms") or 0) / 1000 for item in items]
        stops = Counter(str(item.get("stop_reason") or "?") for item in items)
        calls: Counter = Counter()
        for item in items:
            for key, value in (item.get("calls") or {}).items():
                if isinstance(value, (int, float)):
                    calls[key] += value
        kept = sum(int((item.get("yield") or {}).get("kept") or 0) for item in items)
        dropped = sum(int((item.get("yield") or {}).get("dropped") or 0) for item in items)
        confirmed = sum(int((item.get("yield") or {}).get("confirmed") or 0) for item in items)
        reported = kept + dropped
        summary[agent] = {
            "runs": len(items),
            "rounds_avg": round(sum(rounds) / len(rounds), 2) if rounds else 0.0,
            "seconds_avg": round(sum(elapsed) / len(elapsed), 1) if elapsed else 0.0,
            "seconds_p90": round(_percentile(elapsed, 0.9), 1),
            "stop_reasons": dict(stops),
            "calls": dict(calls),
            "kept": kept,
            "dropped": dropped,
            "confirmed": confirmed,
            "drop_rate": round(dropped / reported, 3) if reported else None,
            "kept_per_run": round(kept / len(items), 2) if items else 0.0,
            "empty_runs": sum(1 for item in items if not int((item.get("yield") or {}).get("kept") or 0)),
        }
    return summary


#: Weights that turn a call's token counts into one figure, the standard token
#: equivalent: ``uncached input × 1 + cached input × 1/30 + output × 3``.  They
#: are the ratios of DeepSeek's list prices (cached input costs 1/30 of uncached,
#: output costs 3× uncached) and hold in both the peak and the off-peak period,
#: so the equivalent compares runs made at different times of day, which money
#: does not.  Money, when the ledger has a price version, stays a trailing column.
TOKEN_WEIGHTS: dict[str, float] = {"input": 1.0, "cached_input": 1.0 / 30.0, "output": 3.0}


def token_equivalent(input_tokens: float, cached_tokens: float, output_tokens: float) -> float:
    """Standard token equivalent of one call or one total.

    ``input_tokens`` is the whole prompt as the provider reports it (cache hits
    included); ``cached_tokens`` is the part served from cache.
    """

    total_input = max(0.0, float(input_tokens or 0))
    cached = min(total_input, max(0.0, float(cached_tokens or 0)))
    output = max(0.0, float(output_tokens or 0))
    return (total_input - cached) * TOKEN_WEIGHTS["input"] + cached * TOKEN_WEIGHTS["cached_input"] + output * TOKEN_WEIGHTS["output"]


def _usage_events(since_days: int | None) -> list[tuple[dict[str, Any], Any]]:
    """The ``agent_v2.*`` LLM events (payload, usd cost) from the usage ledger; empty when it is unavailable."""

    try:
        from v2.data.cost_ledger import _conn
    except Exception:  # noqa: BLE001 — the ledger is optional infrastructure
        return []
    cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=since_days)).isoformat() if since_days else ""
    events: list[tuple[dict[str, Any], Any]] = []
    try:
        with _conn() as conn:
            query = "SELECT payload, cost_usd FROM usage_events WHERE category='llm'" + (" AND occurred_at>=?" if cutoff else "")
            for payload, cost in conn.execute(query, (cutoff,) if cutoff else ()):
                try:
                    event = json.loads(payload)
                except (TypeError, json.JSONDecodeError):
                    continue
                if str(event.get("source") or "").startswith("agent_v2."):
                    events.append((event, cost))
    except Exception as exc:  # noqa: BLE001
        logger.warning("usage ledger unavailable for the sub-agent report: %s", exc)
        return []
    return events


def _event_tokens(event: dict[str, Any]) -> tuple[float, float, float]:
    usage = event.get("usage") or {}
    input_tokens = float(usage.get("input_tokens") or 0)
    cached_tokens = min(input_tokens, float(usage.get("cached_tokens") or 0))
    return input_tokens, cached_tokens, float(usage.get("output_tokens") or 0)


def usage_by_run(since_days: int | None = None, *, events: list[tuple[dict[str, Any], Any]] | None = None) -> dict[str, dict[str, Any]]:
    """Per run id (one question): calls, standard equivalent and the equivalent per source; events without a run id are skipped."""

    runs: dict[str, dict[str, Any]] = defaultdict(lambda: {"calls": 0, "equivalent": 0.0, "sources": {}, "at": ""})
    for event, _cost in (_usage_events(since_days) if events is None else events):
        run_id = str(event.get("run_id") or "")
        if not run_id:
            continue
        bucket = runs[run_id]
        bucket["calls"] += 1
        equivalent = token_equivalent(*_event_tokens(event))
        bucket["equivalent"] += equivalent
        source = str(event.get("source") or "")
        bucket["sources"][source] = bucket["sources"].get(source, 0.0) + equivalent
        bucket["at"] = max(bucket["at"], str(event.get("occurred_at") or ""))
    return {run_id: {**bucket, "equivalent": round(bucket["equivalent"], 1), "sources": {source: round(value, 1) for source, value in bucket["sources"].items()}} for run_id, bucket in runs.items()}


def usage_by_source(since_days: int | None = None, *, events: list[tuple[dict[str, Any], Any]] | None = None) -> dict[str, dict[str, Any]]:
    """Token totals (and the standard equivalent) per ``agent_v2.*`` source from the usage ledger; empty when the ledger is unavailable."""

    totals: dict[str, dict[str, Any]] = defaultdict(lambda: {"calls": 0, "input_tokens": 0.0, "cached_tokens": 0.0, "output_tokens": 0.0, "reasoning_tokens": 0.0, "equivalent": 0.0, "cost": {}, "unpriced": 0, "unpriced_reasons": {}, "failed": 0})
    for event, cost in (_usage_events(since_days) if events is None else events):
        source = str(event.get("source") or "")
        bucket = totals[source]
        bucket["calls"] += 1
        input_tokens, cached_tokens, output_tokens = _event_tokens(event)
        bucket["input_tokens"] += input_tokens
        bucket["cached_tokens"] += cached_tokens
        bucket["output_tokens"] += output_tokens
        bucket["reasoning_tokens"] += float((event.get("usage") or {}).get("reasoning_tokens") or 0)
        bucket["equivalent"] += token_equivalent(input_tokens, cached_tokens, output_tokens)
        # The ledger prices each event in the provider's currency
        # (DeepSeek in CNY); the USD column is only filled for USD.
        amount, currency = event.get("amount"), str(event.get("currency") or "")
        if amount is None and cost is not None:
            amount, currency = cost, "USD"
        if amount is not None and currency:
            bucket["cost"][currency] = bucket["cost"].get(currency, 0.0) + float(amount)
        else:
            bucket["unpriced"] += 1
            reason = str(event.get("reason") or "未说明")
            bucket["unpriced_reasons"][reason] = bucket["unpriced_reasons"].get(reason, 0) + 1
        if event.get("state") != "success":
            bucket["failed"] += 1
    return {source: {**bucket, "equivalent": round(bucket["equivalent"], 1), "cost": {currency: round(value, 4) for currency, value in bucket["cost"].items()}} for source, bucket in sorted(totals.items())}


def usage_totals(usage: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The same buckets summed over every source (the 合计 row)."""

    total: dict[str, Any] = {"calls": 0, "input_tokens": 0.0, "cached_tokens": 0.0, "output_tokens": 0.0, "reasoning_tokens": 0.0, "equivalent": 0.0, "cost": {}, "unpriced": 0, "unpriced_reasons": {}, "failed": 0}
    for row in usage.values():
        for key in ("calls", "input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens", "equivalent", "unpriced", "failed"):
            total[key] += row.get(key) or 0
        for currency, value in (row.get("cost") or {}).items():
            total["cost"][currency] = round(total["cost"].get(currency, 0.0) + value, 4)
        for reason, count in (row.get("unpriced_reasons") or {}).items():
            total["unpriced_reasons"][reason] = total["unpriced_reasons"].get(reason, 0) + count
    total["equivalent"] = round(total["equivalent"], 1)
    return total


def cost_label(row: dict[str, Any]) -> str:
    """``0.0123 CNY`` (one figure per currency), with how many calls had no price."""

    parts = [f"{value:.4f} {currency}" for currency, value in sorted((row.get("cost") or {}).items())]
    if row.get("unpriced"):
        reasons = row.get("unpriced_reasons") or {}
        top = max(reasons.items(), key=lambda item: item[1])[0] if reasons else ""
        parts.append(f"{row['unpriced']} 次待定价" + (f"（{top}）" if top else ""))
    return "、".join(parts) if parts else "—"


def _thousands(value: float) -> str:
    return f"{int(round(value)):,}"


def question_count(rows: list[dict[str, Any]]) -> int:
    """Distinct runs in the ledger: the questions that used at least one sub-agent."""

    return len({str(row.get("run_id") or "") for row in rows if row.get("run_id")})


def _usage_row(source: str, row: dict[str, Any], summary: dict[str, dict[str, Any]], *, priced: bool, questions: int = 0, tagged: float | None = None) -> str:
    """``tagged`` is the equivalent of the calls that carry a run id (what the per-question figure divides); None when the whole total does."""

    input_tokens, cached = float(row.get("input_tokens") or 0), float(row.get("cached_tokens") or 0)
    equivalent = float(row.get("equivalent") or 0)
    calls = int(row.get("calls") or 0)
    runs = (summary.get(source.removeprefix("agent_v2.")) or {}).get("runs") if source.startswith("agent_v2.") else None
    per_call = _thousands(equivalent / calls) if calls else "—"
    per_run = _thousands(equivalent / runs) if runs else "—"
    per_question = _thousands((equivalent if tagged is None else tagged) / questions) if questions else "—"
    hit = f"{cached / input_tokens:.0%}" if input_tokens else "—"
    output = float(row.get("output_tokens") or 0)
    reasoning = float(row.get("reasoning_tokens") or 0)
    thinking = f"{reasoning / output:.0%}" if output and reasoning else ("—" if not output else "0%")
    cells = [source, str(calls), _thousands(input_tokens - cached), _thousands(cached), _thousands(output), thinking, _thousands(equivalent), per_call, per_run, per_question, hit, str(row.get("failed") or 0)]
    if priced:
        cells.append(cost_label(row))
    return "| " + " | ".join(cells) + " |"


def top_questions(runs: dict[str, dict[str, Any]], rows: list[dict[str, Any]], *, limit: int = 5) -> list[dict[str, Any]]:
    """The most expensive runs, with the question text from the sub-agent ledger when that run used one."""

    questions = {str(row.get("run_id") or ""): str(row.get("question") or "") for row in rows if row.get("run_id")}
    ranked = sorted(runs.items(), key=lambda item: (-float(item[1].get("equivalent") or 0), item[0]))[: max(1, limit)]
    result = []
    for run_id, bucket in ranked:
        sources = sorted((bucket.get("sources") or {}).items(), key=lambda item: -item[1])
        result.append({"run_id": run_id, "question": questions.get(run_id, ""), "at": str(bucket.get("at") or "")[:16].replace("T", " "), "calls": int(bucket.get("calls") or 0), "equivalent": float(bucket.get("equivalent") or 0), "top_sources": [(source.removeprefix("agent_v2."), value) for source, value in sources[:3]]})
    return result


def render(summary: dict[str, dict[str, Any]], usage: dict[str, dict[str, Any]], *, since_days: int | None, questions: int = 0, runs: dict[str, dict[str, Any]] | None = None, rows: list[dict[str, Any]] | None = None) -> str:
    """``questions`` is the number of questions behind the usage table: the usage ledger's distinct run ids
    when it has them, otherwise the sub-agent ledger's (an upper bound on per-question figures)."""

    lines = [f"# 子智能体运行报告{f'（近 {since_days} 天）' if since_days else ''}", ""]
    exact = bool(runs)
    if exact:
        questions = len(runs)
        lines.append(f"用量账本里有 {questions} 个问题（按 run_id 计）。")
        lines.append("")
    elif questions:
        lines.append(f"子智能体账本里有 {questions} 个用到子智能体的问题。")
        lines.append("")
    if not summary:
        lines.append("账本里还没有子智能体运行记录。")
    else:
        lines.append("| 子智能体 | 运行 | 平均轮次 | 平均秒 | P90 秒 | 停止原因 | 产出/次 | 丢弃率 | 空跑 |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for agent, row in summary.items():
            stops = "、".join(f"{key} {value}" for key, value in sorted(row["stop_reasons"].items()))
            drop = "—" if row["drop_rate"] is None else f"{row['drop_rate']:.0%}"
            lines.append(f"| {agent} | {row['runs']} | {row['rounds_avg']} | {row['seconds_avg']} | {row['seconds_p90']} | {stops} | {row['kept_per_run']} | {drop} | {row['empty_runs']} |")
        lines.append("")
        lines.append("产出 = 校验后保留的原因或事件数；丢弃率 = 被校验器丢弃的占报出总数的比例；空跑 = 一条都没保留的运行。")
    lines.append("")
    if usage:
        priced = any(row.get("cost") for row in usage.values())
        header = ["来源", "模型调用", "未缓存输入", "缓存输入", "输出", "其中推理", "标准当量", "当量/调用", "当量/次运行", "当量/问题", "缓存命中", "失败"] + (["估算成本"] if priced else [])
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        # With run ids the per-question figure divides only the tagged calls; older untagged calls stay in the totals.
        tagged: dict[str, float] = defaultdict(float)
        for run in (runs or {}).values():
            for source, value in (run.get("sources") or {}).items():
                tagged[source] += float(value)
        for source, row in usage.items():
            lines.append(_usage_row(source, row, summary, priced=priced, questions=questions, tagged=tagged.get(source, 0.0) if exact else None))
        lines.append(_usage_row("合计", usage_totals(usage), {}, priced=priced, questions=questions, tagged=sum(tagged.values()) if exact else None))
        lines.append("")
        lines.append(f"标准当量 = 未缓存输入 × {TOKEN_WEIGHTS['input']:g} + 缓存输入 × 1/{round(1 / TOKEN_WEIGHTS['cached_input'])} + 输出 × {TOKEN_WEIGHTS['output']:g}（按 DeepSeek 价格比例折算，高峰和空闲时段一样，所以不同时段的运行可以直接比）；"
                     "当量/次运行按子智能体账本里的运行数算，规划器和合成器没有运行数；"
                     + ("当量/问题按用量账本里的 run_id 数算，每个问题一个 run_id；" if exact else "当量/问题按子智能体账本里用到子智能体的问题数算，没用子智能体的问题不在分母里，所以是上限；")
                     + "缓存命中 = 缓存输入占全部输入的比例；其中推理 = 输出里模型推理 token 的占比（账本没记推理 token 时为 0%）。")
        untagged = sum(int(row.get("calls") or 0) for row in usage.values()) - sum(int(run.get("calls") or 0) for run in (runs or {}).values())
        if runs and untagged > 0:
            lines.append(f"另有 {untagged} 次调用没有 run_id（记 run_id 之前的旧记录），不在当量/问题的分母里。")
        ranked = top_questions(runs, rows or []) if runs else []
        if ranked:
            lines.append("")
            lines.append("| 最贵的问题 | 时间 | 模型调用 | 标准当量 | 主要来源 |")
            lines.append("|---|---|---|---|---|")
            for entry in ranked:
                sources = "、".join(f"{name} {_thousands(value)}" for name, value in entry["top_sources"])
                lines.append(f"| {entry['question'] or '（无子智能体记录）'} | {entry['at']} | {entry['calls']} | {_thousands(entry['equivalent'])} | {sources} |")
    else:
        lines.append("用量账本不可用或没有 agent_v2.* 的记录（Token 按来源归属需要线上账本）。")
    return "\n".join(lines)
