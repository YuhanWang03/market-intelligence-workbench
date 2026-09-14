"""Capability outcomes: what every data source did on every run, and a report over them.

Each capability execution of a run becomes one row in
``data/agent_v2_capabilities.jsonl``: status, seconds, evidence count,
the first error and the limitations that mark a hollow result ("核心模块
失败", "timed out", "缺失").  The report answers "which sources fail, how
often, and what did they say last time" without reading the service log.

``python -m v2.agent_v2.eval.capability_report [--since DAYS] [--channel quality]``
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from v2.agent_v2.models import AgentResult, ResultStatus

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PATH = _PROJECT_ROOT / "data" / "agent_v2_capabilities.jsonl"

#: Words in a limitation that mark a result the status alone would call fine.
_HOLLOW_WORDS = ("核心模块失败", "timed out", "temporarily unavailable", "不可用", "未完成", "no calendar", "rate limit", "429")


def ledger_path() -> Path:
    return Path(os.environ.get("AGENT_V2_CAPABILITY_LEDGER") or _DEFAULT_PATH)


def rows_for(result: AgentResult, *, channel: str = "") -> list[dict[str, Any]]:
    """One row per capability result of the run; fan-out tables and skipped tasks included, marked."""

    at = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    question = (result.request.original_text or result.request.text or "")[:80]
    session = str(result.request.session_id or "")
    channel = channel or str(result.request.metadata.get("channel") or "") or ("quality" if session.startswith("quality-") else "")
    rows: list[dict[str, Any]] = []
    for envelope in result.results:
        limitations = [str(value) for value in envelope.limitations][:6]
        hollow = [value for value in limitations if any(word in value for word in _HOLLOW_WORDS)]
        rows.append(
            {
                "at": at,
                "run_id": result.run_id,
                "channel": channel,
                "question": question,
                "capability": envelope.capability,
                "subject": str(envelope.subject or "")[:40],
                "status": envelope.status.value,
                "ok": bool(envelope.ok),
                "hollow": bool(hollow),
                "elapsed_ms": int(envelope.elapsed_ms or 0),
                "evidence": len(envelope.evidence),
                "cache_hit": bool(envelope.cache_hit),
                "error": str(envelope.errors[0])[:200] if envelope.errors else "",
                "limitation": (hollow or limitations or [""])[0][:200],
                "table": bool(envelope.metadata.get("fan_out_table")) if isinstance(envelope.metadata, dict) else False,
            }
        )
    return rows


def record_capabilities(result: AgentResult, *, channel: str = "", path: Path | None = None) -> int:
    """Append the run's capability rows; best effort, never raises. Returns the rows written."""

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
        logger.warning("capability ledger not written (%s): %s", target, exc)
        return 0
    return len(rows)


def read_rows(path: Path | None = None, *, since_days: int | None = None, channel: str = "") -> list[dict[str, Any]]:
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
            if not isinstance(row, dict) or not row.get("capability"):
                continue
            if cutoff and str(row.get("at") or "") < cutoff:
                continue
            if channel and str(row.get("channel") or "") != channel:
                continue
            rows.append(row)
    return rows


def _percentile(values: list[int], share: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(share * (len(ordered) - 1)))))
    return ordered[index]


def _error_key(text: str) -> str:
    """Group errors by their first clause; a ticker or a number inside does not make a new kind."""

    head = text.split("\n", 1)[0]
    for sep in ("：", ": ", " - "):
        if sep in head and len(head.split(sep, 1)[0]) >= 8:
            head = head.split(sep, 1)[0] + sep + head.split(sep, 1)[1][:40]
            break
    return head[:80]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per capability: runs, ok rate, statuses, hollow results, seconds, evidence per run, the errors seen most and last."""

    by_capability: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_capability[str(row["capability"])].append(row)
    table = []
    for capability, items in sorted(by_capability.items()):
        real = [row for row in items if not row.get("table")]
        if not real:
            continue
        statuses = Counter(str(row.get("status") or "") for row in real)
        seconds = [int(row.get("elapsed_ms") or 0) for row in real if int(row.get("elapsed_ms") or 0) > 0]
        failures = [row for row in real if not row.get("ok") or row.get("hollow")]
        errors = Counter(_error_key(str(row.get("error") or row.get("limitation") or "")) for row in failures if (row.get("error") or row.get("limitation")))
        last = max(failures, key=lambda row: str(row.get("at") or ""), default=None)
        table.append(
            {
                "capability": capability,
                "runs": len(real),
                "ok": sum(1 for row in real if row.get("ok") and not row.get("hollow")),
                "hollow": sum(1 for row in real if row.get("ok") and row.get("hollow")),
                "failed": statuses.get(ResultStatus.FAILED.value, 0),
                "partial": statuses.get(ResultStatus.PARTIAL_DATA.value, 0) + statuses.get(ResultStatus.PARTIAL_ERROR.value, 0),
                "skipped": statuses.get(ResultStatus.SKIPPED.value, 0),
                "cached": sum(1 for row in real if row.get("cache_hit")),
                "p50_ms": _percentile(seconds, 0.5),
                "p90_ms": _percentile(seconds, 0.9),
                "evidence_per_run": round(sum(int(row.get("evidence") or 0) for row in real) / len(real), 1),
                "top_errors": errors.most_common(3),
                "last_failure": {"at": str(last.get("at") or "")[:16].replace("T", " "), "subject": last.get("subject"), "text": (last.get("error") or last.get("limitation") or "")[:120]} if last else None,
            }
        )
    table.sort(key=lambda row: (row["ok"] / row["runs"] if row["runs"] else 1.0, -row["runs"]))
    runs = {str(row.get("run_id") or "") for row in rows}
    return {"rows": len(rows), "runs": len(runs), "capabilities": table}


def render(summary: dict[str, Any], *, since_days: int | None = None) -> str:
    title = "# 数据源健康报告" + (f"（近 {since_days} 天）" if since_days else "")
    lines = [title, ""]
    if not summary["rows"]:
        lines.append("还没有能力级记录。跑几个问题后再看，或检查 AGENT_V2_CAPABILITY_LEDGER。")
        return "\n".join(lines)
    lines.append(f"{summary['runs']} 次运行，{summary['rows']} 次能力调用。按可靠性从低到高排：")
    lines.append("")
    lines.append("| 能力 | 调用 | 正常 | 空心 | 失败 | 部分 | 跳过 | 缓存 | P50 秒 | P90 秒 | 证据/次 | 最常见的错误 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in summary["capabilities"]:
        errors = "；".join(f"{text}（{count}）" for text, count in row["top_errors"]) or ""
        lines.append(f"| {row['capability']} | {row['runs']} | {row['ok']} | {row['hollow']} | {row['failed']} | {row['partial']} | {row['skipped']} | {row['cached']} | {row['p50_ms'] / 1000:.1f} | {row['p90_ms'] / 1000:.1f} | {row['evidence_per_run']} | {errors} |")
    unhealthy = [row for row in summary["capabilities"] if row["runs"] and row["ok"] / row["runs"] < 0.9]
    if unhealthy:
        lines.append("")
        lines.append("| 需要看的能力 | 正常率 | 最近一次异常 |")
        lines.append("|---|---|---|")
        for row in unhealthy:
            last = row["last_failure"] or {}
            lines.append(f"| {row['capability']} | {row['ok'] / row['runs']:.0%} | {last.get('at', '')} {last.get('subject') or ''}：{last.get('text') or ''} |")
    lines.append("")
    lines.append("空心 = 状态正常但限制里写了核心模块失败、超时或数据不可用；部分 = partial_data 或 partial_error；跳过 = 依赖失败或预算耗尽没有执行。")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Per-capability outcomes over the recent runs.")
    parser.add_argument("--since", type=int, default=1, help="days to look back (default 1)")
    parser.add_argument("--channel", default="", help="only rows of one channel (telegram, quality, web)")
    parser.add_argument("--path", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    summary = summarize(read_rows(args.path, since_days=args.since, channel=args.channel))
    print(json.dumps(summary, ensure_ascii=False, indent=2) if args.json else render(summary, since_days=args.since))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
