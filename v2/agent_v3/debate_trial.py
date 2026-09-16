"""Opt-in live trial of the Agent V3 adversarial review (``debate``) node.

Never part of offline CI; the output is a set of observations, not a score.

Each question runs twice through the full workspace agent — once with
``debate=True`` and once with ``debate=False`` — in isolated sessions and
data directories.  For every run the report records whether the debate node
actually executed, why it was skipped when it did not, the objections the
reviewer raised, whether the revised answer replaced the original, elapsed
time and token usage.  ``report.json`` keeps the full results; ``summary.md``
is the table to read.

    .venv-agent-v3\\Scripts\\python.exe -m v2.agent_v3.debate_trial --deepseek-defaults
    .venv-agent-v3\\Scripts\\python.exe -m v2.agent_v3.debate_trial --demo   # plumbing only, no model
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: Research questions on public data whose verification usually passes, so the
#: debate gate (route == research, verification ok, no cross-reviewed result)
#: has a chance to open.  The last one is a news question: its result carries
#: ``review_stage == source_cross_review`` and documents the skip path.
QUESTIONS = (
    "分析一下苹果",
    "分析一下甲骨文",
    "比较 NVDA 和 AMD 的风险",
    "分析一下 MU 的估值",
    "分析一下 NVDA 最近的财务状况",
    "苹果最近有哪些新闻？",
)


def _tokens(usage_rows) -> dict[str, int]:
    total = {"input": 0, "output": 0}
    for row in usage_rows or ():
        if not isinstance(row, dict):
            continue
        total["input"] += int(row.get("input_tokens") or row.get("prompt_tokens") or 0)
        total["output"] += int(row.get("output_tokens") or row.get("completion_tokens") or 0)
    return total


def analyse(result: dict, debate_enabled: bool) -> dict:
    """Classify one run: did debate execute, why not, and what came of it."""
    synthesis = result.get("synthesis") or {}
    nodes = list(synthesis.get("nodes") or [])
    record = synthesis.get("debate") or {}
    objections = list(record.get("objections") or synthesis.get("objections") or [])
    attempts = list(synthesis.get("attempts") or [])
    ran = bool(record.get("ran")) or "debate" in nodes
    route = (result.get("route") or {}).get("kind", "")
    verification_ok = bool((result.get("verification") or {}).get("ok"))
    cross_reviewed = any(((row.get("metadata") or {}).get("review_stage") == "source_cross_review") for row in result.get("results") or [])
    if ran:
        if record.get("error"):
            outcome = "reviewer_error"
        elif not objections:
            outcome = "no_objections"
        elif not record.get("revised"):
            outcome = "minor_only"
        elif record.get("revised_verified"):
            outcome = "revised_accepted"
        else:
            outcome = "revised_rejected"
        skipped = ""
    else:
        outcome = ""
        if result.get("status") in {"failed", "cancelled"} or result.get("error"):
            skipped = "run_failed"
        elif record.get("skipped"):
            skipped = record["skipped"]  # recorded by the graph itself
        elif not debate_enabled:
            skipped = "disabled"
        else:
            skipped = "unrecorded"
    return {
        "debate_ran": ran,
        "skip_reason": skipped,
        "outcome": outcome,
        "objection_count": len(objections),
        "material_count": sum(1 for row in objections if not isinstance(row, dict) or row.get("severity", "material") == "material"),
        "objections": objections,
        "route": route,
        "status": result.get("status"),
        "verification_ok": verification_ok,
        "synthesis_outcome": (result.get("synthesis") or {}).get("outcome"),
        "attempts": len(attempts),
        "nodes": nodes,
        "cross_reviewed": cross_reviewed,
        "elapsed_ms": result.get("elapsed_ms"),
        "tokens": _tokens((result.get("synthesis") or {}).get("usage")),
        "evidence": len(result.get("evidence") or []),
    }


def _summary(rows: list[dict]) -> str:
    lines = [
        "# Agent V3 debate trial",
        "",
        "| # | question | mode | status | s | debate ran | skipped because | objections (material) | outcome | verify ok | in/out tokens |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        a = row["analysis"]
        tokens = a["tokens"]
        lines.append(
            f"| {row['case']} | {row['question']} | {row['mode']} | {a['status']} | {round((a['elapsed_ms'] or 0) / 1000)} | "
            f"{'yes' if a['debate_ran'] else 'no'} | {a['skip_reason'] or '-'} | {a['objection_count']} ({a.get('material_count', a['objection_count'])}) | {a['outcome'] or '-'} | "
            f"{'yes' if a['verification_ok'] else 'no'} | {tokens['input']}/{tokens['output']} |"
        )
    lines.append("")
    for row in rows:
        if row["analysis"]["objections"]:
            lines.append(f"## Objections · case {row['case']} ({row['mode']}) · {row['question']}")
            for objection in row["analysis"]["objections"]:
                if isinstance(objection, dict):
                    lines.append(f"- ({objection.get('severity', 'material')}) [{objection.get('evidence_id', '?')}] {objection.get('objection') or json.dumps(objection, ensure_ascii=False)}" + (f"\n  > {objection['claim']}" if objection.get('claim') else ""))
                else:
                    lines.append(f"- {objection}")
            lines.append("")
    return "\n".join(lines)


def _apply_deepseek_defaults() -> None:
    """Mirror web/deploy/agent-v3-server.py: only when the V3 trio is unset and a DeepSeek key exists."""
    if os.environ.get("AGENT_V3_API_KEY") or os.environ.get("AGENT_LLM_API_KEY"):
        return
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise SystemExit("--deepseek-defaults needs DEEPSEEK_API_KEY in the environment or .env")
    os.environ.setdefault("AGENT_V3_MODEL", os.environ.get("AGENT_LLM_MODEL") or "deepseek-v4-flash")
    os.environ.setdefault("AGENT_V3_BASE_URL", os.environ.get("AGENT_LLM_BASE_URL") or "https://api.deepseek.com/v1")
    os.environ.setdefault("AGENT_V3_API_KEY", key)
    os.environ.setdefault("AGENT_V3_THINKING", "disabled")
    print(f"model: {os.environ['AGENT_V3_MODEL']} @ {os.environ['AGENT_V3_BASE_URL']} (thinking {os.environ['AGENT_V3_THINKING']})", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env-file", help="extra .env to load (repo .env is loaded first)")
    parser.add_argument("--output-dir", default="data/agent_v3/debate_trial")
    parser.add_argument("--max-seconds", type=float, default=180)
    parser.add_argument("--modes", default="on,off", help="comma list of on/off (default both)")
    parser.add_argument("--no-web", action="store_true", help="run without the web search allowance")
    parser.add_argument("--questions", nargs="*", help="override the default question list")
    parser.add_argument("--deepseek-defaults", action="store_true", help="derive AGENT_V3_* from DEEPSEEK_API_KEY when unset")
    parser.add_argument("--demo", action="store_true", help="offline plumbing check with the fixture agent (debate cannot run)")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # cmd.exe code pages

    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    if args.env_file:
        load_dotenv(args.env_file, override=False)

    modes = [m.strip() for m in args.modes.split(",") if m.strip() in {"on", "off"}]
    questions = tuple(args.questions) if args.questions else QUESTIONS
    root = Path(args.output_dir) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root.mkdir(parents=True, exist_ok=False)
    web = not args.no_web

    from v2.agent_v3.graph import AgentV3Config

    agents: dict[str, object] = {}
    if args.demo:
        from v2.agent_v3.demo import DEMO_QUESTION, build_demo_agent

        questions = (DEMO_QUESTION,)
        modes = ["off"]
        agents["off"] = build_demo_agent()
        web = False
    else:
        if args.deepseek_defaults:
            _apply_deepseek_defaults()
        from v2.agent_v3.runtime import build_workspace_agent

        for mode in modes:
            agents[mode] = build_workspace_agent(config=AgentV3Config(enable_web=web, debate=(mode == "on"), max_seconds=args.max_seconds), data_dir=root / f"data-{mode}")

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": os.environ.get("AGENT_V3_MODEL") or os.environ.get("AGENT_LLM_MODEL") or "(demo)",
        "thinking": os.environ.get("AGENT_V3_THINKING"),
        "conditions": "Full workspace handlers; web " + ("allowed" if web else "off") + "; each question once per mode in an isolated session; live data, so answers are not reproducible; observations, not a quality score.",
        "runs": [],
    }
    rows: list[dict] = []
    try:
        for index, question in enumerate(questions, start=1):
            for mode in modes:
                print(f"START case={index} mode={mode} q={question}", flush=True)
                started = time.monotonic()
                try:
                    result = agents[mode].run(question, session_id=f"debate-trial-{mode}-{index}", allow_web=web).to_dict()
                except Exception as exc:  # noqa: BLE001 — one failed run must not end the trial
                    result = {"status": "failed", "error": f"{type(exc).__name__}: {str(exc)[:300]}", "elapsed_ms": int((time.monotonic() - started) * 1000)}
                analysis = analyse(result, debate_enabled=(mode == "on"))
                row = {"case": index, "question": question, "mode": mode, "analysis": analysis, "result": result}
                rows.append(row)
                report["runs"].append(row)
                (root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                (root / "summary.md").write_text(_summary(rows), encoding="utf-8")
                print(
                    f"END   case={index} mode={mode} status={analysis['status']} elapsed={round((analysis['elapsed_ms'] or 0) / 1000)}s "
                    f"debate={'ran' if analysis['debate_ran'] else 'skipped:' + analysis['skip_reason']} objections={analysis['objection_count']} outcome={analysis['outcome'] or '-'}",
                    flush=True,
                )
    finally:
        for agent in agents.values():
            for closer in (getattr(getattr(agent, "store", None), "close", None), getattr(getattr(agent, "checkpoint_connection", None), "close", None)):
                if closer:
                    try:
                        closer()
                    except Exception:  # noqa: BLE001
                        pass
    print(f"REPORT  {root / 'report.json'}")
    print(f"SUMMARY {root / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
