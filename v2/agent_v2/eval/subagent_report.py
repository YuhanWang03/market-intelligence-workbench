"""``python -m v2.agent_v2.eval.subagent_report [--since DAYS] [--path FILE] [--json]``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from v2.agent_v2.eval.subagent_ledger import TOKEN_WEIGHTS, aggregate, ledger_path, _usage_events, question_count, read_rows, render, top_questions, usage_by_run, usage_by_source, usage_totals


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate the sub-agent run ledger and their token usage (standard token equivalent per source).")
    parser.add_argument("--since", type=int, default=None, help="only runs from the last N days")
    parser.add_argument("--path", type=Path, default=None, help=f"ledger file (default {ledger_path()})")
    parser.add_argument("--json", action="store_true", help="print the aggregates as JSON")
    args = parser.parse_args(argv)
    rows = read_rows(args.path, since_days=args.since)
    summary = aggregate(rows)
    events = _usage_events(args.since)
    usage = usage_by_source(events=events)
    runs = usage_by_run(events=events)
    questions = len(runs) or question_count(rows)
    if args.json:
        print(json.dumps({"agents": summary, "usage": usage, "usage_total": usage_totals(usage), "token_weights": TOKEN_WEIGHTS, "rows": len(rows), "questions": questions, "questions_from": "usage_ledger" if runs else "subagent_ledger", "top_questions": top_questions(runs, rows)}, ensure_ascii=False, indent=2))
    else:
        print(render(summary, usage, since_days=args.since, questions=questions, runs=runs, rows=rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
