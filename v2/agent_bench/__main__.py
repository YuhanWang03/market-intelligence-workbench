"""CLI: list the cases, run them, judge pairs, render the report."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from v2.agent_bench import agents as agent_builders
from v2.agent_bench.bank import Bank
from v2.agent_bench.cases import CATEGORIES, all_cases, by_id, select
from v2.agent_bench.runner import DEFAULT_WORKDIR, Run, read_ledger


def _cases(args) -> list:
    sets = {"dev": ("dev",), "holdout": ("holdout",), "all": ("dev", "holdout")}[args.set]
    chosen = select(all_cases(), sets=sets, ids=tuple(args.cases or ()), categories=tuple(args.categories or ()), tags=tuple(args.tags or ()), mode=getattr(args, "mode", ""))
    if not chosen:
        raise SystemExit("no cases selected")
    return chosen


def cmd_list(args) -> int:
    cases = _cases(args)
    if args.set != "dev" and not args.show_holdout:
        holdout = [c for c in cases if c.set == "holdout"]
        cases = [c for c in cases if c.set != "holdout"]
        print(f"({len(holdout)} holdout cases hidden; pass --show-holdout only when writing the final report)")
    for c in cases:
        flags = ("W" if c.allow_web else "-") + ("F" if c.frozen_only else "-") + ("!" if c.fault else "-") + ("p" if c.preceding else "-")
        print(f"{c.id:36} {c.category:12} {flags} {c.origin:14} {c.question[:60]}")
    counts = {}
    for c in cases:
        counts[c.category] = counts.get(c.category, 0) + 1
    print(f"\n{len(cases)} cases · " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    return 0


def cmd_run(args) -> int:
    cases = _cases(args)
    judge = None
    judge_meta = {}
    if not args.no_judge and args.mode != "offline":
        from v2.agent_bench.judge import judge_llm, rubric_judge
        llm, judge_meta = judge_llm()
        judge = rubric_judge(llm)
        if judge_meta.get("judge_same_as_agent") == "true":
            print("warning: judge model is the agents' model; set AGENT_BENCH_JUDGE_MODEL/BASE_URL/API_KEY to a different family for less self-preference", file=sys.stderr)
    run = Run(label=args.label, mode=args.mode, versions=tuple(args.versions), seconds=args.seconds, workdir=Path(args.workdir), bank=Bank(Path(args.bank)) if args.bank else Bank(),
              judge=judge, debate=args.debate, repeat=args.repeat, progress=lambda message: print(message, flush=True), seed_watchlist=tuple(args.seed_watchlist))
    run.build()
    if judge_meta:
        conditions = json.loads((run.root / "conditions.json").read_text(encoding="utf-8"))
        conditions.update(judge_meta)
        (run.root / "conditions.json").write_text(json.dumps(conditions, ensure_ascii=False, indent=1), encoding="utf-8")
    try:
        rows = run.run(cases)
    finally:
        run.close()
    from v2.agent_bench.report import write_report
    path = write_report(run.root, read_ledger(run.root))
    passed = {v: sum(1 for r in rows if r["version"] == v and r["score"]["passed"]) for v in run.versions}
    print(f"\n{len(rows)} attempts · passed per version: {passed}\nREPORT {path}")
    return 0


def cmd_pair(args) -> int:
    root = Path(args.workdir) / args.label
    rows = read_ledger(root)
    if not rows:
        raise SystemExit(f"no ledger under {root}")
    from v2.agent_bench.judge import PairJudge, compare_pair, judge_llm
    llm, meta = judge_llm()
    judge = PairJudge(llm)
    table = by_id(all_cases())
    answers: dict[str, dict[str, str]] = {}
    for row in rows:
        if row["attempt"] == args.attempt and row["answer"]:
            answers.setdefault(row["case_id"], {})[row["version"]] = row["answer"]
    pairs = []
    for case_id, by_version in sorted(answers.items()):
        if {"v2", "v3"} <= set(by_version) and case_id in table:
            print(f"PAIR {case_id}", flush=True)
            verdict = compare_pair(judge, table[case_id], by_version, seed=args.seed)
            pairs.append(verdict)
            print(f"     -> {verdict['outcome']} (orders: {verdict['winners_by_order']})", flush=True)
    (root / "pairs.json").write_text(json.dumps({"seed": args.seed, "attempt": args.attempt, **meta, "pairs": pairs}, ensure_ascii=False, indent=1), encoding="utf-8")
    from v2.agent_bench.report import write_report
    path = write_report(root, rows, pairs)
    print(f"{len(pairs)} pairs judged\nREPORT {path}")
    return 0


def cmd_report(args) -> int:
    root = Path(args.workdir) / args.label
    rows = read_ledger(root)
    if not rows:
        raise SystemExit(f"no ledger under {root}")
    pairs_path = root / "pairs.json"
    pairs = json.loads(pairs_path.read_text(encoding="utf-8"))["pairs"] if pairs_path.exists() else None
    from v2.agent_bench.report import write_report
    path = write_report(root, rows, pairs)
    print(path.read_text(encoding="utf-8"))
    return 0


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(prog="python -m v2.agent_bench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def selection(p):
        p.add_argument("--set", choices=("dev", "holdout", "all"), default="dev")
        p.add_argument("--cases", nargs="*", help="case ids")
        p.add_argument("--categories", nargs="*", choices=CATEGORIES)
        p.add_argument("--tags", nargs="*")

    p = sub.add_parser("list", help="show the selected cases")
    selection(p)
    p.add_argument("--show-holdout", action="store_true")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("run", help="run cases through the agents and grade them")
    selection(p)
    p.add_argument("--label", required=True)
    p.add_argument("--mode", choices=agent_builders.MODES, default="frozen")
    p.add_argument("--versions", nargs="+", choices=("v2", "v3"), default=["v2", "v3"])
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--seconds", type=float, default=180)
    p.add_argument("--debate", action="store_true", help="enable adversarial review in both agents")
    p.add_argument("--no-judge", action="store_true", help="deterministic checks only")
    p.add_argument("--workdir", default=str(DEFAULT_WORKDIR))
    p.add_argument("--bank", help="frozen bank directory (default data/agent_bench/bank)")
    p.add_argument("--seed-watchlist", nargs="*", default=[], help="record/live: tickers written to a bench-owned watchlist store both agents read")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("pair", help="blind pairwise judging over an existing ledger")
    p.add_argument("--label", required=True)
    p.add_argument("--attempt", type=int, default=1)
    p.add_argument("--seed", type=int, default=20260916)
    p.add_argument("--workdir", default=str(DEFAULT_WORKDIR))
    p.set_defaults(func=cmd_pair)

    p = sub.add_parser("report", help="render report.md for a label")
    p.add_argument("--label", required=True)
    p.add_argument("--workdir", default=str(DEFAULT_WORKDIR))
    p.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
