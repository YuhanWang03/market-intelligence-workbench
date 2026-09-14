"""The merge gate for Agent V2: one command, one verdict.

    python scripts/agent_v2_gate.py            # offline: unit tests + offline eval (no key, no network)
    python scripts/agent_v2_gate.py --live     # also the quick quality subset through the real model (VPS, .env)

Offline it runs the Agent V2 unit tests and the offline evaluation suite
(every recorded case must pass).  With ``--live`` it also runs the quick
quality subset (``quality run --quick``) and requires most of it to pass
(``--min-quick``, default all but one: a single model jitter does not block a
merge, two failures do).  The verdict is printed last and is the exit code.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNIT_TESTS = ("v2/agent_v2/test_agent_v2.py", "v2/agent_v2/eval/test_benchmark.py")


def run_unit_tests() -> tuple[bool, str]:
    proc = subprocess.run([sys.executable, "-m", "pytest", *UNIT_TESTS, "-q", "-p", "no:cacheprovider", "-rf"], cwd=ROOT, capture_output=True, text=True)
    lines = proc.stdout.strip().splitlines() or [""]
    failed = [line.removeprefix("FAILED ").split(" - ")[0] for line in lines if line.startswith("FAILED ")]
    # The names of the failing tests travel with the verdict, so a CI log tail is enough to act on.
    return proc.returncode == 0, lines[-1] + (f"; failed: {', '.join(failed[:6])}" if failed else "")


def run_offline_eval() -> tuple[bool, str]:
    sys.path.insert(0, str(ROOT))
    from v2.agent_v2.eval.runner import run_suite

    report = run_suite()
    failed = [score.case_id for score in (*report.scores, *report.answer_scores, *report.scenario_scores) if not score.passed]
    return not failed, f"{report.passed}/{report.total}" + (f" failed: {', '.join(failed)}" if failed else "")


def run_quick_quality(label: str, min_pass: int | None) -> tuple[bool, str, dict]:
    sys.path.insert(0, str(ROOT))
    from v2.agent_v2.eval import quality
    from v2.agent_v2.eval.quality_cases import QUALITY_CASES

    quick = tuple(case for case in QUALITY_CASES if "quick" in case.tags)
    agent = quality._live_agent(use_web=True)
    judge = quality.QualityJudge(getattr(agent.synthesizer, "llm", None))
    rows = quality.run_cases(agent, quick, judge, label=label, repeat=1, parallel=2)
    passed = sum(1 for row in rows if row["score"]["passed"])
    needed = len(quick) - 1 if min_pass is None else min_pass
    failed = [f"{row['case_id']}（{'; '.join(row['score']['problems'])[:80]}）" for row in rows if not row["score"]["passed"]]
    detail = f"{passed}/{len(quick)} passed, need {needed}" + (f"; failed: {'; '.join(failed)}" if failed else "")
    return passed >= needed, detail, {"passed": passed, "total": len(quick), "needed": needed, "label": label}


def verdict(steps: list[tuple[str, bool, str]]) -> bool:
    """The gate passes when every step passed."""

    return all(ok for _, ok, _ in steps)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Agent V2 merge gate.")
    parser.add_argument("--live", action="store_true", help="also run the quick quality subset through the real model")
    parser.add_argument("--min-quick", type=int, default=None, help="quick cases that must pass with --live (default: all but one)")
    parser.add_argument("--skip-unit", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--json", action="store_true", help="print the steps as JSON before the verdict")
    args = parser.parse_args(argv)

    steps: list[tuple[str, bool, str]] = []
    started = time.monotonic()
    if not args.skip_unit:
        ok, detail = run_unit_tests()
        steps.append(("unit tests", ok, detail))
        print(f"[{'PASS' if ok else 'FAIL'}] unit tests: {detail}", flush=True)
    if not args.skip_eval:
        ok, detail = run_offline_eval()
        steps.append(("offline eval", ok, detail))
        print(f"[{'PASS' if ok else 'FAIL'}] offline eval: {detail}", flush=True)
    if args.live:
        label = "gate-" + datetime.now(tz=timezone.utc).strftime("%m%d-%H%M")
        ok, detail, _ = run_quick_quality(label, args.min_quick)
        steps.append(("quick quality", ok, detail))
        print(f"[{'PASS' if ok else 'FAIL'}] quick quality ({label}): {detail}", flush=True)
    passed = verdict(steps)
    if args.json:
        print(json.dumps([{"step": name, "ok": ok, "detail": detail} for name, ok, detail in steps], ensure_ascii=False))
    print(f"gate: {'PASS' if passed else 'FAIL'} ({len(steps)} steps, {time.monotonic() - started:.0f}s)")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
