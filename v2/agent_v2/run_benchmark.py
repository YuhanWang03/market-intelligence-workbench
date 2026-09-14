"""Run the preserved historical evaluation set against Agent V2.

    python -m v2.agent_v2.run_benchmark                 # v2_rules, dev set
    python -m v2.agent_v2.run_benchmark --holdout       # the 15 held-out questions
    python -m v2.agent_v2.run_benchmark --modes v2_llm --repeat 3   # needs AGENT_LLM_API_KEY (+ AGENT_LLM_BASE_URL/MODEL)
    python -m v2.agent_v2.run_benchmark --modes v2_llm --simulate 0.2 --repeat 3   # harness dry run, no key
    python -m v2.agent_v2.run_benchmark --markdown report.md   # Markdown tables
    python -m v2.agent_v2.run_benchmark --fixtures engine   # engine-shaped research/market envelopes
    python -m v2.agent_v2.run_benchmark --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from v2.agent_v2.eval.benchmark import MODES, gap_summary, render, run_benchmark, to_json, to_markdown
from v2.agent_v2.eval.benchmark_fixtures import FIXTURE_MODES
from v2.agent_v2.eval.benchmark_cases import DEV_CASES, HOLDOUT_CASES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modes", default="v2_rules", help=f"comma-separated subset of {', '.join(MODES)}")
    parser.add_argument("--holdout", action="store_true", help="run the held-out set instead of the development set")
    parser.add_argument("--repeat", type=int, default=1, help="repeats per case for v2_llm (deterministic modes run once)")
    parser.add_argument("--json", dest="json_path", default="", help="write per-case scores to this file")
    parser.add_argument("--fixtures", default="v1", choices=FIXTURE_MODES, help="v1: V1 cards with V1 fact keys; engine: engine-shaped research/market envelopes")
    parser.add_argument("--no-failures", action="store_true", help="omit the per-mode failure list")
    parser.add_argument("--simulate", type=float, default=None, metavar="NOISE", help="run v2_llm with the simulated model (noise = share of drafts with an invented figure); a harness dry run, not a model score")
    parser.add_argument("--seed", type=int, default=0, help="seed for the simulated model")
    parser.add_argument("--markdown", default="", help="append a Markdown report to this file")
    parser.add_argument("--workers", type=int, default=1, help="parallel runs for v2_llm (model-bound); deterministic modes stay sequential")
    parser.add_argument("--progress", action="store_true", help="print one line per v2_llm run to stderr")
    args = parser.parse_args(argv)

    modes = tuple(mode.strip() for mode in args.modes.split(",") if mode.strip())
    unknown = [mode for mode in modes if mode not in MODES]
    if unknown:
        print(f"unknown mode(s): {', '.join(unknown)}", file=sys.stderr)
        return 2
    cases = HOLDOUT_CASES if args.holdout else DEV_CASES
    print(f"Agent V1→V2 benchmark · {'holdout' if args.holdout else 'dev'} set · {len(cases)} cases · modes={', '.join(modes)} · fixtures={args.fixtures}")
    if args.fixtures == "engine":
        from v2.agent_v2.eval.recorded import RecordedStore

        recorded = RecordedStore().summary()
        print("recorded envelopes: " + (", ".join(f"{name}={count}" for name, count in recorded.items()) if recorded else "none (offline synthesis only)"))
    gaps = gap_summary(cases)
    if gaps:
        print("capability gaps: " + "; ".join(f"{tool} → {len(ids)} case(s)" for tool, ids in gaps.items()))
    print()
    llm_factory = None
    if args.simulate is not None:
        from v2.agent_v2.eval.simulated_llm import SimulatedLLMFactory

        llm_factory = SimulatedLLMFactory(noise=args.simulate, seed=args.seed)
        print(f"v2_llm uses the SIMULATED model (noise={args.simulate}); numbers validate the harness, not a model")
    elif "v2_llm" in modes and not any(os.environ.get(name) for name in ("AGENT_LLM_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY")):
        print("v2_llm needs AGENT_LLM_API_KEY (or DEEPSEEK_API_KEY / OPENAI_API_KEY); use --simulate for a harness dry run", file=sys.stderr)
        return 2
    reports = run_benchmark(modes, holdout=args.holdout, repeat=args.repeat, fixtures=args.fixtures, llm_factory=llm_factory, workers=max(1, args.workers), progress=args.progress)
    print(render(reports, failures=not args.no_failures))
    if args.markdown:
        label = f"{'留出集' if args.holdout else '开发集'} · {len(cases)} 例 · fixtures={args.fixtures}" + (f" · 重复 {args.repeat}" if args.repeat > 1 else "")
        note = "v2_llm 由模拟模型驱动，仅验证评测框架。" if args.simulate is not None else ""
        with open(args.markdown, "a", encoding="utf-8") as handle:
            handle.write(to_markdown(reports, title=label, note=note))
        print(f"appended Markdown to {args.markdown}")
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as handle:
            json.dump(to_json(reports), handle, ensure_ascii=False, indent=2)
        print(f"\nwrote {args.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
