"""CLI: ``python -m v2.personas AAPL MSFT [--as-of 2026-06-30] [--personas a,b] [--json] [--narrate] [--demo]``."""

from __future__ import annotations

import argparse
import json
import sys

from v2.personas.committee import CommitteeResult, run_committee
from v2.personas.registry import PERSONAS, get_persona


def _print_table(result: CommitteeResult) -> None:
    keys = result.personas
    width = max(len(k) for k in keys) + 2
    print(f"as of {result.as_of} · {len(result.verdicts)} tickers · {len(keys)} personas · {result.elapsed_s:.1f}s")
    if result.errors:
        for t, e in result.errors.items():
            print(f"  ! {t}: {e}")
    print()
    header = "persona".ljust(width) + "".join(v.ticker.rjust(14) for v in result.verdicts)
    print(header)
    grid = result.matrix()
    glyph = {"bullish": "▲", "bearish": "▼", "neutral": "·"}
    for key in keys:
        row = key.ljust(width)
        for v in result.verdicts:
            s = grid.get(key, {}).get(v.ticker)
            cell = "  abstain" if s is None or s.abstained else f"{glyph[s.signal]} {s.signal[:4]} {s.confidence:>3}"
            row += cell.rjust(14)
        print(row)
    print()
    print("consensus".ljust(width) + "".join(f"{v.consensus:+.2f}".rjust(14) for v in result.verdicts))
    print("votes ▲/▼/·".ljust(width) + "".join(f"{v.bullish}/{v.bearish}/{v.neutral}".rjust(14) for v in result.verdicts))
    print("agreement".ljust(width) + "".join(f"{v.agreement:.0%}".rjust(14) for v in result.verdicts))
    print()
    for v in result.verdicts:
        print(f"#{v.rank} {v.ticker} {v.stance} ({v.consensus:+.2f})")
        for s in v.signals:
            line = s.narrative or s.reasoning
            print(f"   {s.persona:<22} {s.signal:<8} {s.confidence:>3}  {line[:150]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m v2.personas", description="Ask the simulated investors about some tickers.")
    parser.add_argument("tickers", nargs="*", help="ticker symbols")
    parser.add_argument("--as-of", dest="as_of", default=None, help="analysis date (YYYY-MM-DD), default today")
    parser.add_argument("--personas", default=None, help="comma-separated persona keys (default: all)")
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    parser.add_argument("--narrate", action="store_true", help="add LLM-written explanations (needs an LLM key)")
    parser.add_argument("--lang", default="zh", choices=("zh", "en"))
    parser.add_argument("--demo", action="store_true", help="run on the built-in synthetic snapshots, no network")
    parser.add_argument("--list", action="store_true", help="list personas and exit")
    args = parser.parse_args(argv)

    if args.list:
        for key in PERSONAS:
            p = get_persona(key)
            print(f"{key:<22} {p.name:<22} {p.name_zh:<10} {p.style}")
        return 0

    keys = [k.strip() for k in args.personas.split(",")] if args.personas else None
    if args.demo:
        from v2.personas.fixtures import distressed_snapshot, quality_snapshot
        snaps = {s.ticker: s for s in (quality_snapshot(), distressed_snapshot())}
        result = run_committee(list(snaps), personas=keys, snapshots=snaps)
    else:
        if not args.tickers:
            parser.error("give at least one ticker, or --demo")
        result = run_committee(args.tickers, personas=keys, as_of=args.as_of, progress=lambda t, s: print(f"  {t}: {s}", file=sys.stderr))

    if args.narrate:
        from v2.personas.narrate import narrate_many
        for v in result.verdicts:
            narrate_many(v.signals, language=args.lang)

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        _print_table(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
