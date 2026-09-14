"""Record live capability envelopes for the benchmark's ``engine`` fixture layer.

    python -m v2.agent_v2.record_fixtures --live                    # all benchmark tickers, all focuses
    python -m v2.agent_v2.record_fixtures --live --tickers NVDA,AMD --focuses overview,earnings
    python -m v2.agent_v2.record_fixtures --synthetic               # write the offline synthesis to disk

``--live`` runs the real research and market adapters, which need the
production ``v2.data`` package and provider keys; it writes one JSON file per
capability under ``v2/agent_v2/eval/recorded/``.  The benchmark replays a
recording whenever one exists for a key and synthesises the rest, so partial
recordings are useful.  ``--synthetic`` writes the offline envelopes instead,
for inspection or to pin them.
"""

from __future__ import annotations

import argparse
import sys
import time

from v2.agent_v2.eval.recorded import RecordedStore

BENCHMARK_TICKERS = ("CRWD", "NVDA", "SMCI", "MSFT", "AMD", "TSLA", "AAPL", "GOOGL", "ARM", "PLTR", "AVGO")
FOCUSES = ("overview", "fundamentals", "valuation", "earnings", "market", "ownership", "catalysts", "filings", "supply_chain", "risk", "full")
MARKET = ("market.performance", "market.explain_move")


def _live_registry():
    from v2.agent_v2.catalog import default_catalog
    from v2.agent_v2.execution import CapabilityRegistry
    from v2.agent_v2.adapters.market import register_market_capabilities
    from v2.agent_v2.adapters.research import register_research_capabilities

    registry = CapabilityRegistry(default_catalog())
    register_research_capabilities(registry)
    register_market_capabilities(registry)
    return registry


def record(store: RecordedStore, tickers: tuple[str, ...], focuses: tuple[str, ...], *, live: bool, market: bool = True) -> int:
    from v2.agent_v2.execution import ExecutionContext
    from v2.agent_v2.models import BudgetClass, NormalizedRequest, PlanTask

    if live:
        registry = _live_registry()
    else:
        from v2.agent_v2.eval.engine_fixtures import synthesize_market_envelope, synthesize_research_envelope

    context = ExecutionContext("record", NormalizedRequest("record", "record"), BudgetClass.DEEP)
    written = 0
    for ticker in tickers:
        for focus in focuses:
            started = time.time()
            if live:
                envelope = registry.execute(PlanTask("r", "research.stock", {"ticker": ticker, "focus": focus}), context)
            else:
                envelope = synthesize_research_envelope(ticker, focus)
            if not envelope.ok:
                print(f"  research.stock {ticker}:{focus} -> {envelope.status.value}: {'; '.join(envelope.errors)[:120]}")
                continue
            store.save("research.stock", f"{ticker}:{focus}", envelope)
            written += 1
            print(f"  research.stock {ticker}:{focus} -> {envelope.status.value}, {len(envelope.evidence)} evidence, {int((time.time() - started) * 1000)} ms")
        if not market:
            continue
        for capability in MARKET:
            if live:
                envelope = registry.execute(PlanTask("m", capability, {"ticker": ticker}), context)
            else:
                envelope = synthesize_market_envelope(capability, ticker)
            if not envelope.ok:
                print(f"  {capability} {ticker} -> {envelope.status.value}: {'; '.join(envelope.errors)[:120]}")
                continue
            store.save(capability, ticker, envelope)
            written += 1
            print(f"  {capability} {ticker} -> {envelope.status.value}, {len(envelope.evidence)} evidence")
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--live", action="store_true", help="run the real adapters (needs v2.data and provider keys)")
    group.add_argument("--synthetic", action="store_true", help="write the offline synthesis to disk")
    parser.add_argument("--tickers", default=",".join(BENCHMARK_TICKERS))
    parser.add_argument("--focuses", default=",".join(FOCUSES))
    parser.add_argument("--no-market", action="store_true", help="skip market.performance and market.explain_move")
    parser.add_argument("--dir", default="", help="output directory (default: v2/agent_v2/eval/recorded)")
    args = parser.parse_args(argv)

    if args.live:
        try:
            import v2.data as data

            if not hasattr(data, "CachedFDClient"):
                raise ImportError("v2.data is a namespace without the production clients")
        except ImportError as exc:
            print(f"--live needs the production v2.data package: {exc}", file=sys.stderr)
            return 2
    store = RecordedStore(args.dir or None)
    tickers = tuple(value.strip().upper() for value in args.tickers.split(",") if value.strip())
    focuses = tuple(value.strip() for value in args.focuses.split(",") if value.strip())
    print(f"recording {'live' if args.live else 'synthetic'} envelopes for {len(tickers)} ticker(s) × {len(focuses)} focus(es) into {store.directory}")
    written = record(store, tickers, focuses, live=args.live, market=not args.no_market)
    print(f"wrote {written} envelope(s): " + ", ".join(f"{name}={count}" for name, count in store.summary().items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
