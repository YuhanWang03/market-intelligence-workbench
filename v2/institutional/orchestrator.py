"""End-to-end orchestration for institutional 13F tracking.

For each manager:
1. Fetch the 2 most recent 13F-HR filings from EDGAR.
2. If the latest is already in our DB, skip (no new info).
3. Otherwise, compare against the previous quarter (from this fetch OR DB).
4. Persist both filings.
5. Detect significant changes.
6. LLM-interpret the changes.
"""

from __future__ import annotations

import logging
from datetime import date

from v2.institutional.client import fetch_recent_13f
from v2.institutional.detector import detect_changes
from v2.institutional.managers import MANAGERS
from v2.institutional.models import Filing, InstitutionalReport, PositionChange
from v2.institutional.summarizer import interpret_changes
from v2.institutional.tracker import (
    get_db,
    get_positions_for,
    get_previous_filing,
    has_filing,
    save_filing,
)

logger = logging.getLogger(__name__)


def run_institutional_pipeline(
    universe: set[str] | None = None,
    managers: list[tuple[str, str]] | None = None,
) -> InstitutionalReport:
    """Run one full pass over the manager watchlist.

    *managers* (optional) — override the default MANAGERS list. The bot's
    /13f command uses this to process a single manager on demand.
    """
    universe = universe or set()
    today = date.today().isoformat()
    target_managers = managers if managers is not None else MANAGERS

    new_filings: list[Filing] = []
    all_changes: list[PositionChange] = []
    api_calls = 0
    llm_tokens = 0

    with get_db() as conn:
        for cik, manager_name in target_managers:
            logger.info("Checking %s (CIK %s)...", manager_name, cik)
            api_calls += 1

            recent = fetch_recent_13f(cik, manager_name, n_filings=2)
            if not recent:
                logger.info("  No 13F-HR filings found")
                continue

            current_filing, current_positions = recent[0]

            # Already processed this filing? Skip.
            if has_filing(conn, cik, current_filing.accession):
                logger.info("  %s %s already in DB", manager_name,
                            current_filing.quarter)
                continue

            # Find the previous quarter's positions — prefer fresh fetch,
            # fall back to DB if we only got one filing.
            prev_positions_dicts: list[dict] = []
            prev_total = 0.0

            if len(recent) >= 2:
                prev_filing, prev_positions = recent[1]
                prev_positions_dicts = [_pos_to_dict(p) for p in prev_positions]
                prev_total = prev_filing.portfolio_value
                # Persist previous if not already there
                if not has_filing(conn, cik, prev_filing.accession):
                    save_filing(conn, prev_filing, prev_positions)
            else:
                prev = get_previous_filing(conn, cik, current_filing.accession)
                if prev:
                    prev_positions_dicts = get_positions_for(conn, prev["accession"])
                    prev_total = float(prev["portfolio_value"])

            # Always save the current filing
            save_filing(conn, current_filing, current_positions)
            new_filings.append(current_filing)

            if not prev_positions_dicts:
                logger.info("  No prior filing to compare against — recorded only")
                continue

            cur_dicts = [_pos_to_dict(p) for p in current_positions]
            changes = detect_changes(
                cik=cik,
                manager_name=manager_name,
                quarter=current_filing.quarter,
                current_positions=cur_dicts,
                prev_positions=prev_positions_dicts,
                current_total=current_filing.portfolio_value,
                prev_total=prev_total,
            )

            # Flag tickers in user's monitored universe
            for c in changes:
                if c.ticker and c.ticker in universe:
                    c.in_universe = True

            logger.info("  %d significant changes", len(changes))

            if changes:
                # Cap LLM input — DeepSeek's output token ceiling truncates
                # JSON if we send too many at once. Top 20 by impact is plenty.
                top_for_llm = changes[:20]
                interpretations, tokens = interpret_changes(
                    manager_name, top_for_llm,
                )
                llm_tokens += tokens
                for c in top_for_llm:
                    key = c.ticker or c.cusip
                    if key in interpretations:
                        c.interpretation = interpretations[key]

            all_changes.extend(changes)

    return InstitutionalReport(
        date=today,
        new_filings=new_filings,
        changes=all_changes,
        api_calls=api_calls,
        llm_tokens=llm_tokens,
    )


def _pos_to_dict(p) -> dict:
    """Coerce a Position (Pydantic) OR an existing dict to the dict shape
    detector.detect_changes() expects."""
    if isinstance(p, dict):
        return p
    return {
        "cusip": p.cusip,
        "ticker": p.ticker,
        "issuer_name": p.issuer_name,
        "shares": p.shares,
        "market_value": p.market_value,
    }
