"""SEC scan orchestrator — fans out across (8-K + Form 4) × tickers.

Contract per Stage 0:
- Single entry point: ``run_sec_scan(tickers, today_iso, ...)``.
- Always returns ``SecScanResult`` — sub-failures collected in warnings,
  never raised. Cron path needs guaranteed return.
- Test seams for ``edgar_client`` and ``llm_extractor`` so the sandbox
  smoke can exercise the full pipeline without live SEC/DeepSeek.

The orchestration sequence per ticker:

  1. Fetch 8-K filings filed today
  2. Parse items → build EightKEvent per filing
  3. For each 5.02 item, run LLM extractor → escalate priority if senior exec
  4. Fetch Form 4 filings filed today
  5. Parse transactions → split signal (P/S) vs noise (A/M/F/G/C)
  6. Aggregate noise codes into form4_noise_summary per ticker
  7. Across all signal transactions, detect same-day clusters

Step ordering matters: 8-K first because the universe is smaller (a
ticker filing 0 8-Ks per day is typical, hits the early-return), then
Form 4 (the heavier per-ticker query). Same throttle path so EDGAR
rate-limit headroom holds.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Callable

from v2.sec import client as edgar_client_mod
from v2.sec import eight_k_parser, form4_parser, ner_5_02
from v2.sec.cluster import find_clusters
from v2.sec.eight_k_parser import get_item_text
from v2.sec.models import (
    EightKEvent,
    Form4Transaction,
    SecFiling,
    SecScanResult,
)

logger = logging.getLogger(__name__)


# Senior-exec roles that escalate 5.02 from P1 to P0 (per Stage 0 spec).
_SENIOR_5_02_ROLES = {"CEO", "CFO", "President", "Chairman", "COO"}


def run_sec_scan(
    tickers: list[str],
    today_iso: str,
    *,
    edgar_client: Callable[..., list] | None = None,
    llm_extractor: Callable[[str], dict] | None = None,
) -> SecScanResult:
    """Run a full SEC scan for ``tickers`` on the given trading day.

    Args:
        tickers: universe to scan (held + watchlist union, deduped by
            caller — pipeline doesn't dedup).
        today_iso: ISO date — the cron's notion of "today" in ET. Both
            8-K and Form 4 are filtered to filings dated == today_iso.
        edgar_client: test seam. Defaults to ``v2.sec.client.get_recent_filings``.
            Tests pass a function with signature
            ``(ticker, form, since_iso, until_iso) -> list``.
        llm_extractor: test seam for the 5.02 NER. Defaults to
            ``ner_5_02.extract_5_02``. Tests pass a callable that
            returns the canned ``{"departures","appointments","has_senior_exec"}``.

    Returns:
        ``SecScanResult`` with eight_k_events, form4_signal_transactions
        (P/S only), form4_clusters (same-day ≥3), form4_noise_summary
        (per-ticker counts of A/M/F/G/C codes), warnings list.

    Per Stage 0 decision #5: empty results (zero 8-K, zero Form 4 today)
    do NOT log warnings. Quiet days are common — IVV/AAPL/ARM/MU/PLTR/QCOM
    had 0 8-Ks in the entire Stage 0 30-day Task 4 sample. A clean-day
    log line at the end of the cron (info-level) is enough.
    """
    if edgar_client is None:
        edgar_client = edgar_client_mod.get_recent_filings
    if llm_extractor is None:
        llm_extractor = ner_5_02.extract_5_02

    result = SecScanResult()
    noise_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    all_signal_transactions: list[Form4Transaction] = []

    for ticker in tickers:
        # ---- 8-K ----
        try:
            eight_k_filings = edgar_client(ticker, "8-K", today_iso, today_iso)
        except Exception as exc:
            msg = f"{ticker} 8-K fetch failed: {exc}"
            logger.warning(msg)
            result.warnings.append(msg)
            eight_k_filings = []

        for edgar_filing in eight_k_filings:
            sec_filing = _build_sec_filing(edgar_filing, ticker, form="8-K")
            if sec_filing is None:
                continue
            event = eight_k_parser.parse_eight_k_filing(edgar_filing, sec_filing)
            if event is None:
                continue
            # Stage 2 pickup: extract the 5.02 item's raw text from the
            # parsed EightK obj and pass to the LLM extractor. The
            # extractor falls back to safe defaults on empty text.
            try:
                eight_k_obj = edgar_filing.obj()
            except Exception:
                eight_k_obj = None
            _apply_5_02_extraction(event, eight_k_obj, llm_extractor, result.warnings)
            result.eight_k_events.append(event)

        # ---- Form 4 ----
        try:
            form4_filings = edgar_client(ticker, "4", today_iso, today_iso)
        except Exception as exc:
            msg = f"{ticker} Form 4 fetch failed: {exc}"
            logger.warning(msg)
            result.warnings.append(msg)
            form4_filings = []

        for edgar_filing in form4_filings:
            sec_filing = _build_sec_filing(edgar_filing, ticker, form="4")
            if sec_filing is None:
                continue
            txs = form4_parser.parse_form4_filing(edgar_filing, sec_filing)
            for tx in txs:
                if tx.is_signal:
                    all_signal_transactions.append(tx)
                elif tx.is_noise:
                    noise_counts[ticker][tx.transaction_code] += 1
                # Codes outside both sets (rare — V/D/X/K) silently skip.
                # Phase 3.5 can add them to noise_summary if needed.

    result.form4_signal_transactions = all_signal_transactions
    result.form4_clusters = find_clusters(all_signal_transactions)
    # Convert defaultdict → regular dict for clean serialization
    result.form4_noise_summary = {
        ticker: dict(codes) for ticker, codes in noise_counts.items()
    }

    return result


def _build_sec_filing(
    edgar_filing: Any,
    ticker: str,
    *,
    form: str,
) -> SecFiling | None:
    """Extract metadata from an edgartools Filing into our SecFiling.

    Returns None on missing accession_number (filing is unusable
    downstream — accession is the dedup key).
    """
    try:
        accession = str(
            getattr(edgar_filing, "accession_number", None)
            or getattr(edgar_filing, "accession_no", None)
            or ""
        ).strip()
        filing_date = str(getattr(edgar_filing, "filing_date", "") or "").strip()
        cik = str(getattr(edgar_filing, "cik", "") or "").strip()
        # Form field on filing may already encode /A suffix; trust SDK
        actual_form = str(getattr(edgar_filing, "form", form) or form).strip()
    except Exception as exc:
        logger.warning("Filing metadata read failed for %s: %s", ticker, exc)
        return None

    if not accession:
        return None

    return SecFiling(
        ticker=ticker,
        cik=cik,
        form=actual_form,
        filing_date=filing_date,
        accession_number=accession,
        is_amendment=actual_form.endswith("/A"),
    )


def _apply_5_02_extraction(
    event: EightKEvent,
    eight_k_obj,
    llm_extractor: Callable[[str], dict],
    warnings: list[str],
) -> None:
    """Find the 5.02 item (if any), run LLM extraction, escalate priority.

    Mutates ``event`` in place. Adds a warning if the extractor degrades
    to the fallback (so the dashboard trace shows the degraded run).

    Args:
        event: the EightKEvent to mutate in place.
        eight_k_obj: the underlying edgartools ``EightK`` instance used
            to extract the raw 5.02 item text. Can be ``None`` on
            upstream parse failure; the LLM fallback path handles that.
        llm_extractor: callable taking item-text string and returning
            the structured extraction dict.
        warnings: list to append failure messages to.
    """
    for item in event.items:
        if item.code != "5.02":
            continue

        # Stage 2 pickup: read the raw 5.02 section text from the
        # underlying edgartools obj. Empty text triggers ner_5_02's
        # conservative fallback (has_senior_exec=True → P0 escalation).
        if eight_k_obj is not None:
            item_text = get_item_text(eight_k_obj, "5.02")
        else:
            item_text = ""

        try:
            extracted = llm_extractor(item_text)
        except Exception as exc:
            warnings.append(f"5.02 LLM extraction failed for {event.filing.ticker}: {exc}")
            extracted = {
                "departures": [],
                "appointments": [],
                "has_senior_exec": True,    # conservative
            }

        # Replace the item with a copy carrying the extracted_meta and
        # escalated priority. EightKItem is frozen, so we rebuild.
        new_tier = "P0" if extracted.get("has_senior_exec") else item.priority_tier
        new_item = item.__class__(
            code=item.code,
            priority_tier=new_tier,
            description=item.description,
            extracted_meta=extracted,
        )
        # Replace in-place (event.items is a regular list)
        idx = event.items.index(item)
        event.items[idx] = new_item
        return    # only one 5.02 per filing — done
