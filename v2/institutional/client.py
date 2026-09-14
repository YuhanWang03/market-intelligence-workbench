"""edgartools wrapper for fetching 13F-HR filings from SEC EDGAR.

SEC requires a real-looking User-Agent. edgartools handles that via
set_identity(), which we call once per process from EDGAR_IDENTITY env var
(falls back to a default).
"""

from __future__ import annotations

import logging
import os

from edgar import Company, set_identity

from v2.institutional.models import Filing, Position
from v2.observability import emit

logger = logging.getLogger(__name__)

_IDENTITY_SET = False


def _ensure_identity() -> None:
    """Set SEC identity once per process — required by SEC's fair-use policy."""
    global _IDENTITY_SET
    if _IDENTITY_SET:
        return
    identity = os.environ.get(
        "EDGAR_IDENTITY",
        "Market Intelligence Workbench contact@example.com",
    )
    set_identity(identity)
    _IDENTITY_SET = True


def fetch_recent_13f(
    cik: str,
    manager_name: str,
    n_filings: int = 2,
) -> list[tuple[Filing, list[Position]]]:
    """Fetch the *n_filings* most recent 13F-HR filings for one manager.

    Returns list of (Filing metadata, positions). Empty list on any error.
    """
    _ensure_identity()

    try:
        co = Company(cik)
        filings = co.get_filings(form="13F-HR")
    except Exception as exc:
        logger.warning("EDGAR fetch failed for %s (CIK %s): %s",
                       manager_name, cik, exc)
        return []

    if filings is None:
        return []

    # edgartools filings object is iterable but Pandas-backed; coerce to list
    try:
        filings_list = list(filings)[:n_filings]
    except Exception as exc:
        logger.warning("EDGAR filing iteration failed for %s: %s",
                       manager_name, exc)
        return []

    results: list[tuple[Filing, list[Position]]] = []
    for f in filings_list:
        parsed = _parse_one(f, cik, manager_name)
        if parsed is not None:
            results.append(parsed)

    return results


def _parse_one(
    filing,
    cik: str,
    manager_name: str,
) -> tuple[Filing, list[Position]] | None:
    """Parse one edgartools Filing into our Filing + Positions."""
    try:
        accession = str(
            getattr(filing, "accession_no", None)
            or getattr(filing, "accession_number", None)
            or ""
        )
        filing_date = str(getattr(filing, "filing_date", ""))
        period = str(getattr(filing, "period_of_report", "") or "")
    except Exception as exc:
        logger.warning("Filing metadata read failed: %s", exc)
        return None

    if not accession or not period:
        return None

    quarter = _quarter_from_period(period)

    try:
        thirteenF = filing.obj()
        infotable = getattr(thirteenF, "infotable", None)
        if infotable is None:
            infotable = getattr(thirteenF, "holdings", None)
    except Exception as exc:
        logger.warning("13F parse failed (%s %s): %s",
                       manager_name, accession, exc)
        return None

    if infotable is None or len(infotable) == 0:
        return None

    raw_positions: list[Position] = []
    for _, row in infotable.iterrows():
        pos = _row_to_position(row, cik, accession, quarter)
        if pos is None:
            continue
        raw_positions.append(pos)

    # Aggregate by CUSIP. 13F-HR can list the same security multiple times
    # when a filer reports it across subsidiaries (Berkshire's AAPL spans
    # BHRG / GEICO / Nat'l Indemnity → 3 rows of one position).
    # Our DB PK is (accession, cusip) so unaggregated rows would also collide
    # on INSERT OR REPLACE — losing all but the last. Aggregating once at
    # source fixes both the display AND the persistence layer.
    positions = _aggregate_by_cusip(raw_positions)
    emit(
        "transform",
        op="cusip_aggregate",
        before=len(raw_positions),
        after=len(positions),
        manager=manager_name,
        accession=accession[-12:],
    )
    total_value = sum(p.market_value for p in positions)

    filing_obj = Filing(
        cik=cik,
        manager_name=manager_name,
        accession=accession,
        quarter=quarter,
        filing_date=filing_date,
        period_of_report=period,
        portfolio_value=total_value,
        n_positions=len(positions),
    )
    return (filing_obj, positions)


def _aggregate_by_cusip(positions: list[Position]) -> list[Position]:
    """Collapse same-CUSIP rows into one — sums shares + market_value."""
    merged: dict[str, Position] = {}
    for p in positions:
        existing = merged.get(p.cusip)
        if existing is None:
            merged[p.cusip] = p
            continue
        merged[p.cusip] = Position(
            cik=existing.cik,
            accession=existing.accession,
            quarter=existing.quarter,
            cusip=existing.cusip,
            ticker=existing.ticker or p.ticker,
            issuer_name=existing.issuer_name or p.issuer_name,
            shares=existing.shares + p.shares,
            market_value=existing.market_value + p.market_value,
        )
    return list(merged.values())


def _row_to_position(row, cik: str, accession: str, quarter: str) -> Position | None:
    """Convert one infotable row to a Position. Tolerates column name variants."""
    issuer = (
        _try_str(row, "Issuer")
        or _try_str(row, "NameOfIssuer")
        or _try_str(row, "name_of_issuer")
        or "Unknown"
    )
    cusip = (
        _try_str(row, "Cusip")
        or _try_str(row, "CUSIP")
        or _try_str(row, "cusip")
        or ""
    )
    shares_raw = (
        _try_num(row, "SharesPrnAmount")   # edgartools 5.x canonical name
        or _try_num(row, "Shares")
        or _try_num(row, "ShrsOrPrnAmt")
        or 0
    )
    value_raw = (
        _try_num(row, "Value")
        or _try_num(row, "value")
        or 0
    )
    ticker = _try_str(row, "Ticker") or _try_str(row, "ticker") or None

    if not cusip or shares_raw <= 0:
        return None

    # edgartools 5.x already converts 13F's "thousands of dollars" reporting
    # convention to plain dollars. Don't re-scale.
    market_value = float(value_raw)

    return Position(
        cik=cik,
        accession=accession,
        quarter=quarter,
        cusip=cusip,
        ticker=ticker,
        issuer_name=issuer[:200],
        shares=int(shares_raw),
        market_value=market_value,
    )


def _try_str(row, key) -> str:
    try:
        v = row.get(key) if hasattr(row, "get") else row[key]
        return str(v).strip() if v is not None else ""
    except (KeyError, AttributeError, TypeError):
        return ""


def _try_num(row, key) -> float:
    try:
        v = row.get(key) if hasattr(row, "get") else row[key]
        return float(v) if v is not None else 0.0
    except (KeyError, AttributeError, TypeError, ValueError):
        return 0.0


def _quarter_from_period(period: str) -> str:
    """Convert YYYY-MM-DD to YYYY-QN (e.g. '2026-03-31' -> '2026-Q1')."""
    if not period or len(period) < 10:
        return "Unknown"
    try:
        year = period[:4]
        month = int(period[5:7])
        q = (month - 1) // 3 + 1
        return f"{year}-Q{q}"
    except (ValueError, IndexError):
        return "Unknown"
