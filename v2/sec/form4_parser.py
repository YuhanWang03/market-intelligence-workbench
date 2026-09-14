"""Form 4 (insider transaction) parsing.

Stage 0 bug catch (Check 4): edgartools' ``Form4.to_dataframe()`` returns
the transaction-code column as ``"Code"`` (PascalCase, from the SDK's
XML-extraction path), **not** ``"transaction_code"`` (the underlying
dataclass field). Using the wrong name silently produces an empty
DataFrame and 100% of Form 4s appear to "have no insider activity" —
critical bug.

Full column list edgartools emits:
    Security, Date, Shares, Remaining, Price, AcquiredDisposed,
    DirectIndirect, NatureOfOwnership, form, Code, EquitySwap, footnotes

Same module also handles:
- 10b5-1 plan footnote detection (regex against ``footnotes`` free text)
- transaction_usd computation (shares × price, with None tolerance)
- per-row defensive type coercion (SDK occasionally returns strings)
"""

from __future__ import annotations

import logging
import re
from typing import Any

from v2.sec.insider_role import lookup_insider_role
from v2.sec.models import Form4Transaction, SecFiling

logger = logging.getLogger(__name__)


# Stage 0 Check 4 — DO NOT change to "transaction_code", that's the
# dataclass field name not the DataFrame column name.
_DF_CODE_COL = "Code"
_DF_SHARES_COL = "Shares"
_DF_PRICE_COL = "Price"
_DF_DATE_COL = "Date"
_DF_DIRECT_COL = "DirectIndirect"
_DF_FOOTNOTES_COL = "footnotes"


# Match common Rule 10b5-1 phrasings in footnotes free text.
# Real footnotes use varied capitalization and sometimes drop the rule
# number — regex is permissive.
_10B5_1_RE = re.compile(
    r"(?:rule\s+)?10b5-1|trading\s+plan|pre-?arranged\s+plan",
    re.IGNORECASE,
)


def _detect_10b5_1(footnotes: Any) -> bool:
    """True if footnotes mention Rule 10b5-1 / trading plan."""
    if not footnotes:
        return False
    return bool(_10B5_1_RE.search(str(footnotes)))


def _coerce_float(v: Any) -> float | None:
    """SDK occasionally returns strings, numpy floats, or NaN. Defensive."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    # pandas NaN check
    if f != f:        # NaN != NaN
        return None
    return f


def parse_form4_filing(
    edgar_filing: Any,
    sec_filing: SecFiling,
) -> list[Form4Transaction]:
    """Convert an edgartools Form 4 filing into ``Form4Transaction`` rows.

    One Form 4 filing typically contains 1-5 transactions (multi-share-
    class executives report all classes in one filing). Returns one
    ``Form4Transaction`` per non-derivative row; derivative-only rows
    are skipped (covered separately if Phase 3.5 adds derivatives).

    Args:
        edgar_filing: edgartools ``Filing`` object (form="4"). Its
            ``.obj()`` returns ``Form4`` with ``insider_name`` and
            ``reporting_owners`` attributes.
        sec_filing: pre-built metadata (caller passes it; we don't
            re-extract).

    Returns:
        List of transactions. Empty on parse failure (warning logged).
    """
    try:
        form4 = edgar_filing.obj()
    except Exception as exc:
        logger.warning(
            "Form 4 .obj() failed for %s acc=%s: %s",
            sec_filing.ticker, sec_filing.accession_number, exc,
        )
        return []

    insider_name = str(getattr(form4, "insider_name", "") or "")
    insider_role = lookup_insider_role(form4)

    try:
        df = form4.to_dataframe()
    except Exception as exc:
        logger.warning(
            "Form 4 to_dataframe() failed for %s acc=%s: %s",
            sec_filing.ticker, sec_filing.accession_number, exc,
        )
        return []

    if df is None or len(df) == 0:
        return []

    # Verify the critical column exists — defensive guard against
    # SDK changes. Falling through silently would re-introduce the
    # Stage 0 bug.
    if _DF_CODE_COL not in df.columns:
        logger.warning(
            "Form 4 DataFrame missing %r column for %s — SDK output shape "
            "may have changed. Available columns: %s",
            _DF_CODE_COL, sec_filing.ticker, list(df.columns),
        )
        return []

    transactions: list[Form4Transaction] = []
    for _, row in df.iterrows():
        code = str(row.get(_DF_CODE_COL, "") or "").strip().upper()
        if not code:
            continue

        shares = _coerce_float(row.get(_DF_SHARES_COL))
        price = _coerce_float(row.get(_DF_PRICE_COL))
        if shares is None:
            continue  # zero-share rows are filings noise, skip

        usd: float | None = None
        if price is not None and price > 0:
            usd = shares * price

        # ``DirectIndirect`` returns "D" or "I"; missing → assume direct.
        direct = str(row.get(_DF_DIRECT_COL, "D") or "D").strip().upper()
        if direct not in {"D", "I"}:
            direct = "D"

        # Footnotes is a free-text column containing 10b5-1 markers.
        footnotes_raw = row.get(_DF_FOOTNOTES_COL)
        is_10b5_1 = _detect_10b5_1(footnotes_raw)

        transaction_date = str(row.get(_DF_DATE_COL, "") or "") or sec_filing.filing_date

        transactions.append(Form4Transaction(
            filing=sec_filing,
            insider_name=insider_name,
            insider_role=insider_role,
            transaction_code=code,
            transaction_date=transaction_date,
            shares=shares,
            price=price,
            transaction_usd=usd,
            is_10b5_1=is_10b5_1,
            direct_indirect=direct,
        ))

    return transactions
