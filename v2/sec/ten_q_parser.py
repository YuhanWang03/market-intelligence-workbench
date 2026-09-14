"""10-Q parser — Phase 3.5.

Extracts MD&A (Part I, Item 2) + Risk Factors (Part II, Item 1A) from
an edgartools ``TenQ`` filing object and computes a delta versus the
prior quarter's filing. Used by ⑧ Earnings Summaries to add a
``📋 10-Q MD&A 关键变化`` block to the post-release card.

Modeled after :mod:`v2.macro.fomc_parser` (one filing → one
diff-aware result) NOT :mod:`v2.sec.eight_k_parser` (multi-item
per filing). 10-Q is a single comprehensive document with fixed
sections; the diff is the interesting signal, not the per-section
list.

edgartools API (verified Stage 0, edgartools 5.35.1):

- ``TenQ.get_item_with_part(part, item, markdown=True) -> Optional[str]``
  Returns the text content of a specific item. ``part`` is "Part I" or
  "Part II"; ``item`` is "Item 2" / "Item 1A" etc.
- ``TenQ.items -> List[str]`` returns part-qualified item names like
  ``"Part I, Item 2"`` — we use this only to gate availability.

No ``.text`` / ``.mdna`` / ``.risk_factors`` attrs exist on ``TenQ``
(those are 8-K-style; don't bother trying).

Conservative behavior:
- Any extraction failure → return None / empty fields. Never raise.
- Diff with no prior 10-Q → ``TenQDelta`` with empty added paragraphs
  + no flags set. Section renders nothing meaningful but doesn't crash.
- Going concern + material weakness keywords gate the conservative
  P0 escalation in :mod:`v2.reporting.priority`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Item path constants (10-Q convention)
# ---------------------------------------------------------------------------

# Part I Item 2 = Management's Discussion and Analysis (MD&A)
_MDA_PART = "Part I"
_MDA_ITEM = "Item 2"

# Part II Item 1A = Risk Factors (updates only — full list lives in 10-K)
_RISK_FACTORS_PART = "Part II"
_RISK_FACTORS_ITEM = "Item 1A"


# Going concern keyword — regulator-flagged language signalling
# substantial doubt about the company's ability to continue operating
# for the next 12 months. Conservative P0 trigger.
_GOING_CONCERN_RE = re.compile(
    r"\b(?:going\s+concern|substantial\s+doubt\s+about\s+(?:its|the)\s+ability\s+to\s+continue)\b",
    re.IGNORECASE,
)

# Material weakness in internal controls — auditor finding that the
# company's financial reporting controls have a defect. Conservative
# +15 priority bump (less severe than going concern but still
# material).
_MATERIAL_WEAKNESS_RE = re.compile(
    r"\bmaterial\s+weakness(?:es)?\b",
    re.IGNORECASE,
)


# Paragraph splitter — MD&A text returned by edgartools is one big
# markdown blob. We split on double-newline to get distinct
# paragraphs. Empty / whitespace-only fragments are dropped.
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")


# Risk-factor section headings start with "##" / "###" markdown or
# uppercase phrases. We detect new headings as a proxy for "new risk
# factor introduced this quarter". Heuristic — false positives are OK
# (card just shows a count, not the heading text).
_RISK_HEADING_RE = re.compile(
    r"(?m)^(?:#{1,4}\s+|\*\*)(.+?)(?:\*\*\s*$|\s*$)",
)


# Truncate-paragraph display length for the card. ⑧ card has limited
# vertical real estate; we surface added paragraphs (truncated) only.
# 80 chars + "…" keeps each line single-row on mobile Telegram.
_PARAGRAPH_DISPLAY_LEN = 80
_TRUNCATE_SUFFIX = "…"


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class TenQDelta:
    """A 10-Q filing parsed for MD&A + risk-factors signal.

    ``mda_added_paragraphs`` carries paragraphs present in the current
    filing that don't substantially match any paragraph in the prior
    filing. Substantial match = first 80 chars equal (after whitespace
    normalisation) — catches small wording tweaks but flags genuinely
    new content.

    Flags (``has_going_concern`` / ``has_material_weakness``) are
    derived from regex scan of the MD&A text. The cron's priority
    layer reads these to apply conservative escalation.
    """

    ticker: str
    filing_date: str
    period: str                            # e.g. "Q1 2026"

    # MD&A diff (against prior quarter, or empty if no prior available)
    mda_added_paragraphs: list[str] = field(default_factory=list)

    # Risk factors diff (count of NEW headings — cheaper to render than
    # full text, and matches the "did the company add any new disclosed
    # risks?" signal users actually want)
    new_risk_factor_count: int = 0

    # Conservative auditor / regulator flags
    has_going_concern: bool = False
    has_material_weakness: bool = False

    # Raw text caches — kept so the diff_ten_q stand-alone can re-run
    # without re-pulling from edgartools. Not surfaced on the card.
    mda_text: str = ""
    risk_factors_text: str = ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_ten_q(filing) -> TenQDelta | None:
    """Parse an edgartools ``Filing`` (10-Q) into a :class:`TenQDelta`
    with the current quarter's MD&A + risk-factors text + conservative
    flags. Prior-quarter diff is computed separately via
    :func:`diff_ten_q` — this entry point fills only the current side.

    Any extraction failure → returns ``None`` and logs a warning.
    Callers should treat None as "10-Q present but unparseable" — they
    silently skip the card section.
    """
    try:
        obj = filing.obj()
    except Exception as exc:                          # noqa: BLE001
        logger.warning(
            "parse_ten_q: filing.obj() failed for %s: %s",
            getattr(filing, "accession_number", "?"), exc,
        )
        return None

    mda_text = _safe_get_item(obj, _MDA_PART, _MDA_ITEM)
    rf_text = _safe_get_item(obj, _RISK_FACTORS_PART, _RISK_FACTORS_ITEM)

    if not mda_text and not rf_text:
        logger.info(
            "parse_ten_q: no MD&A or risk factors text for %s",
            getattr(filing, "accession_number", "?"),
        )
        return None

    ticker = _safe_attr(filing, "ticker") or _safe_attr(filing, "symbol") or ""
    filing_date = str(_safe_attr(filing, "filing_date") or "")
    period = _derive_period(obj, filing_date)

    delta = TenQDelta(
        ticker=ticker,
        filing_date=filing_date,
        period=period,
        mda_text=mda_text or "",
        risk_factors_text=rf_text or "",
    )

    # Flags from current-quarter text — independent of diff. Going
    # concern + material weakness are absolute signals; we surface them
    # whether or not a prior 10-Q exists.
    if mda_text:
        delta.has_going_concern = bool(_GOING_CONCERN_RE.search(mda_text))
        delta.has_material_weakness = bool(_MATERIAL_WEAKNESS_RE.search(mda_text))

    return delta


def diff_ten_q(current: TenQDelta, prior: TenQDelta | None) -> TenQDelta:
    """Compute the diff fields on ``current`` against ``prior``.

    Mutates and returns ``current`` for caller convenience. If
    ``prior`` is None (no prior 10-Q within lookback window), returns
    ``current`` unmodified — ``mda_added_paragraphs`` stays empty and
    ``new_risk_factor_count`` stays 0. The card section will render
    nothing meaningful (just the flags if any), which is the correct
    behavior for first-quarter-after-deploy.
    """
    if prior is None:
        return current

    # MD&A added paragraphs — current paragraphs whose first 80 chars
    # don't match any prior paragraph. Truncate the displayed text to
    # _PARAGRAPH_DISPLAY_LEN (card has limited vertical space).
    current_paras = _split_paragraphs(current.mda_text)
    prior_paras = _split_paragraphs(prior.mda_text)
    prior_prefixes = {_normalize(p)[:80] for p in prior_paras}

    added: list[str] = []
    for para in current_paras:
        norm = _normalize(para)
        if norm[:80] in prior_prefixes:
            continue
        # Truncate for display. Append "…" suffix when the original
        # paragraph exceeded the display window so the reader knows
        # there's more text in the filing.
        stripped = para.strip()
        if len(stripped) > _PARAGRAPH_DISPLAY_LEN:
            display = stripped[:_PARAGRAPH_DISPLAY_LEN] + _TRUNCATE_SUFFIX
        else:
            display = stripped
        if display:
            added.append(display)
    current.mda_added_paragraphs = added[:5]   # truncate to 5 paragraphs

    # New risk factor heading count — diff of headings sets
    current_rf_headings = set(_extract_risk_headings(current.risk_factors_text))
    prior_rf_headings = set(_extract_risk_headings(prior.risk_factors_text))
    new_headings = current_rf_headings - prior_rf_headings
    current.new_risk_factor_count = len(new_headings)

    return current


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_get_item(obj, part: str, item: str) -> str | None:
    """Call obj.get_item_with_part defensively. Returns None on any
    failure (KeyError / AttributeError / network error)."""
    try:
        text = obj.get_item_with_part(part, item, markdown=True)
    except Exception as exc:                          # noqa: BLE001
        logger.warning(
            "_safe_get_item(%r, %r) failed: %s", part, item, exc,
        )
        return None
    if not text or not isinstance(text, str):
        return None
    return text


def _safe_attr(obj, name: str):
    """getattr with None fallback (no AttributeError leak)."""
    try:
        return getattr(obj, name, None)
    except Exception:                                 # noqa: BLE001
        return None


def _derive_period(obj, filing_date: str) -> str:
    """Best-effort period label. edgartools TenQ exposes
    ``period_of_report`` — use it if present, else derive from filing
    date quarter."""
    period_of_report = _safe_attr(obj, "period_of_report")
    if period_of_report:
        return str(period_of_report)
    # Fallback: derive Q from filing date month
    try:
        from datetime import date
        d = date.fromisoformat(filing_date)
        q = (d.month - 1) // 3 + 1
        return f"Q{q} {d.year}"
    except (ValueError, TypeError):
        return filing_date or "?"


def _split_paragraphs(text: str) -> list[str]:
    """Split markdown blob on double-newline; drop empty fragments."""
    if not text:
        return []
    parts = _PARAGRAPH_SPLIT_RE.split(text)
    return [p for p in parts if p and p.strip()]


def _normalize(s: str) -> str:
    """Collapse whitespace + lowercase for fuzzy-match. We compare on
    normalized prefixes so a re-typeset paragraph (same content, new
    spacing) doesn't count as added."""
    return " ".join(s.lower().split())


def _extract_risk_headings(text: str) -> list[str]:
    """Pull markdown / bold headings from a risk-factors section blob.
    Used as a proxy for distinct risk factor entries."""
    if not text:
        return []
    matches = _RISK_HEADING_RE.findall(text)
    # Normalize to dedup
    return [_normalize(m) for m in matches if m and m.strip()]


__all__ = [
    "TenQDelta",
    "parse_ten_q",
    "diff_ten_q",
]
