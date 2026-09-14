"""FOMC statement diff + SEP dot-plot extraction — Phase 4 Stage 1.

Layer 3 hallucination defense (Stage 0 design ack): the LLM is NOT
asked whether the FOMC statement is hawkish or dovish. Python does
the diff against the prior statement (which key phrases were added /
removed), and the SEP dot plot is a numeric table — also Python.

Tavily aggregates external sell-side opinions separately in
:mod:`v2.macro.tavily_consensus`; the union of (a) Python phrase diff,
(b) SEP numeric shift, and (c) Tavily majority vote is what the card
renders. The system never asserts a verdict the LLM made up.
"""

from __future__ import annotations

import logging
import re
from collections import OrderedDict

logger = logging.getLogger(__name__)


# Key phrases that historically signal tightening / easing direction
# in FOMC statements. Ordered roughly by importance (most policy-
# loaded first); the diff function reports each one independently.
KEY_PHRASES: tuple[str, ...] = (
    "appropriate firming",
    "additional policy firming",
    "substantial further progress",
    "considerable additional tightening",
    "modest additional firming",
    "moderate",
    "modest",
    "substantial",
    "considerable",
    "elevated",
    "longer-run",
    "gradual",
    "remains highly attentive to inflation risks",
    "balance of risks",
    "data-dependent",
)


# ---------------------------------------------------------------------------
# Statement diff (Python only)
# ---------------------------------------------------------------------------

def diff_statements(current: str, prior: str) -> dict:
    """Compare two FOMC statements at the key-phrase level.

    Returns a dict with:
    - ``added_phrases``: phrases that appear in current but not prior.
    - ``removed_phrases``: phrases that appear in prior but not current.
    - ``unchanged_phrases``: phrases present in both (provided for
      completeness — most use cases only care about added / removed).

    Case-insensitive substring match. Phrases are checked independently
    so "moderate" + "modest" each get their own verdict.
    """
    current_lc = (current or "").lower()
    prior_lc = (prior or "").lower()

    added: list[str] = []
    removed: list[str] = []
    unchanged: list[str] = []

    for phrase in KEY_PHRASES:
        in_curr = phrase in current_lc
        in_prior = phrase in prior_lc
        if in_curr and not in_prior:
            added.append(phrase)
        elif in_prior and not in_curr:
            removed.append(phrase)
        elif in_curr and in_prior:
            unchanged.append(phrase)

    return {
        "added_phrases": added,
        "removed_phrases": removed,
        "unchanged_phrases": unchanged,
    }


# ---------------------------------------------------------------------------
# SEP dot plot extraction
# ---------------------------------------------------------------------------

# The SEP "Summary of Economic Projections" PDF contains a table that
# (when extracted with pdfplumber / pdf2text) typically renders as
# something like:
#
#   Variable          2024    2025    2026    2027    Longer run
#   Federal funds rate 5.4    4.6     3.6     2.9      2.9
#
# We pin to the "Federal funds rate" row. The values are rounded to 1
# decimal in the SEP — our extractor preserves that precision.

_SEP_FED_FUNDS_RE = re.compile(
    r"Federal\s+funds\s+rate[^\d\n]*"
    r"((?:\d+\.\d+\s+){3,5}\d+\.\d+)",
    re.IGNORECASE,
)


_SEP_YEAR_HEADER_RE = re.compile(
    r"\b(20\d{2})\s+(20\d{2})\s+(20\d{2})(?:\s+(20\d{2}))?(?:\s+(20\d{2}))?",
)


def extract_dot_plot_table(sep_pdf_text: str) -> dict | None:
    """Pull the Federal Funds Rate row out of the SEP text and pair
    each rate with its year header.

    Returns a dict like
    ``{2024: 5.4, 2025: 4.6, 2026: 3.6, 2027: 2.9, "longer_run": 2.9}``
    on success, or None if the regex can't find both the year row and
    the rate row (in which case the cron path skips SEP entirely).
    """
    if not sep_pdf_text:
        return None

    fed_funds_match = _SEP_FED_FUNDS_RE.search(sep_pdf_text)
    if not fed_funds_match:
        return None

    rates = [float(x) for x in fed_funds_match.group(1).split()]

    year_match = _SEP_YEAR_HEADER_RE.search(sep_pdf_text)
    if not year_match:
        return None
    years = [int(g) for g in year_match.groups() if g]

    if len(rates) < len(years):
        return None

    result: dict = OrderedDict()
    for year, rate in zip(years, rates):
        result[year] = rate
    # The trailing rate in the table is the longer-run dot
    if len(rates) > len(years):
        result["longer_run"] = rates[len(years)]
    return dict(result)


# ---------------------------------------------------------------------------
# Classify SEP shift (current vs prior medians)
# ---------------------------------------------------------------------------

def classify_sep_shift(current: dict, prior: dict | None) -> str:
    """Compare two SEP fed-funds-rate medians. Returns:

    - ``"hawkish_shift"`` if any year's median is higher in current
      than prior (no offsetting lower years).
    - ``"dovish_shift"`` if any year's median is lower (no offsetting
      higher years).
    - ``"no_change"`` for exact match, missing prior, or mixed shifts
      (some years up, some down) — the latter is rare but happens when
      the Fed steepens vs front-loads.

    Only compares years present in BOTH dicts (so a longer-run rate
    appearing for the first time doesn't count).
    """
    if not current or not prior:
        return "no_change"

    ups, downs = 0, 0
    for key, val in current.items():
        if key not in prior:
            continue
        prior_val = prior[key]
        if val > prior_val:
            ups += 1
        elif val < prior_val:
            downs += 1

    if ups > 0 and downs == 0:
        return "hawkish_shift"
    if downs > 0 and ups == 0:
        return "dovish_shift"
    return "no_change"


__all__ = [
    "KEY_PHRASES",
    "diff_statements",
    "extract_dot_plot_table",
    "classify_sep_shift",
]
