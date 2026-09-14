"""Insider title classifier — Form 4 reporting owner → CEO/CFO/Director/None.

The Stage 0 priority spec wants per-role bumps:
- CEO/CFO purchase → +10 floor bump even on small amounts
- Director purchase → standard magnitude rules
- Other Officer / 10%-holder → no special bump

edgartools' ``Form4.reporting_owners`` exposes a ``ReportingOwners``
collection. Each owner has ``relationship`` flags
(``is_officer`` / ``is_director`` / ``is_ten_percent_owner``) and a
free-text ``officer_title`` field with values like:

    "Chief Executive Officer"
    "President and CEO"
    "Chief Financial Officer"
    "EVP, Chief Financial Officer"
    "Senior VP and General Counsel"
    "Director"
    ""

We classify by substring match — case-insensitive, longest-priority-first
ordering. Returns ``None`` if no clear match (defensive — never assert a
role we can't substantiate, because Stage 2 priority bumps key off this).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# Substring → canonical role label. Order matters: more specific patterns
# first so "Chief Executive Officer" wins over a generic "Officer" match.
# Each label maps to None at lookup-failure so the cron prioritizes
# conservatively.
_TITLE_PATTERNS: tuple[tuple[str, str], ...] = (
    # Long phrases first — match before falling through to short codes.
    ("chief executive officer", "CEO"),
    ("chief exec", "CEO"),
    ("president and ceo", "CEO"),
    ("chief financial officer", "CFO"),
    ("chief financial", "CFO"),
    ("chief operating officer", "COO"),
    ("general counsel", "GC"),
    ("chairman", "Chairman"),
    ("president", "President"),
    # Short codes — accept bare "CEO" / "CFO" / "COO" since real
    # officer_title strings sometimes use abbreviations alone. The
    # substring risk (e.g. "OCEONOGRAPHY" containing "ceo") is
    # negligible for corporate titles.
    ("ceo", "CEO"),
    ("cfo", "CFO"),
    ("coo", "COO"),
    # Bare "director" must come last among officer-ish matches so it
    # doesn't shadow titled execs who also sit on the board.
    ("director", "Director"),
)


def _classify_title(title: str) -> str | None:
    """Return canonical role label from a free-text officer title.

    Empty / unrecognized title returns ``None`` — callers conservatively
    treat that as "no special priority bump."
    """
    if not title:
        return None
    needle = title.lower()
    for pattern, role in _TITLE_PATTERNS:
        if pattern in needle:
            return role
    return None


def lookup_insider_role(form4: Any) -> str | None:
    """Return the canonical role label for a Form 4's reporting owner.

    The Form 4 typically has a single reporting owner. When multiple
    (rare — joint filings), the first owner is used. If the SDK's
    ``reporting_owners`` shape is unexpected (edgartools occasionally
    refactors), this returns ``None`` rather than raising, and the
    pipeline falls back to magnitude-only priority.

    Returns:
        ``"CEO"`` / ``"CFO"`` / ``"COO"`` / ``"Chairman"`` / ``"President"``
        / ``"GC"`` / ``"Director"`` / ``None`` (unclassified / lookup failed).
    """
    try:
        owners = getattr(form4, "reporting_owners", None)
        if owners is None:
            return None

        # ReportingOwners may be iterable, may have an ``.owners`` list,
        # or may be a single ``Owner``. Defensive normalization.
        owner_list: list[Any] = []
        if hasattr(owners, "owners"):
            owner_list = list(getattr(owners, "owners", []) or [])
        elif isinstance(owners, (list, tuple)):
            owner_list = list(owners)
        else:
            # Single owner object
            owner_list = [owners]

        if not owner_list:
            return None

        first = owner_list[0]

        # Try the structured-relationship path first.
        relationship = getattr(first, "relationship", None) or getattr(first, "relation", None)
        if relationship is not None:
            # Free-text title is the high-signal field
            title = (
                getattr(relationship, "officer_title", None)
                or getattr(relationship, "title", None)
                or ""
            )
            label = _classify_title(str(title))
            if label:
                return label

            # If title was empty but director flag is set → "Director"
            if getattr(relationship, "is_director", False):
                return "Director"
            if getattr(relationship, "is_officer", False):
                return "Officer"
            if getattr(relationship, "is_ten_percent_owner", False):
                return "10% holder"

        # Fallback: try the owner's free-text "title" attribute directly
        title = getattr(first, "officer_title", None) or getattr(first, "title", None) or ""
        return _classify_title(str(title))

    except Exception as exc:
        logger.warning("insider_role lookup failed: %s", exc)
        return None
