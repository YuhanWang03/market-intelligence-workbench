"""Data classes for v2/sec — Phase 3 SEC monitoring agent.

Three filing types are in scope for Phase 3:
- **8-K** (event-driven material disclosures, parsed by item codes)
- **Form 4** (insider transaction reports, parsed per-transaction)

10-Q is deferred to Phase 3.5 per Stage 0 decision.

The shape distinction vs ``v2.institutional`` (13F):
- 13F is a *list of positions* in a portfolio snapshot
- 8-K is a *list of events* with priority-graded item codes
- Form 4 is a *list of transactions* with code-graded magnitude signals

Two separate modules keep their parsing logic decoupled — see Stage 0 task 1.

All numeric fields default to ``None`` where the SEC submission can omit
them (e.g. Form 4 with no transaction price); renderers downstream gate
on ``None`` rather than crash.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


PriorityTier = Literal["P0", "P1", "P2", "P3"]


# Tier ordering for max_priority_tier — lower index = higher priority
_TIER_ORDER: dict[str, int] = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}


# ---------------------------------------------------------------------------
# Common metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SecFiling:
    """Metadata common to every SEC filing this module handles.

    ``ticker`` is the local universe symbol (set by the cron); ``cik`` is
    SEC's central identifier (set by edgartools). ``accession_number`` is
    the unique per-filing key — primary dedup pin downstream.

    ``is_amendment`` flags 8-K/A and Form 4/A re-filings: priority gets a
    -5 nudge (Stage 0 priority spec) since amendments rarely add signal.
    """

    ticker: str
    cik: str
    form: str                   # "8-K" / "4" / "8-K/A" / "4/A"
    filing_date: str            # ISO date — when SEC received it
    accession_number: str
    is_amendment: bool = False


# ---------------------------------------------------------------------------
# 8-K
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EightKItem:
    """One item code disclosed inside a single 8-K filing.

    A single 8-K can carry multiple items (HPE Stage-0 example: 1.01 + 2.02
    + 5.02 + 7.01 + 9.01 in one filing). Each item gets its own
    EightKItem with its own per-item priority; the parent EightKEvent
    rolls them up via ``max_priority_tier``.

    ``extracted_meta`` is item-specific structured data. For 5.02 (the
    only item requiring LLM extraction in Stage 1) it carries:
        {"departures": [{"name", "title"}, ...],
         "appointments": [{"name", "title"}, ...],
         "has_senior_exec": bool}
    For other items it's empty {}.
    """

    code: str                   # "5.02" / "1.05" / "2.02" / ...
    priority_tier: PriorityTier
    description: str            # human-readable label for the card
    extracted_meta: dict = field(default_factory=dict)


@dataclass
class EightKEvent:
    """One 8-K filing with all its items aggregated.

    Single-card output per filing (Stage 0 calibration #3 from real data
    — don't fragment HPE's 5-item filing into 5 cards).

    ``has_earnings_overlap`` is True when the items list contains 2.02
    (Results of Operations) — Stage 0 calibration #4 — and the cron
    skips the entire filing if 2.02 is the ONLY material item (with 9.01
    financial-statements exhibit attached, which is always co-filed).
    Mixed cases (e.g. 2.02 + 5.02) keep the card and annotate 2.02
    with "(handled by ⑧)".
    """

    filing: SecFiling
    items: list[EightKItem] = field(default_factory=list)
    has_earnings_overlap: bool = False

    @property
    def max_priority_tier(self) -> PriorityTier:
        """Highest-priority item drives the filing's tier."""
        if not self.items:
            return "P3"
        return min(
            (it.priority_tier for it in self.items),
            key=lambda t: _TIER_ORDER[t],
        )

    @property
    def is_2_02_only(self) -> bool:
        """True iff items reduce to {2.02} or {2.02, 9.01}.

        9.01 (financial statements/exhibits) is administrative and
        always co-filed — treat it as transparent for skip-detection.
        """
        material_codes = {it.code for it in self.items if it.code != "9.01"}
        return material_codes == {"2.02"}


# ---------------------------------------------------------------------------
# Form 4
# ---------------------------------------------------------------------------

# These codes account for ~83% of Form 4 volume in the Stage 0 universe
# (39 A + 19 M + 13 F + 3 G). They batch into the weekly insider digest
# rather than pushing single notifications.
NOISE_TRANSACTION_CODES: frozenset[str] = frozenset({
    "A",   # Award (RSU / stock grant) — comp, not signal
    "M",   # Exercise of derivative (option → stock) — routine vesting
    "F",   # Tax withholding via share retention — pure tax operation
    "G",   # Gift — no economic signal
    "C",   # Conversion of derivative — routine
})

# These are the high-signal codes that get individual notifications.
SIGNAL_TRANSACTION_CODES: frozenset[str] = frozenset({"P", "S"})


@dataclass(frozen=True)
class Form4Transaction:
    """One row from a Form 4's non-derivative transaction table.

    ``transaction_usd`` is derived (shares × price). Either side can be
    None — Form 4 occasionally omits price (e.g. gifts, M-code exercises
    settled in shares-of-self) — in which case ``transaction_usd`` is
    also None and renderers display "未披露".

    ``is_10b5_1`` parses Rule 10b5-1 trading plan markers out of the
    footnotes free-text. Discretionary (non-10b5-1) large sales are the
    interesting case — pre-arranged plan sales mute the signal.
    """

    filing: SecFiling
    insider_name: str
    insider_role: str | None    # "CEO" / "CFO" / "Director" / "Officer" / None
    transaction_code: str       # "P" / "S" / "A" / "M" / "F" / "G" / "C" / ...
    transaction_date: str       # ISO date — when the trade settled
    shares: float
    price: float | None
    transaction_usd: float | None
    is_10b5_1: bool = False
    direct_indirect: str = "D"  # "D" = direct, "I" = indirect

    @property
    def direction(self) -> str | None:
        """Returns ``'purchase'`` for P, ``'sale'`` for S, ``None`` for others.

        Cluster detection groups by direction so an A-code grant on the
        same day as a P-code purchase doesn't dilute the cluster count.
        """
        if self.transaction_code == "P":
            return "purchase"
        if self.transaction_code == "S":
            return "sale"
        return None

    @property
    def is_signal(self) -> bool:
        """True iff this transaction warrants its own card (not noise-batched)."""
        return self.transaction_code in SIGNAL_TRANSACTION_CODES

    @property
    def is_noise(self) -> bool:
        """True iff this transaction goes into the weekly digest, not its own card."""
        return self.transaction_code in NOISE_TRANSACTION_CODES


@dataclass(frozen=True)
class Form4Cluster:
    """≥ 3 same-day same-direction P/S transactions for one ticker.

    Stage 0 calibration #1 — the threshold tightened from
    "30-day rolling cluster" (which fired on 80% of the universe) to
    "same-day same-direction". Same-day means a coordinated event;
    coincidence is implausible at ≥ 3.

    The pipeline returns a Form4Cluster alongside the individual
    Form4Transaction entries that compose it. The cron's priority
    layer (Stage 2) applies a +15 P1 bump on cluster presence.
    """

    ticker: str
    cluster_date: str           # ISO date — the shared filing_date
    direction: str              # "purchase" or "sale"
    transaction_count: int
    total_usd: float            # sum of transactions[i].transaction_usd (None → 0)
    insider_names: list[str]
    transactions: list[Form4Transaction]


# ---------------------------------------------------------------------------
# Top-level scan result
# ---------------------------------------------------------------------------

@dataclass
class SecScanResult:
    """Result envelope returned by ``pipeline.run_sec_scan``.

    Sub-failures (per-ticker EDGAR 503, per-filing parse error, per-item
    LLM extraction failure) populate ``warnings`` instead of raising.
    Cron path needs guaranteed return so the schedulered job can always
    write SOMETHING to archive even on a degraded day.

    Same pattern as Phase 2's ``RiskReport.warnings`` — formatter renders
    the warnings list as small italics at the bottom of the card if any.
    """

    eight_k_events: list[EightKEvent] = field(default_factory=list)
    form4_signal_transactions: list[Form4Transaction] = field(default_factory=list)
    form4_clusters: list[Form4Cluster] = field(default_factory=list)
    # noise_summary: {ticker: {code: count}} for the weekly digest cron
    form4_noise_summary: dict[str, dict[str, int]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
