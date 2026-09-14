"""Weekly insider activity digest — Phase 3.5.

⑫b Friday 19:15 ET cron aggregates the past week's ⑫ Form 4 cron
pushes into a single weekly summary card. Sits between ⑩ Portfolio
Weekly (19:00 ET) and ⑰ Macro Weekly Recap (19:30 ET).

Data source — title-only fallback (Stage 1 trace probe):

⑫ Form 4 cron stores each push in ``archive.pushes`` with:

- ``agent = 'sec'``
- ``title`` = ``"Form 4 · NVDA · 买入"`` (per-tx individual cards)
  OR ``"Form 4 集群 · ARM · purchase"`` (cluster cards)
- ``tickers`` = comma-joined string
- ``trace_json`` carries only the aggregate framing event
  (``"SEC Form 4 扫描 · N 只 · X signal · Y cluster · Z noise codes"``)
  — NOT per-transaction structured data

So this digest aggregates at the ``(ticker, direction)`` granularity
from the title field. The spec's full ``A/M/F/G/C`` breakdown lives
ONLY in the cron's runtime memory (``form4_noise_summary`` dict, not
persisted) — that part is deferred to Phase 3.5.5 which would add a
structured ``form4_transactions`` table. Not in scope for Phase 3.5
per the Stage 0 decision.

What the digest CAN report from title-only:

- Number of unique tickers active this week
- Per-direction push counts (buy vs sell vs cluster)
- Top tickers by push count (proxy for "unusual activity")

What it can't (without schema change):

- Per-code breakdown (A awards vs M exercises vs F tax)
- Aggregate USD totals (would need text_html parsing — fragile)

The card surfaces what it can and explicitly captions the limitation
so users understand the digest is push-count-based, not
transaction-volume-based.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta

logger = logging.getLogger(__name__)


# Title parse — match ⑫'s _push_signal / _push_cluster title shapes.
# Per the source-of-truth in scripts/sec_form4_to_telegram.py:
#   individual buy:  "Form 4 · NVDA · 买入"
#   individual sell: "Form 4 · NVDA · 卖出"
#   cluster:         "Form 4 集群 · ARM · purchase" / "... · sale"
_TITLE_SIGNAL_RE = re.compile(
    r"^Form\s+4\s+·\s+([A-Z]{1,5})\s+·\s+(买入|卖出)$"
)
_TITLE_CLUSTER_RE = re.compile(
    r"^Form\s+4\s+集群\s+·\s+([A-Z]{1,5})\s+·\s+(purchase|sale)$"
)


# Unusual activity threshold: a ticker with ≥ this many pushes in
# one week is flagged. 3 is a deliberately low bar — picks up coordinated
# inside-buying clusters (Stage 0 calibration found ~17% of weeks have
# at least one ticker with 3+ pushes; the rest are quiet).
_UNUSUAL_PUSH_THRESHOLD = 3


@dataclass
class WeeklyInsiderSummary:
    """Title-only aggregation of one week's ⑫ Form 4 cron pushes.

    All fields derive from title parsing — no LLM, no text_html scan.
    Fields the spec implied but we can't fill from archive alone
    (per-A/M/F/G/C code breakdown + USD totals) are absent here and
    deferred to Phase 3.5.5.
    """

    week_start: str         # ISO YYYY-MM-DD, inclusive
    week_end: str           # ISO YYYY-MM-DD, inclusive

    # Per-direction push counts
    purchase_push_count: int = 0
    sale_push_count: int = 0
    cluster_purchase_count: int = 0
    cluster_sale_count: int = 0

    # Per-ticker totals (sum of all push kinds above for that ticker)
    by_ticker: dict[str, int] = field(default_factory=dict)

    # Tickers ≥ _UNUSUAL_PUSH_THRESHOLD this week
    unusual_tickers: list[str] = field(default_factory=list)

    # Total distinct tickers active (any direction)
    total_tickers_active: int = 0

    # Total pushes (all directions, all kinds)
    total_push_count: int = 0

    @property
    def is_quiet_week(self) -> bool:
        """No ⑫ Form 4 cron pushes landed this week."""
        return self.total_push_count == 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_weekly_digest(
    archive,
    week_start_iso: str,
    week_end_iso: str,
) -> WeeklyInsiderSummary:
    """Aggregate ⑫ Form 4 pushes between ``week_start_iso`` and
    ``week_end_iso`` (both ISO YYYY-MM-DD, inclusive).

    Reads ``pushes`` table directly: ``agent='sec'`` AND ``title LIKE
    'Form 4%'`` AND ``ts`` in window. Each matched row's ``title`` is
    parsed via :data:`_TITLE_SIGNAL_RE` / :data:`_TITLE_CLUSTER_RE` to
    derive (ticker, direction, is_cluster) — non-matching titles are
    silently skipped.

    Empty week → returns :class:`WeeklyInsiderSummary` with all counts
    zero and ``is_quiet_week=True``. The ⑫b cron renders a "本周内部人
    活动平静" card in that case — operator visibility floor.
    """
    summary = WeeklyInsiderSummary(
        week_start=week_start_iso, week_end=week_end_iso,
    )

    rows = _query_form4_pushes(archive, week_start_iso, week_end_iso)
    ticker_counts: Counter[str] = Counter()

    for row in rows:
        title = (row.get("title") or "").strip()
        if not title:
            continue

        # Try cluster shape FIRST (more specific — has "集群" prefix
        # that distinguishes from individual)
        m = _TITLE_CLUSTER_RE.match(title)
        if m:
            ticker, direction = m.group(1), m.group(2)
            if direction == "purchase":
                summary.cluster_purchase_count += 1
            else:
                summary.cluster_sale_count += 1
            ticker_counts[ticker] += 1
            continue

        m = _TITLE_SIGNAL_RE.match(title)
        if m:
            ticker, direction_zh = m.group(1), m.group(2)
            if direction_zh == "买入":
                summary.purchase_push_count += 1
            else:
                summary.sale_push_count += 1
            ticker_counts[ticker] += 1
            continue

        # Non-Form-4 push title or unrecognized shape — skip silently
        # (these will be e.g. /summary or other agent rows that don't
        # match our query but slipped through due to test seam)
        logger.debug("insider_digest: skip unmatched title %r", title)

    summary.by_ticker = dict(ticker_counts.most_common())
    summary.total_tickers_active = len(ticker_counts)
    summary.total_push_count = (
        summary.purchase_push_count + summary.sale_push_count
        + summary.cluster_purchase_count + summary.cluster_sale_count
    )
    summary.unusual_tickers = [
        ticker for ticker, n in ticker_counts.most_common()
        if n >= _UNUSUAL_PUSH_THRESHOLD
    ]
    return summary


def default_week_window(today_iso: str) -> tuple[str, str]:
    """Compute (week_start, week_end) ISO bounds for the digest.

    Default window: Monday-Friday OF THE WEEK CONTAINING ``today_iso``.
    Since ⑫b runs Fri 19:15 ET, today_iso IS the Friday → week_start
    = today - 4 days (Mon), week_end = today (Fri).

    On weekend or off-day reruns the window still anchors to the
    week containing today_iso for predictable behavior.
    """
    try:
        today_d = date.fromisoformat(today_iso)
    except ValueError:
        logger.warning("default_week_window: bad iso %r", today_iso)
        return today_iso, today_iso
    # weekday() Monday=0 ... Sunday=6
    days_since_monday = today_d.weekday()
    week_start = today_d - timedelta(days=days_since_monday)
    week_end = week_start + timedelta(days=4)   # always end on Friday
    return week_start.isoformat(), week_end.isoformat()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _query_form4_pushes(
    archive,
    week_start_iso: str,
    week_end_iso: str,
) -> list[dict]:
    """SELECT ⑫ Form 4 pushes from archive.pushes in window.

    Done via the archive's internal _conn() context manager so the
    test harness's in-memory archive fakes can patch the helper if
    they don't want to expose raw SQLite. Production Archive's
    _conn() is a sqlite3 connection that returns sqlite3.Row, which
    is dict-like for indexing — we convert to plain dict for the
    consumers below.
    """
    # Some callers (tests) inject an object with a pre-built
    # ``get_form4_pushes_in_window`` method to skip SQL. Production
    # Archive doesn't have that helper so we fall back to the raw
    # query. Both paths return list[dict].
    helper = getattr(archive, "get_form4_pushes_in_window", None)
    if helper is not None:
        return helper(week_start_iso, week_end_iso)

    # Production path — straight SQL.
    sql = (
        "SELECT id, ts, agent, title, tickers, text_html "
        "FROM pushes "
        "WHERE agent='sec' AND title LIKE 'Form 4%' "
        "AND ts >= ? AND ts < ? "
        "ORDER BY ts ASC"
    )
    # ts is stored as ISO datetime; comparing against date-prefix
    # strings works because ISO dates sort lexically. Inclusive
    # end_iso = include all of week_end day → use "<" against
    # week_end + 1 day's start.
    try:
        end_d = date.fromisoformat(week_end_iso) + timedelta(days=1)
        end_exclusive = end_d.isoformat()
    except ValueError:
        end_exclusive = week_end_iso

    try:
        with archive._conn() as conn:
            cur = conn.execute(sql, (week_start_iso, end_exclusive))
            return [dict(row) for row in cur.fetchall()]
    except Exception as exc:                          # noqa: BLE001
        logger.warning("_query_form4_pushes failed: %s", exc)
        return []


__all__ = [
    "WeeklyInsiderSummary",
    "build_weekly_digest",
    "default_week_window",
]
