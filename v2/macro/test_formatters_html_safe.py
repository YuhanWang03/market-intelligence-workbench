"""HTML safety lint for the 6 public macro formatters.

A naked ``<`` in the output (i.e. one that isn't part of a known
Telegram HTML tag like ``<b>`` / ``</b>`` / ``<code>`` / ``<i>``) means
the formatter forgot to ``html.escape()`` user-supplied content and
Telegram will reject the message with the dreaded "can't parse entities"
500. Same regex pattern as the Phase 3 SEC HTML lint
(``v2/sec/test_formatters_html_safe.py``).

Defensive surfaces:
- LLM-produced fields: ``release.narrative`` / ``release.bull_takeaway``
  / ``release.bear_takeaway`` — Layer 1+2 reject in summarizer.py
  removes prediction verbs + numeric leaks, but cards still escape
  defensively since Layer 1+2 doesn't filter HTML markup.
- Tavily aggregate fields: ``event.sell_side_sources`` (URLs/hosts)
  — third-party data, must be escaped.
- FOMC statement_diff phrases: KEY_PHRASES from fomc_parser.py are
  literal strings, but a future extension that parses freer text
  could leak markup; escape anyway.

This lint runs every macro formatter through fixtures laced with
HTML metachars in the user-influenced fields.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from v2.macro._bot_cards import (  # noqa: E402
    format_macro_claims_card,
    format_macro_daily_snapshot,
    format_macro_dashboard,
    format_macro_fomc_card,
    format_macro_release_card,
    format_macro_weekly_recap,
)
from v2.macro.models import (  # noqa: E402
    FOMCEvent, MacroRelease, MacroSnapshot,
)


# Known Telegram HTML tags the formatters emit. Any `<` not followed
# by one of these (with optional closing slash) is a bare metachar
# that bypassed html.escape and will tank the Telegram render.
_ALLOWED_TAGS = ("b", "code", "i")
_UNESCAPED_LT = re.compile(
    r"<(?!/?(?:" + "|".join(_ALLOWED_TAGS) + r")\b)",
)


# ---------------------------------------------------------------------------
# Hostile fixtures (HTML metachars in user-influenced fields)
# ---------------------------------------------------------------------------

def _hostile_snapshot() -> MacroSnapshot:
    return MacroSnapshot(
        snapshot_date="2026-06-12",
        vix=14.0, vix_pct_change_1d=0.01,
        dxy=99.5, wti_crude=78.4, gold=2650.5,
        fed_funds_upper=5.50, fed_funds_lower=5.25,
        dgs2=4.21, dgs10=4.42, t10y2y=0.21, t10y2y_prior=0.18,
        warnings=["yfinance <script>VIX</script>: HTTPError",
                  "FRED DGS10 <b>fake</b>"],
    )


def _hostile_release() -> MacroRelease:
    """LLM output that smuggled HTML through Layer 1+2 — formatter
    must escape it on output."""
    return MacroRelease(
        release_type="CPI",
        release_date="2026-06-10",
        period="CPI <script>May</script> 2026",
        headline=320.5, core=315.2,
        mom_pct=0.003, yoy_pct=0.029,
        consensus=0.003, surprise_sigma=0.5,
        surprise_label="in_line<bad>",
        trailing_3mo_trend="<script>flat</script>",
        bull_takeaway="<b>bull</b> & good",
        bear_takeaway="<script>bad</script>",
        narrative="<i>narrative</i> & data",
        tone="neutral",
    )


def _hostile_fomc() -> FOMCEvent:
    return FOMCEvent(
        meeting_date="2026-06-17",
        statement_diff={
            "added_phrases": ["<script>added</script>"],
            "removed_phrases": ["<a href=\"x\">removed</a>"],
            "unchanged_phrases": [],
        },
        has_sep=True,
        sep_median_dots={2026: 4.0},
        sep_dot_plot_change="<bad>hawkish_shift",
        sell_side_sentiment="<script>hawkish</script>",
        sell_side_sources=["<script>evil.com</script>", "ben&jerry.com"],
    )


def _hostile_weekly_recap() -> dict:
    return {
        "week_start": "2026-06-08",
        "week_end": "2026-06-12",
        "weekly_deltas": {"VIXCLS": 1.0, "DGS10": 0.0, "DGS2": 0.0, "T10Y2Y": 0.0},
        "this_week_releases": {
            "2026-06-10": [("CPI<script>", "CPI", "BLS")],
        },
        "next_week_releases": {
            "2026-06-17": [("<b>FOMC</b>", "Jun FOMC", "Fed")],
        },
    }


# ---------------------------------------------------------------------------
# Lint helper
# ---------------------------------------------------------------------------

def _assert_html_safe(name: str, html_out: str) -> None:
    bad = _UNESCAPED_LT.findall(html_out)
    assert not bad, (
        f"\n{name} produced UNESCAPED '<' in output:\n"
        f"--- output ---\n{html_out}\n--- end ---\n"
        f"first bad indices: {[html_out.find(b) for b in bad[:5]]}"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_snapshot_warning_escape_html_safe():
    """⑭ snapshot — hostile warnings list must be escaped."""
    out = format_macro_daily_snapshot(_hostile_snapshot())
    _assert_html_safe("format_macro_daily_snapshot", out)


def test_release_card_html_safe():
    """⑮ release card — LLM-produced bull/bear/narrative escape."""
    out = format_macro_release_card(_hostile_release(), tier="P0")
    _assert_html_safe("format_macro_release_card", out)
    # Sanity: literal <script> tag should not survive
    assert "<script>" not in out


def test_release_card_with_next_date_html_safe():
    """Bot path with next_release_date — still safe."""
    out = format_macro_release_card(
        _hostile_release(), next_release_date="2026-07-10",
    )
    _assert_html_safe("format_macro_release_card (bot path)", out)


def test_fomc_card_html_safe():
    """⑮ FOMC card — statement diff phrases + sell-side sources escape."""
    out = format_macro_fomc_card(_hostile_fomc(), tier="P0")
    _assert_html_safe("format_macro_fomc_card", out)
    assert "<script>" not in out


def test_claims_card_html_safe():
    """⑯ Claims card — narrative + trend escape."""
    rel = _hostile_release()
    # Reuse hostile release as the Claims input — same field shape
    rel.release_type = "Claims"
    out = format_macro_claims_card(rel, tier="P2")
    _assert_html_safe("format_macro_claims_card", out)


def test_weekly_recap_html_safe():
    """⑰ Weekly recap — release-type strings in this_week/next_week escape."""
    out = format_macro_weekly_recap(_hostile_weekly_recap())
    _assert_html_safe("format_macro_weekly_recap", out)
    assert "<script>" not in out


def test_dashboard_html_safe():
    """/macro dashboard — hostile warnings + window release-types escape."""
    out = format_macro_dashboard(
        _hostile_snapshot(),
        {"2026-06-10": [("<script>CPI</script>", "label", "BLS")]},
        "2026-06-12",
    )
    _assert_html_safe("format_macro_dashboard", out)
    assert "<script>" not in out


# ---------------------------------------------------------------------------
# Sanity: the regex itself catches obvious cases
# ---------------------------------------------------------------------------

def test_regex_catches_bare_lt():
    assert _UNESCAPED_LT.search("<script>") is not None
    assert _UNESCAPED_LT.search("foo <bar") is not None


def test_regex_allows_known_tags():
    assert _UNESCAPED_LT.search("<b>hi</b>") is None
    assert _UNESCAPED_LT.search("<code>x</code>") is None
    assert _UNESCAPED_LT.search("<i>note</i>") is None
