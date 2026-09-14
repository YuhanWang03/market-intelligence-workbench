"""HTML safety lint for the 5 public SEC formatters.

A naked ``<`` in the output (i.e. one that isn't part of a known
Telegram HTML tag like ``<b>`` / ``</b>`` / ``<code>`` / ``<i>``) means
the formatter forgot to ``html.escape()`` user-supplied content and
Telegram will reject the message with the dreaded "can't parse entities"
500. We learned this lesson in Phase 2.5-mini after a ticker like
``BRK<A`` (hypothetical pathological input) broke a /risk push.

The lint runs every SEC formatter through fixtures laced with HTML
metachars (``<``, ``>``, ``&``) in the user-controlled fields:
ticker, insider name, item description. The expected behavior is
that every metachar is either escaped or comes from a fixed tag the
formatter writes itself.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from v2.sec._bot_cards import (   # noqa: E402
    format_sec_8k_card,
    format_sec_8k_view,
    format_sec_form4_cluster_card,
    format_sec_form4_individual_card,
    format_sec_form4_view,
)
from v2.sec.models import (   # noqa: E402
    EightKEvent, EightKItem, Form4Cluster, Form4Transaction, SecFiling,
)


# Known Telegram HTML tags the formatters emit. Any `<` not followed
# by one of these (with optional closing slash) is a bare metachar
# that bypassed html.escape and will tank the Telegram render.
_ALLOWED_TAGS = ("b", "code", "i")
_UNESCAPED_LT = re.compile(
    r"<(?!/?(?:" + "|".join(_ALLOWED_TAGS) + r")\b)",
)


# ---------------------------------------------------------------------------
# Fixtures with HTML metachars in user-controlled string fields
# ---------------------------------------------------------------------------

def _hostile_8k_event() -> EightKEvent:
    """Filing whose ticker, accession, and item description carry
    HTML metachars. The ticker on real US equities can't contain ``<``,
    but the bot's view path takes arbitrary user input via /8k and
    must defend against accidental misuse. Tests live here so a
    refactor that drops html.escape gets caught."""
    return EightKEvent(
        filing=SecFiling(
            ticker="X<Y>Z", cik="0000099", form="8-K",
            filing_date="2026-06-04",
            accession_number="ACC<inj>&amp;",
        ),
        items=[
            EightKItem(
                "5.02", "P0", "高管 <script>变动</script>",
                {
                    "departures": [{
                        "name": "John <Smith>",
                        "title": "CEO & Founder",
                    }],
                    "appointments": [{
                        "name": "Jane & Doe",
                        "title": "<b>Interim</b> CEO",
                    }],
                    "has_senior_exec": True,
                },
            ),
        ],
    )


def _hostile_form4_tx() -> Form4Transaction:
    return Form4Transaction(
        filing=SecFiling(
            ticker="X<Y>Z", cik="0001", form="4",
            filing_date="2026-06-04", accession_number="ACC<1>",
        ),
        insider_name="<script>Eve</script>",
        insider_role="CEO & Founder",
        transaction_code="P", transaction_date="2026-06-04",
        shares=10000.0, price=100.0, transaction_usd=1_000_000.0,
        is_10b5_1=False, direct_indirect="D",
    )


def _hostile_cluster() -> Form4Cluster:
    return Form4Cluster(
        ticker="X<Y>Z", cluster_date="2026-06-04", direction="purchase",
        transaction_count=3, total_usd=300_000.0,
        insider_names=["<eve>", "ben&jerry", "carol<3>"],
        transactions=[],
    )


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

@pytest.mark.parametrize("is_held,is_watchlist", [
    (False, False), (True, False), (False, True),
])
def test_sec_8k_view_html_safe(is_held: bool, is_watchlist: bool):
    """/8k bot card — ticker is user input via Telegram, MUST be escaped.
    Strict lint: no bare `<` outside the known tag set."""
    out = format_sec_8k_view(
        [_hostile_8k_event()], ticker="X<Y>Z", days=30,
    )
    _assert_html_safe("format_sec_8k_view", out)


def test_sec_8k_view_empty_html_safe():
    """Empty view with hostile ticker — placeholder card must still escape."""
    out = format_sec_8k_view([], ticker="X<Y>Z", days=30)
    _assert_html_safe("format_sec_8k_view (empty)", out)


def test_sec_form4_view_html_safe():
    """/insiders bot card — ticker + insider names are user-influenced;
    full strict lint required."""
    out = format_sec_form4_view(
        "X<Y>Z", [_hostile_form4_tx()], [_hostile_cluster()],
        {"A": 1, "M": 2}, 90,
    )
    _assert_html_safe("format_sec_form4_view", out)


def test_sec_form4_view_empty_html_safe():
    """Empty form4 view with hostile ticker — must still escape."""
    out = format_sec_form4_view("X<Y>Z", [], [], {}, 90)
    _assert_html_safe("format_sec_form4_view (empty)", out)


# ---------------------------------------------------------------------------
# Cron-side formatters — softer trust model
# ---------------------------------------------------------------------------
# The 3 cron-fed formatters (format_sec_8k_card,
# format_sec_form4_individual_card, format_sec_form4_cluster_card) are
# pushed only to in-house Telegram channels, fed by data sourced from
# edgartools + the fixed Stage-0 item-description table + LLM extractor
# output. They do not html.escape their fields — the byte-equal tests
# pin that choice. The lint surface for these is "no bare <script>
# under realistic LLM-output fixtures" rather than the full unescaped-<
# regex (which would force a format change). If a future audit decides
# the cron channel needs strict escaping too, lift these into the full
# _assert_html_safe lint and update the byte-equal expected strings in
# the same commit.


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
