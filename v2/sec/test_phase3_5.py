"""Smoke tests for Phase 3.5 — ten_q_parser + insider_digest.

ten_q_parser tests use SimpleNamespace mocks of the edgartools TenQ
shape (verified Stage 0: ``get_item_with_part(part, item, markdown=True)
-> Optional[str]``). No edgartools install required for these tests.

insider_digest tests use a fake archive that exposes
``get_form4_pushes_in_window`` so the SQL path isn't required for
smoke coverage; the SQL path is exercised end-to-end in Stage 4 cron
integration tests against a real Archive instance.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from v2.sec.insider_digest import (   # noqa: E402
    WeeklyInsiderSummary,
    build_weekly_digest,
    default_week_window,
)
from v2.sec.ten_q_parser import (     # noqa: E402
    TenQDelta,
    diff_ten_q,
    parse_ten_q,
)


# ---------------------------------------------------------------------------
# edgartools mock helpers
# ---------------------------------------------------------------------------

def _make_filing(
    *,
    ticker: str = "AAPL",
    filing_date: str = "2026-05-15",
    accession_number: str = "0000123-26-000999",
    mda_text: str | None = "Revenue increased 5% year-over-year.\n\nWe continued to invest in R&D.",
    risk_factors_text: str | None = None,
    period_of_report: str | None = "Q1 2026",
    raise_on_obj: bool = False,
):
    """Build a SimpleNamespace that mimics the edgartools filing +
    .obj() result for parse_ten_q. Each call configures the response
    of get_item_with_part for the two paths we care about."""
    item_map: dict[tuple[str, str], str | None] = {
        ("Part I", "Item 2"): mda_text,
        ("Part II", "Item 1A"): risk_factors_text,
    }

    def get_item_with_part(part, item, markdown=True):
        return item_map.get((part, item))

    obj = SimpleNamespace(
        get_item_with_part=get_item_with_part,
        period_of_report=period_of_report,
    )

    def filing_obj():
        if raise_on_obj:
            raise RuntimeError("edgar 503")
        return obj

    return SimpleNamespace(
        ticker=ticker,
        filing_date=filing_date,
        accession_number=accession_number,
        obj=filing_obj,
    )


# ---------------------------------------------------------------------------
# ten_q_parser tests
# ---------------------------------------------------------------------------

def test_parse_ten_q_extracts_mda():
    """MD&A text populated from Part I Item 2."""
    filing = _make_filing(mda_text="Revenue up 5%.\n\nCash position strong.")
    delta = parse_ten_q(filing)
    assert delta is not None
    assert delta.ticker == "AAPL"
    assert "Revenue up 5%" in delta.mda_text
    assert delta.period == "Q1 2026"


def test_parse_ten_q_extracts_risk_factors():
    """Risk factors text populated from Part II Item 1A when present."""
    filing = _make_filing(
        mda_text="MD&A content.",
        risk_factors_text="## New risk: supply chain disruption.",
    )
    delta = parse_ten_q(filing)
    assert delta is not None
    assert "supply chain" in delta.risk_factors_text


def test_parse_ten_q_handles_missing_text():
    """No MD&A AND no risk factors → returns None (10-Q exists but
    unparseable). Caller silently skips the section."""
    filing = _make_filing(mda_text=None, risk_factors_text=None)
    assert parse_ten_q(filing) is None


def test_parse_ten_q_obj_exception_returns_none():
    """filing.obj() raises → None, no propagation."""
    filing = _make_filing(raise_on_obj=True)
    assert parse_ten_q(filing) is None


def test_parse_ten_q_going_concern_flag():
    """Going concern keyword in MD&A → has_going_concern=True."""
    filing = _make_filing(
        mda_text=(
            "Our cash position has deteriorated.\n\n"
            "There is substantial doubt about our ability to continue "
            "operating as a going concern."
        ),
    )
    delta = parse_ten_q(filing)
    assert delta is not None
    assert delta.has_going_concern is True


def test_parse_ten_q_material_weakness_flag():
    """Material weakness keyword → has_material_weakness=True."""
    filing = _make_filing(
        mda_text=(
            "Our auditor identified a material weakness in internal "
            "controls over financial reporting."
        ),
    )
    delta = parse_ten_q(filing)
    assert delta is not None
    assert delta.has_material_weakness is True


def test_diff_ten_q_detects_new_mda_paragraph():
    """Current has paragraph absent from prior → added."""
    current = parse_ten_q(_make_filing(
        mda_text=(
            "Revenue up 5%.\n\n"
            "Cash position strong.\n\n"
            "We acquired Foo Corp in March for $500M."   # new
        ),
    ))
    prior = parse_ten_q(_make_filing(
        mda_text=(
            "Revenue up 5%.\n\n"
            "Cash position strong."
        ),
    ))
    diffed = diff_ten_q(current, prior)
    assert len(diffed.mda_added_paragraphs) == 1
    assert "Foo Corp" in diffed.mda_added_paragraphs[0]


def test_diff_ten_q_no_prior_returns_current_unchanged():
    """No prior 10-Q (first quarter after deploy) → added empty,
    count=0. Card section renders nothing meaningful but doesn't
    crash."""
    current = parse_ten_q(_make_filing(
        mda_text="Revenue up 5%.\n\nNew acquisition completed.",
    ))
    diffed = diff_ten_q(current, None)
    assert diffed is current
    assert diffed.mda_added_paragraphs == []
    assert diffed.new_risk_factor_count == 0


def test_diff_ten_q_truncates_added_paragraph_display():
    """Added paragraph display capped at 100 chars per row + max 5
    rows total — ⑧ card vertical real estate limit."""
    long_para = "Acquisition details: " + ("x" * 200)
    current = parse_ten_q(_make_filing(mda_text=long_para))
    prior = parse_ten_q(_make_filing(mda_text="unrelated prior content"))
    diffed = diff_ten_q(current, prior)
    assert len(diffed.mda_added_paragraphs) == 1
    assert len(diffed.mda_added_paragraphs[0]) <= 100


def test_diff_ten_q_caps_added_at_5_paragraphs():
    """≥5 new paragraphs → only first 5 rendered."""
    paras = "\n\n".join(f"New paragraph {i} about Foo." for i in range(10))
    current = parse_ten_q(_make_filing(mda_text=paras))
    prior = parse_ten_q(_make_filing(mda_text="unrelated"))
    diffed = diff_ten_q(current, prior)
    assert len(diffed.mda_added_paragraphs) == 5


def test_diff_ten_q_new_risk_factor_count():
    """Headings present in current but not prior → counted."""
    current = parse_ten_q(_make_filing(
        mda_text="MD&A content.",
        risk_factors_text=(
            "## Cybersecurity risk\n"
            "Some text.\n\n"
            "## Supply chain disruption\n"
            "More text.\n\n"
            "## New AI regulatory risk\n"
            "Newly added."
        ),
    ))
    prior = parse_ten_q(_make_filing(
        mda_text="MD&A content.",
        risk_factors_text=(
            "## Cybersecurity risk\n"
            "Some text.\n\n"
            "## Supply chain disruption\n"
            "More text."
        ),
    ))
    diffed = diff_ten_q(current, prior)
    assert diffed.new_risk_factor_count == 1


# ---------------------------------------------------------------------------
# insider_digest tests
# ---------------------------------------------------------------------------

class _FakeArchive:
    """Mocks the get_form4_pushes_in_window helper that build_weekly_digest
    prefers. Skips the raw SQL path covered by Stage 4 cron integration
    tests."""
    def __init__(self, rows: list[dict]) -> None:
        self._rows = list(rows)

    def get_form4_pushes_in_window(self, week_start, week_end):
        return [
            r for r in self._rows
            if week_start <= r.get("ts", "")[:10] <= week_end
        ]


def _push_row(title: str, ts: str, tickers: str = "") -> dict:
    return {"id": 1, "ts": ts, "agent": "sec",
            "title": title, "tickers": tickers, "text_html": ""}


def test_weekly_digest_aggregates_ticker_counts():
    """5 ⑫ pushes across 3 tickers + 2 directions → totals correct."""
    rows = [
        _push_row("Form 4 · NVDA · 买入", "2026-06-08T17:45:01"),
        _push_row("Form 4 · NVDA · 买入", "2026-06-09T17:45:02"),
        _push_row("Form 4 · AAPL · 卖出", "2026-06-10T17:45:03"),
        _push_row("Form 4 · AAPL · 卖出", "2026-06-11T17:45:04"),
        _push_row("Form 4 · TSLA · 买入", "2026-06-12T17:45:05"),
    ]
    summary = build_weekly_digest(
        _FakeArchive(rows),
        week_start_iso="2026-06-08", week_end_iso="2026-06-12",
    )
    assert summary.total_push_count == 5
    assert summary.purchase_push_count == 3
    assert summary.sale_push_count == 2
    assert summary.cluster_purchase_count == 0
    assert summary.cluster_sale_count == 0
    assert summary.total_tickers_active == 3
    # by_ticker sorted by count desc
    assert summary.by_ticker == {"NVDA": 2, "AAPL": 2, "TSLA": 1}


def test_weekly_digest_cluster_titles_parsed_correctly():
    """Cluster cards have different title shape with '集群' prefix."""
    rows = [
        _push_row("Form 4 集群 · ARM · purchase", "2026-06-08T17:45:01"),
        _push_row("Form 4 集群 · NVDA · sale",    "2026-06-09T17:45:02"),
    ]
    summary = build_weekly_digest(
        _FakeArchive(rows),
        week_start_iso="2026-06-08", week_end_iso="2026-06-12",
    )
    assert summary.cluster_purchase_count == 1
    assert summary.cluster_sale_count == 1
    assert summary.purchase_push_count == 0
    assert summary.sale_push_count == 0
    assert summary.total_push_count == 2


def test_weekly_digest_empty_week_returns_quiet():
    """No ⑫ pushes → is_quiet_week=True, all counts zero."""
    summary = build_weekly_digest(
        _FakeArchive([]),
        week_start_iso="2026-06-08", week_end_iso="2026-06-12",
    )
    assert summary.is_quiet_week is True
    assert summary.total_push_count == 0
    assert summary.total_tickers_active == 0
    assert summary.unusual_tickers == []


def test_weekly_digest_unusual_activity_threshold():
    """Ticker with ≥3 pushes flagged as unusual."""
    rows = [_push_row(f"Form 4 · NVDA · 买入",
                      f"2026-06-{8+i:02d}T17:45:01") for i in range(4)]
    rows.append(_push_row("Form 4 · AAPL · 买入", "2026-06-10T17:45:01"))
    summary = build_weekly_digest(
        _FakeArchive(rows),
        week_start_iso="2026-06-08", week_end_iso="2026-06-12",
    )
    assert summary.unusual_tickers == ["NVDA"]   # AAPL only 1 push
    assert summary.by_ticker["NVDA"] == 4


def test_weekly_digest_skips_unrecognized_titles():
    """Non-Form-4 push titles silently skipped (defensive against
    schema drift or test seam leakage)."""
    rows = [
        _push_row("Form 4 · NVDA · 买入", "2026-06-09T17:45:01"),
        _push_row("Form 4 · 5G stuff",    "2026-06-09T17:45:02"),  # malformed
        _push_row("portfolio_risk · ...", "2026-06-10T17:45:01"),  # wrong agent path
    ]
    summary = build_weekly_digest(
        _FakeArchive(rows),
        week_start_iso="2026-06-08", week_end_iso="2026-06-12",
    )
    assert summary.total_push_count == 1
    assert summary.by_ticker == {"NVDA": 1}


def test_default_week_window_friday_anchored():
    """Friday today → Mon-Fri window of this week."""
    # 2026-06-12 is a Friday (verified: 2026-06-08 = Monday)
    start, end = default_week_window("2026-06-12")
    assert start == "2026-06-08"
    assert end == "2026-06-12"


def test_default_week_window_midweek_anchored():
    """Wednesday today → still anchors to this week's Mon-Fri."""
    # 2026-06-10 is a Wednesday
    start, end = default_week_window("2026-06-10")
    assert start == "2026-06-08"
    assert end == "2026-06-12"


# ---------------------------------------------------------------------------
# Surface contract
# ---------------------------------------------------------------------------

def test_ten_q_parser_public_surface():
    from v2.sec import ten_q_parser
    for name in ("TenQDelta", "parse_ten_q", "diff_ten_q"):
        assert hasattr(ten_q_parser, name)
        assert name in ten_q_parser.__all__


def test_insider_digest_public_surface():
    from v2.sec import insider_digest
    for name in (
        "WeeklyInsiderSummary", "build_weekly_digest", "default_week_window",
    ):
        assert hasattr(insider_digest, name)
        assert name in insider_digest.__all__
