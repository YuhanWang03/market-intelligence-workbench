"""Stage-1 smoke test for v2/sec/.

Sandbox CANNOT reach SEC EDGAR (Stage 0 confirmed 403 from this IP).
All tests mock edgartools objects via ``SimpleNamespace`` constructions
matching the real SDK's class shapes:

  - EightK exposes ``.items``
  - Form4 exposes ``.insider_name``, ``.reporting_owners``, ``.to_dataframe()``
  - Filing exposes ``.obj()``, ``.accession_number``, ``.filing_date``,
    ``.cik``, ``.form``

Reference shapes verified in Phase 3 Stage 0 task 2.

Coverage maps to Stage 1 prompt's 13 listed test cases plus 2 bonus
edge cases (15 total). Each test runs ≤ 1 second; full suite < 5s.
"""

from __future__ import annotations

import sys
import traceback
from types import SimpleNamespace

import pandas as pd


def _section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _fake_8k_filing(
    *,
    ticker: str = "HPE",
    acc: str = "0000123-26-000001",
    filing_date: str = "2026-06-04",
    form: str = "8-K",
    items: list[str] | None = None,
) -> SimpleNamespace:
    """Build a fake edgartools 8-K Filing object."""
    items = items if items is not None else ["ITEM 5.02: Departure of CEO"]
    return SimpleNamespace(
        ticker=ticker,
        accession_number=acc,
        accession_no=acc,        # SDK has both
        filing_date=filing_date,
        cik="0001645590",
        form=form,
        obj=lambda: SimpleNamespace(items=list(items)),
    )


def _fake_form4_filing(
    *,
    ticker: str = "ARM",
    acc: str = "0000999-26-000001",
    filing_date: str = "2026-06-04",
    insider_name: str = "Rene Haas",
    insider_title: str = "Chief Executive Officer",
    is_director: bool = False,
    is_officer: bool = True,
    transactions: list[dict] | None = None,
) -> SimpleNamespace:
    """Build a fake edgartools Form 4 Filing object.

    Each transactions[i] is a dict with the PascalCase column names
    edgartools actually emits via ``to_dataframe()``.
    """
    if transactions is None:
        transactions = [{
            "Security": "Common Stock", "Date": filing_date,
            "Shares": 1000.0, "Remaining": 50000.0, "Price": 150.0,
            "AcquiredDisposed": "A", "DirectIndirect": "D",
            "form": "4", "Code": "P", "footnotes": "",
        }]
    df = pd.DataFrame(transactions)

    # ReportingOwners: a single-owner shape with .relationship.officer_title
    relationship = SimpleNamespace(
        officer_title=insider_title,
        is_officer=is_officer,
        is_director=is_director,
        is_ten_percent_owner=False,
    )
    owner = SimpleNamespace(relationship=relationship)
    reporting_owners = SimpleNamespace(owners=[owner])

    form4_obj = SimpleNamespace(
        insider_name=insider_name,
        reporting_owners=reporting_owners,
        to_dataframe=lambda: df.copy(),
    )

    return SimpleNamespace(
        ticker=ticker,
        accession_number=acc,
        accession_no=acc,
        filing_date=filing_date,
        cik="0001973239",
        form="4",
        obj=lambda: form4_obj,
    )


# ---------------------------------------------------------------------------
# 8-K parser tests
# ---------------------------------------------------------------------------

def test_eight_k_extract_items_correct_codes():
    from v2.sec.eight_k_parser import parse_eight_k_filing
    from v2.sec.models import SecFiling

    filing = _fake_8k_filing(items=[
        "ITEM 5.02: Departure of Directors or Certain Officers",
        "ITEM 1.01 Entry into a Material Definitive Agreement",
        "Item 9.01 Financial Statements and Exhibits",
    ])
    sec = SecFiling("HPE", "0001645590", "8-K", "2026-06-04", "acc-1", False)
    event = parse_eight_k_filing(filing, sec)
    assert event is not None
    codes = sorted(it.code for it in event.items)
    assert codes == ["1.01", "5.02", "9.01"], f"got {codes}"
    print(f"  ok   3 items parsed: {codes}")


def test_eight_k_multi_item_aggregation():
    """HPE Stage 0 case — one filing with 5 items in one card."""
    from v2.sec.eight_k_parser import parse_eight_k_filing
    from v2.sec.models import SecFiling

    filing = _fake_8k_filing(items=[
        "ITEM 1.01: ...", "ITEM 2.02: ...", "ITEM 5.02: ...",
        "ITEM 7.01: ...", "ITEM 9.01: ...",
    ])
    sec = SecFiling("HPE", "0001645590", "8-K", "2026-06-04", "acc-1", False)
    event = parse_eight_k_filing(filing, sec)
    assert event is not None
    codes = sorted(it.code for it in event.items)
    assert codes == ["1.01", "2.02", "5.02", "7.01", "9.01"]
    # max_priority_tier: 5.02 = P1 (base, no LLM extraction here yet),
    # 1.01 = P1, 2.02 = P2, 7.01 = P2, 9.01 = P3. max = P1.
    assert event.max_priority_tier == "P1"
    # has_earnings_overlap because 2.02 is present (but it's NOT 2.02-only)
    assert event.has_earnings_overlap is True
    assert event.is_2_02_only is False
    print(f"  ok   5-item HPE filing → max_tier={event.max_priority_tier}, "
          f"has_earnings_overlap=True, is_2_02_only=False")


def test_eight_k_2_02_only_marked_overlap():
    from v2.sec.eight_k_parser import parse_eight_k_filing
    from v2.sec.models import SecFiling

    # Earnings-only filing: 2.02 + 9.01 (administrative exhibit)
    filing = _fake_8k_filing(items=[
        "ITEM 2.02: Results of Operations",
        "ITEM 9.01: Financial Statements and Exhibits",
    ])
    sec = SecFiling("MRVL", "0001835632", "8-K", "2026-06-04", "acc-1", False)
    event = parse_eight_k_filing(filing, sec)
    assert event is not None
    assert event.has_earnings_overlap is True
    assert event.is_2_02_only is True
    print("  ok   2.02 + 9.01 → is_2_02_only=True (cron will skip)")


def test_eight_k_amendment_flag():
    """8-K/A amendments must be tagged so priority gets -5 nudge."""
    from v2.sec.eight_k_parser import parse_eight_k_filing
    from v2.sec.models import SecFiling

    # SecFiling itself tracks is_amendment — verified in pipeline build.
    sec = SecFiling("XYZ", "0001234567", "8-K/A", "2026-06-04", "acc-x", True)
    filing = _fake_8k_filing(items=["ITEM 5.02: ..."])
    event = parse_eight_k_filing(filing, sec)
    assert event.filing.is_amendment is True
    assert event.filing.form == "8-K/A"
    print("  ok   8-K/A flagged is_amendment=True")


def test_eight_k_unknown_item_defaults_p3():
    """SEC adds new items occasionally — graceful fall-through."""
    from v2.sec.eight_k_parser import parse_eight_k_filing
    from v2.sec.models import SecFiling

    filing = _fake_8k_filing(items=["ITEM 9.99: Future Item Not Yet Defined"])
    sec = SecFiling("XYZ", "0001234567", "8-K", "2026-06-04", "acc-y", False)
    event = parse_eight_k_filing(filing, sec)
    assert event is not None
    assert event.items[0].code == "9.99"
    assert event.items[0].priority_tier == "P3"
    assert event.items[0].description == "其他 item"
    print("  ok   unknown 9.99 → P3 default + 其他 item description")


def test_get_item_text_finds_5_02_section_in_multi_item():
    """Stage 2 pickup — extract the 5.02 body from a multi-item 8-K.

    Real 8-K filings concatenate items in document order with the
    "Item X.YY" headers. Our extractor must isolate just the 5.02
    section so the LLM doesn't get confused by adjacent items'
    discussion.
    """
    from v2.sec.eight_k_parser import get_item_text

    # Simulated EightK obj with .text — mirrors HPE multi-item filing
    document = (
        "Item 1.01 Entry into a Material Definitive Agreement\n"
        "On May 30, 2026, the Company entered into a credit facility...\n"
        "\n"
        "Item 2.02 Results of Operations and Financial Condition\n"
        "Quarterly earnings of $X.YY per share, attached as Exhibit 99.1.\n"
        "\n"
        "Item 5.02 Departure of Directors or Certain Officers; Election...\n"
        "Effective June 4, 2026, John Smith resigned as Chief Executive Officer.\n"
        "The Board appointed Jane Doe as interim CEO effective June 5, 2026.\n"
        "\n"
        "Item 7.01 Regulation FD Disclosure\n"
        "A press release announcing the foregoing is attached as Exhibit 99.2.\n"
        "\n"
        "Item 9.01 Financial Statements and Exhibits\n"
        "99.1 Press release dated June 4, 2026\n"
    )
    eight_k = SimpleNamespace(text=document)

    extracted = get_item_text(eight_k, "5.02")
    # The section header should be present
    assert extracted.lower().startswith("item 5.02"), (
        f"section should start with header, got: {extracted[:80]!r}"
    )
    # The 5.02 body content must be present
    assert "John Smith resigned" in extracted
    assert "Jane Doe" in extracted
    # The 7.01 section must NOT bleed in
    assert "Regulation FD" not in extracted
    assert "Exhibit 99.2" not in extracted
    # Length check — typical 5.02 sections are 200-2000 chars
    assert 50 < len(extracted) < 1000
    print(f"  ok   5.02 section extracted: {len(extracted)} chars, no 7.01 bleed")


def test_get_item_text_missing_code_returns_empty():
    """Item not in document → empty string, no exception."""
    from v2.sec.eight_k_parser import get_item_text

    eight_k = SimpleNamespace(text="Item 1.01 Material Agreement\n...\n")
    assert get_item_text(eight_k, "5.02") == ""
    assert get_item_text(eight_k, "9.99") == ""
    print("  ok   missing item code → '' (no exception)")


def test_get_item_text_no_text_attr_returns_empty():
    """Defensive: SDK could return EightK without .text — handle gracefully."""
    from v2.sec.eight_k_parser import get_item_text

    # No text attribute at all
    eight_k = SimpleNamespace()
    assert get_item_text(eight_k, "5.02") == ""
    # Empty text
    eight_k2 = SimpleNamespace(text="")
    assert get_item_text(eight_k2, "5.02") == ""
    # None text
    eight_k3 = SimpleNamespace(text=None)
    assert get_item_text(eight_k3, "5.02") == ""
    print("  ok   missing/empty/None .text → '' (no AttributeError)")


# ---------------------------------------------------------------------------
# Form 4 parser tests
# ---------------------------------------------------------------------------

def test_form4_to_dataframe_uses_Code_column():
    """REGRESSION TEST for Stage 0 Check 4 bug.

    edgartools to_dataframe() uses PascalCase 'Code', NOT 'transaction_code'.
    Asserting this here so a future SDK refactor that flips back will fail
    loudly instead of silently emptying every Form 4 scan."""
    from v2.sec.form4_parser import _DF_CODE_COL, parse_form4_filing
    from v2.sec.models import SecFiling

    assert _DF_CODE_COL == "Code", "Stage 0 column-name bug regressed"

    filing = _fake_form4_filing(transactions=[
        {"Security": "Common", "Date": "2026-06-04",
         "Shares": 100.0, "Remaining": 50000.0, "Price": 100.0,
         "AcquiredDisposed": "A", "DirectIndirect": "D",
         "form": "4", "Code": "P", "footnotes": ""},
    ])
    sec = SecFiling("ARM", "x", "4", "2026-06-04", "acc-z", False)
    txs = parse_form4_filing(filing, sec)
    assert len(txs) == 1
    assert txs[0].transaction_code == "P"
    print(f"  ok   _DF_CODE_COL=={_DF_CODE_COL!r}, parser returned {len(txs)} P-tx")


def test_form4_classify_PSAMFGC():
    """Mix of all 7 codes — verify signal vs noise classification."""
    from v2.sec.form4_parser import parse_form4_filing
    from v2.sec.models import (
        NOISE_TRANSACTION_CODES, SIGNAL_TRANSACTION_CODES, SecFiling,
    )

    transactions = []
    for code in ("P", "S", "A", "M", "F", "G", "C"):
        transactions.append({
            "Security": "Common", "Date": "2026-06-04",
            "Shares": 100.0, "Remaining": 50000.0, "Price": 50.0,
            "AcquiredDisposed": "A", "DirectIndirect": "D",
            "form": "4", "Code": code, "footnotes": "",
        })
    filing = _fake_form4_filing(transactions=transactions)
    sec = SecFiling("ARM", "x", "4", "2026-06-04", "acc-mix", False)
    txs = parse_form4_filing(filing, sec)

    by_code = {tx.transaction_code: tx for tx in txs}
    assert by_code["P"].is_signal and not by_code["P"].is_noise
    assert by_code["S"].is_signal and not by_code["S"].is_noise
    for code in ("A", "M", "F", "G", "C"):
        assert by_code[code].is_noise and not by_code[code].is_signal, (
            f"{code} should be noise"
        )
    assert {"P", "S"} == SIGNAL_TRANSACTION_CODES
    assert {"A", "M", "F", "G", "C"} == NOISE_TRANSACTION_CODES
    print("  ok   P/S = signal; A/M/F/G/C = noise (matches Stage 0 calibration)")


def test_form4_10b5_1_detection():
    """Footnote regex must catch typical 10b5-1 phrasings."""
    from v2.sec.form4_parser import _detect_10b5_1

    assert _detect_10b5_1("Sales executed under Rule 10b5-1 trading plan")
    assert _detect_10b5_1("10b5-1 plan adopted 2025-12-15")
    assert _detect_10b5_1("Pre-arranged plan dated...")
    assert _detect_10b5_1("Pursuant to a trading plan")
    assert not _detect_10b5_1("Routine open-market sale")
    assert not _detect_10b5_1("")
    assert not _detect_10b5_1(None)
    print("  ok   10b5-1 regex catches Rule/plan/pre-arranged variants")


def test_form4_insider_role_lookup():
    """Title pattern matcher: CEO/CFO/COO/Director/None."""
    from v2.sec.insider_role import _classify_title

    assert _classify_title("Chief Executive Officer") == "CEO"
    assert _classify_title("CEO") == "CEO"
    assert _classify_title("President and CEO") == "CEO"
    assert _classify_title("Chief Financial Officer") == "CFO"
    assert _classify_title("EVP and CFO") == "CFO"
    assert _classify_title("Chief Operating Officer") == "COO"
    assert _classify_title("Director") == "Director"
    # Director also wins for plain director when titled
    assert _classify_title("Lead Independent Director") == "Director"
    # General Counsel + VP
    assert _classify_title("Senior VP, General Counsel") == "GC"
    # Unknown title
    assert _classify_title("Senior VP, Engineering") is None
    assert _classify_title("") is None
    assert _classify_title(None) is None
    print("  ok   title classifier covers CEO/CFO/COO/Director/GC + None")


def test_form4_insider_role_via_form4_object():
    """End-to-end: full fake Form 4 → lookup_insider_role returns 'CEO'."""
    from v2.sec.insider_role import lookup_insider_role

    filing = _fake_form4_filing(insider_title="Chief Executive Officer")
    role = lookup_insider_role(filing.obj())
    assert role == "CEO"

    # Director-only owner (no officer_title)
    filing2 = _fake_form4_filing(
        insider_title="", is_director=True, is_officer=False,
    )
    role2 = lookup_insider_role(filing2.obj())
    assert role2 == "Director"

    print(f"  ok   end-to-end: CEO + Director-only both resolve correctly")


# ---------------------------------------------------------------------------
# Cluster tests
# ---------------------------------------------------------------------------

def test_cluster_3_purchases_same_day():
    """3 distinct insiders all buy same ticker same day → 1 cluster."""
    from v2.sec.cluster import find_clusters
    from v2.sec.models import Form4Transaction, SecFiling

    sec = SecFiling("NVDA", "0001045810", "4", "2026-06-04", "acc-cluster", False)

    txs = []
    for name in ("Alice", "Bob", "Carol"):
        txs.append(Form4Transaction(
            filing=sec, insider_name=name, insider_role="Director",
            transaction_code="P", transaction_date="2026-06-04",
            shares=1000.0, price=100.0, transaction_usd=100_000.0,
            is_10b5_1=False, direct_indirect="D",
        ))

    clusters = find_clusters(txs)
    assert len(clusters) == 1
    c = clusters[0]
    assert c.transaction_count == 3
    assert c.direction == "purchase"
    assert c.total_usd == 300_000.0
    assert c.insider_names == ["Alice", "Bob", "Carol"]
    print(f"  ok   3 distinct insiders P → 1 cluster total ${c.total_usd:,.0f}")


def test_cluster_ignore_AMF_codes():
    """A/M/F transactions don't contribute to clusters."""
    from v2.sec.cluster import find_clusters
    from v2.sec.models import Form4Transaction, SecFiling

    sec = SecFiling("NVDA", "x", "4", "2026-06-04", "acc-noise", False)
    txs = []
    # 5 insiders all on the same day, but with non-signal codes
    for code in ("A", "M", "F", "G", "C"):
        txs.append(Form4Transaction(
            filing=sec, insider_name=f"Person {code}",
            insider_role="Officer",
            transaction_code=code, transaction_date="2026-06-04",
            shares=1000.0, price=100.0, transaction_usd=100_000.0,
            is_10b5_1=False, direct_indirect="D",
        ))
    clusters = find_clusters(txs)
    assert clusters == [], "A/M/F/G/C must not form clusters"
    print("  ok   5 same-day A/M/F/G/C → 0 clusters (noise filtered)")


def test_cluster_single_insider_multi_lot_not_a_cluster():
    """One actor with 3 lots same day = 1 decision, not a cluster."""
    from v2.sec.cluster import find_clusters
    from v2.sec.models import Form4Transaction, SecFiling

    sec = SecFiling("NVDA", "x", "4", "2026-06-04", "acc-single", False)
    # Same insider, 3 lots
    txs = [
        Form4Transaction(
            filing=sec, insider_name="Alice", insider_role="Director",
            transaction_code="P", transaction_date="2026-06-04",
            shares=1000.0 + i, price=100.0, transaction_usd=100_000.0 + i * 100,
            is_10b5_1=False, direct_indirect="D",
        )
        for i in range(3)
    ]
    clusters = find_clusters(txs)
    assert clusters == [], "Single-actor multi-lot is not a cluster"
    print("  ok   1 insider × 3 lots → 0 clusters (distinct-actor count = 1)")


def test_cluster_mixed_directions_dont_merge():
    """3 P + 2 S on same day → only the P cluster fires."""
    from v2.sec.cluster import find_clusters
    from v2.sec.models import Form4Transaction, SecFiling

    sec = SecFiling("NVDA", "x", "4", "2026-06-04", "acc-mixed", False)
    txs = []
    for name in ("A1", "A2", "A3"):
        txs.append(Form4Transaction(
            filing=sec, insider_name=name, insider_role="Director",
            transaction_code="P", transaction_date="2026-06-04",
            shares=1000.0, price=100.0, transaction_usd=100_000.0,
            is_10b5_1=False, direct_indirect="D",
        ))
    for name in ("B1", "B2"):
        txs.append(Form4Transaction(
            filing=sec, insider_name=name, insider_role="Officer",
            transaction_code="S", transaction_date="2026-06-04",
            shares=500.0, price=100.0, transaction_usd=50_000.0,
            is_10b5_1=False, direct_indirect="D",
        ))
    clusters = find_clusters(txs)
    assert len(clusters) == 1
    assert clusters[0].direction == "purchase"
    print("  ok   3P + 2S → 1 purchase cluster (2S below threshold)")


# ---------------------------------------------------------------------------
# Pipeline orchestration tests
# ---------------------------------------------------------------------------

def test_pipeline_empty_ticker_silent():
    """Stage 0 calibration #5 — empty results don't add to warnings."""
    from v2.sec.pipeline import run_sec_scan

    def empty_client(ticker, form, since, until):
        return []

    def noop_extractor(text):
        return {"departures": [], "appointments": [], "has_senior_exec": False}

    result = run_sec_scan(
        ["AAPL", "IVV", "MU"], "2026-06-04",
        edgar_client=empty_client, llm_extractor=noop_extractor,
    )
    assert result.eight_k_events == []
    assert result.form4_signal_transactions == []
    assert result.form4_clusters == []
    assert result.form4_noise_summary == {}
    # CRITICAL — Stage 0 said empty days don't log warnings
    assert result.warnings == [], f"empty days must not log: got {result.warnings}"
    print("  ok   3 empty tickers → 0 warnings (clean-day silent)")


def test_pipeline_edgar_unavailable_per_ticker_skip():
    """One ticker fails → its failure logged; other tickers continue."""
    from v2.sec.pipeline import run_sec_scan

    def partial_client(ticker, form, since, until):
        if ticker == "BROKEN":
            raise RuntimeError("EDGAR 503")
        if ticker == "AAPL" and form == "8-K":
            return [_fake_8k_filing(ticker="AAPL", acc="aapl-1",
                                    items=["ITEM 1.01: Material Agreement"])]
        return []

    def noop_extractor(text):
        return {"departures": [], "appointments": [], "has_senior_exec": False}

    result = run_sec_scan(
        ["AAPL", "BROKEN", "NVDA"], "2026-06-04",
        edgar_client=partial_client, llm_extractor=noop_extractor,
    )
    # AAPL event survives
    assert len(result.eight_k_events) == 1
    assert result.eight_k_events[0].filing.ticker == "AAPL"
    # BROKEN failures appear in warnings (8-K + Form 4 both attempted)
    broken_warnings = [w for w in result.warnings if "BROKEN" in w]
    assert len(broken_warnings) >= 1, f"expected BROKEN warning: {result.warnings}"
    print(f"  ok   1 ticker fails (BROKEN), AAPL succeeds; warnings={len(broken_warnings)}")


def test_pipeline_5_02_llm_failure_conservative():
    """LLM raises → fallback {has_senior_exec=True} → 5.02 escalates to P0."""
    from v2.sec.pipeline import run_sec_scan

    def client_with_5_02(ticker, form, since, until):
        if form != "8-K":
            return []
        return [_fake_8k_filing(
            ticker=ticker, acc="hpe-5-02",
            items=["ITEM 5.02: Officer Change"],
        )]

    def boom_extractor(text):
        raise RuntimeError("DeepSeek 503")

    result = run_sec_scan(
        ["HPE"], "2026-06-04",
        edgar_client=client_with_5_02, llm_extractor=boom_extractor,
    )
    assert len(result.eight_k_events) == 1
    event = result.eight_k_events[0]
    item_5_02 = next(it for it in event.items if it.code == "5.02")
    # Fallback path: has_senior_exec=True → escalate to P0
    assert item_5_02.priority_tier == "P0", (
        f"conservative fallback must escalate to P0; got {item_5_02.priority_tier}"
    )
    assert item_5_02.extracted_meta.get("has_senior_exec") is True
    # And a warning was logged about the LLM failure
    llm_warnings = [w for w in result.warnings if "5.02" in w and "HPE" in w]
    assert len(llm_warnings) == 1
    print(f"  ok   LLM fails → 5.02 escalated to P0 conservatively + warning logged")


def test_pipeline_form4_noise_aggregation():
    """A/M/F/G codes batch into noise_summary, NOT into signal_transactions."""
    from v2.sec.pipeline import run_sec_scan

    def client_with_form4(ticker, form, since, until):
        if form != "4":
            return []
        # Mix of signal + noise codes for ARM
        return [_fake_form4_filing(
            ticker=ticker, acc=f"{ticker}-mixed",
            transactions=[
                # 1 P (signal), 2 A (noise), 1 M (noise), 1 F (noise)
                {"Security": "Common", "Date": "2026-06-04",
                 "Shares": 1000.0, "Remaining": 50000.0, "Price": 100.0,
                 "AcquiredDisposed": "A", "DirectIndirect": "D",
                 "form": "4", "Code": "P", "footnotes": ""},
                {"Security": "Common", "Date": "2026-06-04",
                 "Shares": 500.0, "Remaining": 50000.0, "Price": 100.0,
                 "AcquiredDisposed": "A", "DirectIndirect": "D",
                 "form": "4", "Code": "A", "footnotes": ""},
                {"Security": "Common", "Date": "2026-06-04",
                 "Shares": 500.0, "Remaining": 50000.0, "Price": 100.0,
                 "AcquiredDisposed": "A", "DirectIndirect": "D",
                 "form": "4", "Code": "A", "footnotes": ""},
                {"Security": "Common", "Date": "2026-06-04",
                 "Shares": 100.0, "Remaining": 50000.0, "Price": 100.0,
                 "AcquiredDisposed": "A", "DirectIndirect": "D",
                 "form": "4", "Code": "M", "footnotes": ""},
                {"Security": "Common", "Date": "2026-06-04",
                 "Shares": 50.0, "Remaining": 50000.0, "Price": 100.0,
                 "AcquiredDisposed": "D", "DirectIndirect": "D",
                 "form": "4", "Code": "F", "footnotes": ""},
            ],
        )]

    def noop_extractor(text):
        return {"departures": [], "appointments": [], "has_senior_exec": False}

    result = run_sec_scan(
        ["ARM"], "2026-06-04",
        edgar_client=client_with_form4, llm_extractor=noop_extractor,
    )

    # 1 signal (P), 4 noise (2A + 1M + 1F)
    assert len(result.form4_signal_transactions) == 1
    assert result.form4_signal_transactions[0].transaction_code == "P"
    assert result.form4_noise_summary == {"ARM": {"A": 2, "M": 1, "F": 1}}
    print(f"  ok   1P signal, noise_summary={result.form4_noise_summary['ARM']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    failed: list[str] = []
    for name, fn in [
        # 8-K parser
        ("eight_k_extract_items",       test_eight_k_extract_items_correct_codes),
        ("eight_k_multi_item_aggreg",   test_eight_k_multi_item_aggregation),
        ("eight_k_2_02_only_overlap",   test_eight_k_2_02_only_marked_overlap),
        ("eight_k_amendment_flag",      test_eight_k_amendment_flag),
        ("eight_k_unknown_item_p3",     test_eight_k_unknown_item_defaults_p3),
        ("get_item_text_5_02_section",  test_get_item_text_finds_5_02_section_in_multi_item),
        ("get_item_text_missing_empty", test_get_item_text_missing_code_returns_empty),
        ("get_item_text_no_text_attr",  test_get_item_text_no_text_attr_returns_empty),
        # Form 4 parser
        ("form4_Code_column_fix",       test_form4_to_dataframe_uses_Code_column),
        ("form4_classify_PSAMFGC",      test_form4_classify_PSAMFGC),
        ("form4_10b5_1_detection",      test_form4_10b5_1_detection),
        ("form4_insider_role_titles",   test_form4_insider_role_lookup),
        ("form4_insider_role_e2e",      test_form4_insider_role_via_form4_object),
        # Cluster
        ("cluster_3_purchases_same_day",test_cluster_3_purchases_same_day),
        ("cluster_ignore_noise_codes",  test_cluster_ignore_AMF_codes),
        ("cluster_single_actor_skip",   test_cluster_single_insider_multi_lot_not_a_cluster),
        ("cluster_mixed_directions",    test_cluster_mixed_directions_dont_merge),
        # Pipeline
        ("pipeline_empty_silent",       test_pipeline_empty_ticker_silent),
        ("pipeline_per_ticker_skip",    test_pipeline_edgar_unavailable_per_ticker_skip),
        ("pipeline_5_02_llm_fallback",  test_pipeline_5_02_llm_failure_conservative),
        ("pipeline_noise_aggregation",  test_pipeline_form4_noise_aggregation),
    ]:
        _section(name)
        try:
            fn()
        except Exception:
            traceback.print_exc()
            failed.append(name)

    print()
    if failed:
        print(f"FAILED ({len(failed)}): {failed}")
        return 1
    print("ALL SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
