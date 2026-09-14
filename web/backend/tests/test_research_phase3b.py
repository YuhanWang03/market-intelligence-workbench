from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from v2.research.cache import ResearchCache
from v2.research.depth import classify_catalysts, compare_risk_sections, parse_sec_filings, sanitize_error
from v2.research.engine import ResearchEngine
from v2.research.store import ENGINE_VERSION, FEATURE_FREEZE, ResearchStore
from web.backend.tests.test_research import FakeFD, FakePrices, fake_services


class FilingObject:
    def __init__(self, items: dict[str, str]):
        self.items = items

    def get_item(self, name: str, markdown: bool = True):
        return self.items.get(name)

    def get_item_with_part(self, part: str, name: str, markdown: bool = True):
        return self.items.get(f"{part}:{name}")


class Filing:
    def __init__(self, form: str, date: str, accession: str, items: dict[str, str]):
        self.form, self.filing_date, self.accession_number = form, date, accession
        self.homepage_url = f"https://sec.example/{accession}"
        self.cik = "1"
        self._object = FilingObject(items)

    def obj(self):
        return self._object


def test_sec_depth_and_risk_delta_are_structured():
    current = "## Supply dependency\n" + "We depend on a sole supplier and shortages may materially affect operations. " * 8
    previous = "## Supply dependency\n" + "We depend on suppliers. " * 5
    result = parse_sec_filings("AAPL", {"10-K": [Filing("10-K", "2026-01-01", "new", {"Item 1": "We sell devices and services and generate revenue from products.", "Item 1A": current, "Item 7": "We expect revenue to increase next year."}), Filing("10-K", "2025-01-01", "old", {"Item 1A": previous})], "10-Q": [], "8-K": []})
    assert result["findings"]
    assert result["company_profile"]["description"]
    assert result["guidance"][0]["status"] == "NOT_COMPARABLE"
    assert result["guidance"][0]["group"] == "outlook"
    assert result["risk_factor_changes"][0]["change_type"] == "EXPANDED"


def test_catalyst_v2_deduplicates_and_classifies():
    rows = classify_catalysts([
        {"title": "Company wins major contract", "event_date": "2026-09-05", "source_kind": "NEWS"},
        {"title": "Company wins major contract", "event_date": "2026-09-05", "source_kind": "NEWS"},
        {"title": "Quarterly earnings", "event_date": "2026-10-01", "source_kind": "EVENT"},
        {"title": "CEO speaks at conference", "event_date": "2026-09-06", "source_kind": "NEWS"},
    ])
    assert [row["item_type"] for row in rows] == ["CATALYST", "EVENT", "NEWS"]


def test_annual_report_never_uses_properties_as_management_discussion():
    result = parse_sec_filings('AAPL', {'10-K': [Filing('10-K', '2026-01-01', 'x', {
        'Part I:Item 2': 'We expect capital expenditure of $5 billion for our facilities next year.'
    })]})
    assert result['guidance'] == []
    assert not any(row['category'] == 'MD&A' for row in result['findings'])


def test_eight_k_guidance_retains_section_and_source(monkeypatch):
    from v2.sec import eight_k_parser
    from v2.research.expectations import prepare_expectations
    filing = Filing('8-K', '2026-08-26', 'release', {})
    filing._object.text = 'Item 2.02 Results. We forecast revenue of $40 billion for next quarter. Item 9.01 Exhibits.'
    monkeypatch.setattr(eight_k_parser, 'parse_eight_k_filing', lambda *args: SimpleNamespace(items=[]))
    result = parse_sec_filings('NVDA', {'8-K': [filing]})
    row = result['guidance'][0]
    assert row['filing_type'] == '8-K'
    assert row['source_section'] == '2.02'
    assert row['source_url'] == filing.homepage_url
    presented = prepare_expectations({'modules': {'expectations': {'details': {'guidance': [row]}}}})
    assert presented['modules']['expectations']['details']['guidance'][0]['source_section'] == '2.02'


def test_peer_preferences_persist_and_invalidate_cache(tmp_path: Path):
    path = tmp_path / "research.db"
    store = ResearchStore(path)
    store.put_module_cache("AAPL", "valuation", {"status": "COMPLETED"}, 3600)
    store.set_peer_preference("AAPL", "DELL", "ADD")
    reopened = ResearchStore(path)
    assert reopened.peer_preferences("AAPL")["added"] == ["DELL"]
    assert reopened.get_module_cache("AAPL", "valuation") is None
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"


@pytest.mark.parametrize("ticker,expected", [("AAPL", "MSFT"), ("NVDA", "AMD"), ("JPM", "BAC"), ("XOM", "CVX"), ("TSLA", "GM")])
def test_five_industries_have_distinct_peer_sets(tmp_path: Path, ticker: str, expected: str):
    result = ResearchEngine(fd=FakeFD(), prices=FakePrices(), cache=ResearchCache(tmp_path / f"{ticker}.db"), services=fake_services()).run(ticker, modules=["valuation"])
    assert expected in result["modules"]["valuation"]["details"]["peer_set"]


def test_error_redaction_and_engine_version():
    cleaned = sanitize_error("request failed?api_key=secret123&token=abc")
    assert "secret123" not in cleaned and "abc" not in cleaned
    assert ENGINE_VERSION == "research-v1.0"
    assert FEATURE_FREEZE is True
