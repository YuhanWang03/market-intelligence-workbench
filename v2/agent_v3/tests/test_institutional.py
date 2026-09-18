"""institutional.manager_portfolio under V3: alias resolution and field-level 13F evidence."""
import pytest

from v2.agent_v3.institutional import CAPABILITY, portfolio_envelope, register_institutional, resolve_manager, tracked_filings
from v2.agent_v3.tools import Registry


def _filing(quarter, period, filed, value, accession):
    return {"cik": "1067983", "accession": accession, "manager_name": "Berkshire Hathaway", "quarter": quarter, "filing_date": filed, "period_of_report": period, "portfolio_value": value, "n_positions": 2}


def _position(accession, cusip, ticker, issuer, shares, value):
    return {"accession": accession, "cusip": cusip, "ticker": ticker, "issuer_name": issuer, "shares": shares, "market_value": value}


CURRENT = (_filing("2026-Q1", "2026-03-31", "2026-05-15", 100_000_000_000, "acc-1"), [
    _position("acc-1", "037833100", "AAPL", "APPLE INC", 300_000_000, 60_000_000_000),
    _position("acc-1", "060505104", "BAC", "BANK OF AMERICA", 200_000_000, 10_000_000_000),
    _position("acc-1", "166764100", "CVX", "CHEVRON", 100_000_000, 15_000_000_000),
])
PREVIOUS = (_filing("2025-Q4", "2025-12-31", "2026-02-14", 95_000_000_000, "acc-0"), [
    _position("acc-0", "037833100", "AAPL", "APPLE INC", 300_000_000, 58_000_000_000),
    _position("acc-0", "060505104", "BAC", "BANK OF AMERICA", 700_000_000, 30_000_000_000),
    _position("acc-0", "22160K105", "COST", "COSTCO", 20_000_000, 7_000_000_000),
])


@pytest.mark.parametrize("asked, expected", [
    ("巴菲特", "Berkshire Hathaway"), ("伯克希尔最近", "Berkshire Hathaway"), ("Warren Buffett", "Berkshire Hathaway"), ("BRK", "Berkshire Hathaway"),
    ("木头姐", "ARK Investment Mgmt"), ("Cathie Wood", "ARK Investment Mgmt"), ("Michael Burry", "Scion Asset Mgmt"), ("Renaissance Technologies", "Renaissance Technologies"),
])
def test_manager_aliases_resolve_in_both_languages(asked, expected):
    assert resolve_manager(asked)[1] == expected


def test_unknown_manager_is_none_not_a_guess():
    assert resolve_manager("张三") is None and resolve_manager("") is None


def test_portfolio_evidence_is_one_item_per_fact():
    envelope = portfolio_envelope("1067983", "Berkshire Hathaway", [CURRENT, PREVIOUS], top=2, run_id="run-x")
    assert envelope.status.value == "completed"
    by_id = {item.id: item for item in envelope.evidence}
    summary = by_id["13f-1067983-2026-Q1-portfolio"]
    assert summary.value == 100_000_000_000 and summary.as_of == "2026-03-31" and summary.period == "2026-Q1"
    assert "2026-05-15" in summary.claim and "3 个持仓" in summary.claim
    top = by_id["13f-1067983-2026-Q1-pos-AAPL"]
    assert top.metadata["rank"] == 1 and top.metadata["weight_pct"] == 60.0 and "300,000,000 股" in top.claim
    assert "13f-1067983-2026-Q1-pos-CVX" in by_id and "13f-1067983-2026-Q1-pos-BAC" not in by_id, "top=2 lists the two largest positions only"
    changes = {item.metadata["ticker"]: item for item in envelope.evidence if item.metric == "position_change_value"}
    assert changes["COST"].metadata["change_type"] == "exit" and changes["CVX"].metadata["change_type"] == "new"
    assert changes["BAC"].metadata["change_type"] == "decrease" and changes["BAC"].value == -20_000_000_000
    assert all(item.source_id == "sec_13f_hr" and item.producer_run_id == "run-x" and "CIK=1067983" in item.source_url for item in envelope.evidence)
    assert any("45 天" in note for note in envelope.limitations)


def test_single_filing_is_partial_without_a_comparison():
    envelope = portfolio_envelope("1067983", "Berkshire Hathaway", [CURRENT])
    assert envelope.status.value == "partial_data"
    assert not [item for item in envelope.evidence if item.metric == "position_change_value"]
    assert any("只有一期" in note for note in envelope.limitations)


def test_handler_resolves_alias_validates_arguments_and_reports_unknown_managers():
    seen = []
    def reader(cik, name, n):
        seen.append((cik, name, n))
        return [CURRENT, PREVIOUS], "tracked"
    registry = Registry()
    register_institutional(registry, reader=reader)
    assert registry.registered(CAPABILITY)
    handler = registry.handlers[CAPABILITY]
    result = handler({"manager": "巴菲特", "top": 1}, None)
    assert seen == [("1067983", "Berkshire Hathaway", 2)]
    assert result.subject == "Berkshire Hathaway" and result.metrics["top_listed"] == 1
    assert any("跟踪库" in note for note in result.limitations)
    unknown = handler({"manager": "张三"}, None)
    assert unknown.status.value == "partial_data" and "Berkshire Hathaway" in unknown.limitations[0]
    from v2.agent_v2.models import PlanTask
    registry.validate(PlanTask("m", CAPABILITY, {"manager": "buffett", "top": 5}))
    with pytest.raises(Exception):
        registry.validate(PlanTask("m", CAPABILITY, {"manager": "buffett", "top": 99}))


def test_reader_failure_is_a_partial_error_not_a_crash():
    registry = Registry()
    def reader(cik, name, n):
        raise ConnectionError("EDGAR unreachable")
    register_institutional(registry, reader=reader)
    result = registry.handlers[CAPABILITY]({"manager": "Ackman"}, None)
    assert result.status.value == "partial_error" and "EDGAR unreachable" in result.errors[0]


def test_tracked_filings_reads_the_tracker_schema(tmp_path):
    import sqlite3
    from v2.institutional import tracker
    db = tmp_path / "edgar.db"
    conn = sqlite3.connect(db)
    conn.executescript(tracker._SCHEMA)
    for filing, positions in (CURRENT, PREVIOUS):
        conn.execute("INSERT INTO filings VALUES (?,?,?,?,?,?,?,?)", (filing["cik"], filing["accession"], filing["manager_name"], filing["quarter"], filing["filing_date"], filing["period_of_report"], filing["portfolio_value"], filing["n_positions"]))
        conn.executemany("INSERT INTO positions VALUES (?,?,?,?,?,?)", [(p["accession"], p["cusip"], p["ticker"], p["issuer_name"], p["shares"], p["market_value"]) for p in positions])
    conn.commit(); conn.close()
    rows = tracked_filings("1067983", 2, db_path=db)
    assert [f["quarter"] for f, _ in rows] == ["2026-Q1", "2025-Q4"] and len(rows[0][1]) == 3
    assert tracked_filings("0000000", 2, db_path=db) == []


def test_a_store_missing_positions_is_not_complete():
    from v2.agent_v3.institutional import filing_is_complete

    filing = {**CURRENT[0], "n_positions": 90, "portfolio_value": 263_000_000_000}
    assert not filing_is_complete(filing, CURRENT[1])  # 3 of 90 stored, 85B of 263B
    assert filing_is_complete({**CURRENT[0], "n_positions": 3, "portfolio_value": 85_000_000_000}, CURRENT[1])


def test_incomplete_store_falls_back_to_edgar_and_is_flagged_when_edgar_fails(monkeypatch):
    from v2.agent_v3 import institutional

    incomplete = [({**CURRENT[0], "n_positions": 90, "portfolio_value": 263_000_000_000}, CURRENT[1])]
    monkeypatch.setattr(institutional, "tracked_filings", lambda cik, n, db_path=None: incomplete)
    monkeypatch.setattr(institutional, "live_filings", lambda cik, name, n_filings=2: [CURRENT, PREVIOUS])
    assert institutional.default_reader("1067983", "Berkshire Hathaway")[1] == "edgar"

    def edgar_down(cik, name, n_filings=2):
        raise ConnectionError("offline")
    monkeypatch.setattr(institutional, "live_filings", edgar_down)
    filings, provenance = institutional.default_reader("1067983", "Berkshire Hathaway")
    assert provenance == "tracked_incomplete"
    envelope = portfolio_envelope("1067983", "Berkshire Hathaway", filings, provenance=provenance)
    assert envelope.status.value == "partial_data"
    assert any("90 个持仓中的 3 个" in limit for limit in envelope.limitations)
