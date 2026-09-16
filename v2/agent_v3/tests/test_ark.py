"""etf.ark_activity under V3: fund resolution, field-level snapshot evidence, day-over-day changes."""
import pytest

from v2.agent_v3.ark import CAPABILITY, activity_envelope, register_ark, resolve_fund, tracked_snapshot
from v2.agent_v3.tools import Registry


def _row(ticker, company, shares, value, weight, etf="ARKK", date="2026-09-15"):
    return {"etf": etf, "date": date, "ticker": ticker, "cusip": None, "company": company, "shares": shares, "market_value": value, "weight_pct": weight}


TODAY = [_row("TSLA", "TESLA INC", 2_000_000, 500_000_000, 10.0), _row("COIN", "COINBASE", 1_000_000, 300_000_000, 6.0), _row("PLTR", "PALANTIR", 3_000_000, 200_000_000, 4.0)]
YESTERDAY = ("2026-09-12", [_row("TSLA", "TESLA INC", 2_000_000, 480_000_000, 9.8, date="2026-09-12"), _row("COIN", "COINBASE", 1_500_000, 420_000_000, 8.5, date="2026-09-12"), _row("ROKU", "ROKU INC", 800_000, 60_000_000, 1.2, date="2026-09-12")])


@pytest.mark.parametrize("asked, expected", [("ARKK", "ARKK"), ("arkk", "ARKK"), ("ARK", "ARKK"), ("木头姐", "ARKK"), ("Cathie Wood", "ARKK"), ("ARKG", "ARKG"), ("ARKQ", None), ("SPY", None), ("", None)])
def test_fund_resolution(asked, expected):
    assert resolve_fund(asked) == expected


def test_activity_evidence_is_one_item_per_fact():
    envelope = activity_envelope("ARKK", "2026-09-15", TODAY, YESTERDAY, top=2, run_id="run-y")
    assert envelope.status.value == "completed" and envelope.as_of == "2026-09-15"
    by_id = {item.id: item for item in envelope.evidence}
    summary = by_id["ark-ARKK-2026-09-15-snapshot"]
    assert summary.value == 1_000_000_000 and "3 个持仓" in summary.claim and summary.metadata["date_basis"] == "publication"
    top = by_id["ark-ARKK-2026-09-15-pos-TSLA"]
    assert top.value == 10.0 and top.unit == "%" and "2,000,000 股" in top.claim and top.metadata["rank"] == 1
    assert "ark-ARKK-2026-09-15-pos-PLTR" not in by_id, "top=2"
    changes = {item.metadata["ticker"]: item for item in envelope.evidence if item.metric == "shares_change"}
    assert changes["PLTR"].metadata["change_type"] == "new" and changes["ROKU"].metadata["change_type"] == "exit"
    assert changes["COIN"].metadata["change_type"] == "decrease" and changes["COIN"].value == -500_000 and changes["COIN"].metadata["shares_diff_pct"] == -33.33
    assert "TSLA" not in changes, "unchanged share count is below the 1% threshold"
    assert all(item.source_id == "ark_daily_holdings_csv" and item.producer_run_id == "run-y" and "ARKK" in item.source_url for item in envelope.evidence)
    assert any("2026-09-12" in note for note in envelope.limitations)


def test_no_prior_snapshot_is_partial_data():
    envelope = activity_envelope("ARKG", "2026-09-15", TODAY, None)
    assert envelope.status.value == "partial_data"
    assert not [item for item in envelope.evidence if item.metric == "shares_change"]
    assert any("没有更早" in note for note in envelope.limitations)


def test_handler_wires_reader_prior_and_saver():
    saved, read = [], []
    registry = Registry()
    register_ark(registry, reader=lambda symbol: read.append(symbol) or ("2026-09-15", TODAY, "live"), prior=lambda symbol, before: YESTERDAY if before == "2026-09-15" else None, saver=lambda symbol, date, rows: saved.append((symbol, date, len(rows))))
    assert registry.registered(CAPABILITY)
    result = registry.handlers[CAPABILITY]({"symbol": "木头姐", "top": 1}, None)
    assert read == ["ARKK"] and saved == [("ARKK", "2026-09-15", 3)]
    assert result.metrics == {"holdings_value_usd": 1_000_000_000.0, "position_count": 3, "top_listed": 1, "significant_changes": 3}
    unsupported = registry.handlers[CAPABILITY]({"symbol": "ARKQ"}, None)
    assert unsupported.status.value == "partial_data" and "ARKK" in unsupported.limitations[0]
    from v2.agent_v2.models import PlanTask
    registry.validate(PlanTask("a", CAPABILITY, {"symbol": "ARKK", "top": 5}))
    with pytest.raises(Exception):
        registry.validate(PlanTask("a", CAPABILITY, {"symbol": "ARKK", "extra": 1}))


def test_tracked_fallback_is_not_saved_again_and_is_labelled():
    saved = []
    registry = Registry()
    register_ark(registry, reader=lambda symbol: ("2026-09-12", YESTERDAY[1], "tracked"), prior=lambda symbol, before: None, saver=lambda *a: saved.append(a))
    result = registry.handlers[CAPABILITY]({"symbol": "ARKK"}, None)
    assert saved == [] and result.metadata["provenance"] == "tracked"
    assert any("跟踪库" in note for note in result.limitations)


def test_reader_failure_is_a_partial_error():
    registry = Registry()
    def reader(symbol):
        raise LookupError("ARK CSV unavailable and no tracked snapshot")
    register_ark(registry, reader=reader, saver=False)
    result = registry.handlers[CAPABILITY]({"symbol": "ARKW"}, None)
    assert result.status.value == "partial_error" and "no tracked snapshot" in result.errors[0]


def test_tracked_snapshot_reads_the_tracker_schema(tmp_path):
    import sqlite3
    from v2.etf import tracker
    db = tmp_path / "etf.db"
    conn = sqlite3.connect(db)
    conn.executescript(tracker._SCHEMA)
    for date, rows in (("2026-09-15", TODAY), YESTERDAY):
        conn.executemany("INSERT INTO snapshots VALUES (?,?,?,?,?,?,?,?)", [("ARKK", date, r["ticker"], r["cusip"], r["company"], r["shares"], r["market_value"], r["weight_pct"]) for r in rows])
    conn.commit(); conn.close()
    latest = tracked_snapshot("ARKK", db_path=db)
    assert latest[0] == "2026-09-15" and len(latest[1]) == 3
    assert tracked_snapshot("ARKK", before="2026-09-15", db_path=db)[0] == "2026-09-12"
    assert tracked_snapshot("ARKF", db_path=db) is None
