"""Point-in-time S&P 500 membership from the Wikipedia change table."""

from __future__ import annotations

import json

import pytest

from v2.screening import universes as U

HTML = """
<table class="wikitable sortable" id="changes"><tbody>
<tr><th rowspan="2">Effective Date</th><th colspan="2">Added</th><th colspan="2">Removed</th><th rowspan="2">Reason</th></tr>
<tr><th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th></tr>
<tr><td>July 9, 2025</td><td>BRK.B</td><td>Berkshire</td><td></td><td></td><td>x</td></tr>
<tr><td>June 30, 2025</td><td></td><td></td><td>ANSS</td><td>Ansys</td><td>Acquired by Synopsys.</td></tr>
<tr><td>March 24, 2025</td><td>DASH</td><td>DoorDash</td><td>BWA</td><td>BorgWarner</td><td>Market cap.</td></tr>
<tr><td rowspan="2">September 23, 2024</td><td>PLTR</td><td>Palantir</td><td>AAL</td><td>American Airlines</td><td rowspan="2">Market cap change.</td></tr>
<tr><td>DELL</td><td>Dell</td><td>ETSY</td><td>Etsy</td></tr>
</tbody></table>
<table class="wikitable"><tr><th>Symbol</th><th>Security</th></tr><tr><td>AAPL</td><td>Apple</td></tr></table>
"""


def test_parse_changes_handles_rowspan_dates_empty_cells_and_dotted_tickers():
    rows = U.parse_changes(HTML)
    assert [(r["date"], r["added"], r["removed"]) for r in rows] == [
        ("2025-07-09", "BRK.B", None), ("2025-06-30", None, "ANSS"), ("2025-03-24", "DASH", "BWA"),
        ("2024-09-23", "PLTR", "AAL"), ("2024-09-23", "DELL", "ETSY"),
    ]
    assert rows[2]["added_name"] == "DoorDash" and rows[2]["removed_name"] == "BorgWarner"
    assert U.parse_constituents(HTML, U._HEADERS) == ["AAPL"]  # the constituent parser still picks the other table


def test_parse_changes_tolerates_footnotes_nbsp_and_iso_dates():
    html = """<table><tr><th>Date</th><th colspan=2>Added</th><th colspan=2>Removed</th><th>Reason</th></tr>
    <tr><th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th></tr>
    <tr><td>September&nbsp;22, 2025<sup>[3]</sup></td><td>APP</td><td>AppLovin</td><td>MKTX</td><td>MarketAxess</td><td>r</td></tr>
    <tr><td>2025-07-23</td><td>BLK</td><td>BlackRock</td><td>WBA</td><td>Walgreens</td><td>r</td></tr></table>"""
    rows = U.parse_changes(html)
    assert [(r["date"], r["added"], r["removed"]) for r in rows] == [("2025-09-22", "APP", "MKTX"), ("2025-07-23", "BLK", "WBA")]
    assert U.describe_changes(html)[0].startswith("candidate table 0: 4 rows")


def test_join_dates_give_additions_only_membership(tmp_path, monkeypatch):
    html = """<table><tr><th>Symbol</th><th>Security</th><th>GICS Sector</th><th>Date added</th><th>CIK</th></tr>
    <tr><td>AAPL</td><td>Apple</td><td>IT</td><td>1982-11-30</td><td>1</td></tr>
    <tr><td>PLTR</td><td>Palantir</td><td>IT</td><td>2024-09-23</td><td>2</td></tr>
    <tr><td>APP</td><td>AppLovin</td><td>IT</td><td>2025-09-22</td><td>3</td></tr>
    <tr><td>BRK.B</td><td>Berkshire</td><td>Fin</td><td></td><td>4</td></tr></table>"""
    pairs = U.parse_constituents_with_dates(html, U._HEADERS)
    assert pairs == [("AAPL", "1982-11-30"), ("PLTR", "2024-09-23"), ("APP", "2025-09-22"), ("BRK.B", None)]
    assert U.parse_constituents(html, U._HEADERS) == ["AAPL", "PLTR", "APP", "BRK.B"]

    path = tmp_path / "universes.json"
    monkeypatch.setattr(U, "DATA_PATH", path)
    path.write_text(json.dumps({"sp500": {"tickers": [t for t, _ in pairs], "as_of": "2026-09-01", "date_added": {t: d for t, d in pairs if d}}}))
    assert U.membership_mode("sp500") == "additions"
    assert U.members_at("sp500", "2024-09-10") == (["AAPL", "BRK.B"], True)      # PLTR / APP had not joined; unknown date is kept
    assert U.members_at("sp500", "2025-01-01") == (["AAPL", "PLTR", "BRK.B"], True)
    assert U.members_at("sp500", "2026-09-01")[0] == ["AAPL", "PLTR", "APP", "BRK.B"]
    assert U.membership_lookup("sp500")("2024-09-10") == ["AAPL", "BRK.B"]
    assert U.universe_status()["sp500"]["membership"] == "additions"


def test_members_at_rewinds_todays_list_through_the_changes(tmp_path, monkeypatch):
    path = tmp_path / "universes.json"
    monkeypatch.setattr(U, "DATA_PATH", path)
    assert U.members_at("sp500", "2024-06-01") == (U._dedupe(U.SP500), False)  # no history → today's list, flagged
    assert U.membership_lookup("sp500") is None and U.membership_mode("sp500") == "none"

    today = ["AAPL", "PLTR", "DELL", "DASH", "BRK.B"]
    path.write_text(json.dumps({"sp500": {"tickers": today, "as_of": "2025-09-01", "changes": U.parse_changes(HTML)}}))
    assert U.members_at("sp500", "2025-09-01") == (today, True)                       # on/after the snapshot date
    assert U.members_at("sp500", "2025-07-01") == (sorted(["AAPL", "PLTR", "DELL", "DASH"]), True)   # before BRK.B joined
    assert U.members_at("sp500", "2025-06-01") == (sorted(["AAPL", "PLTR", "DELL", "DASH", "ANSS"]), True)  # ANSS still in
    assert U.members_at("sp500", "2024-09-01") == (sorted(["AAPL", "AAL", "ETSY", "BWA", "ANSS"]), True)     # before the 2024 changes
    lookup = U.membership_lookup("sp500")
    assert lookup is not None and "AAL" in lookup("2024-09-01") and "PLTR" not in lookup("2024-09-01")
    status = U.universe_status()["sp500"]
    assert status["changes"] == 5 and status["history_from"] == "2024-09-23"
    with pytest.raises(KeyError):
        U.members_at("nope", "2024-01-01")
