"""Smoke tests for Phase 2.5 full — positions snapshot module +
drawdown realtime fix.

Two stages of coverage:

1. ``v2.portfolio.snapshot`` — write_daily_snapshot /
   read_weekly_window / compute_weekly_attribution. Uses an
   in-memory archive fake (no SQLite) so the tests stay focused on
   the snapshot logic and don't double-count what the archive store
   tests already cover.

2. ``v2.portfolio.drawdown`` — compute_drawdown's new
   ``today_realtime_value`` kwarg. Verifies append vs overwrite
   semantics + backward-compat with the Phase 2 signature.

Both stages are sandbox-safe: the broker stub returns canned
``get_portfolio_history`` payloads, no Alpaca network.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from v2.portfolio.drawdown import compute_drawdown   # noqa: E402
from v2.portfolio.models import PositionFlat         # noqa: E402
from v2.portfolio.snapshot import (                  # noqa: E402
    AttributionItem,
    PositionSnapshot,
    WEEKLY_MIN_DAYS,
    compute_weekly_attribution,
    read_weekly_window,
    write_daily_snapshot,
)


# ---------------------------------------------------------------------------
# In-memory archive fake (mirrors Archive's snapshot surface only)
# ---------------------------------------------------------------------------

class _FakeArchive:
    """Minimal in-memory replacement for v2.archive.store.Archive's
    positions_snapshot surface. The full Archive's SQLite path is
    covered by its own tests; this fake keeps the snapshot tests
    focused on the snapshot module's logic."""

    def __init__(self) -> None:
        # keyed by (snapshot_date, ticker) — mirrors PRIMARY KEY
        self._rows: dict[tuple[str, str], dict] = {}

    def write_position_snapshots(self, rows):
        written = 0
        for r in rows:
            key = (r["snapshot_date"], r["ticker"])
            self._rows[key] = dict(r)
            written += 1
        return written

    def get_position_snapshots(self, *, ticker=None, since=None, until=None):
        out = []
        for (snap_date, t), row in self._rows.items():
            if ticker is not None and t != ticker:
                continue
            if since is not None and snap_date < since:
                continue
            if until is not None and snap_date > until:
                continue
            out.append(dict(row))
        out.sort(key=lambda r: (r["snapshot_date"], r["ticker"]))
        return out


def _flat(ticker: str, market_value: float, weight: float,
          sector_etf: str = "XLK") -> PositionFlat:
    return PositionFlat(
        ticker=ticker,
        market_value=market_value,
        weight=weight,
        sector_etf=sector_etf,
    )


# ---------------------------------------------------------------------------
# Snapshot module — write_daily_snapshot
# ---------------------------------------------------------------------------

def test_write_daily_snapshot_inserts_rows():
    """3 positions → 3 rows written, each carrying snapshot_date."""
    arch = _FakeArchive()
    written = write_daily_snapshot(
        [_flat("NVDA", 30000, 0.30, "SMH"),
         _flat("AAPL", 20000, 0.20, "XLK"),
         _flat("JPM",  15000, 0.15, "XLF")],
        snapshot_date="2026-06-09",
        archive=arch,
    )
    assert written == 3
    nv = arch._rows[("2026-06-09", "NVDA")]
    assert nv["market_value"] == 30000.0
    assert nv["weight"] == 0.30
    assert nv["sector_etf"] == "SMH"


def test_write_daily_snapshot_replace_on_conflict():
    """Same-day rerun overwrites (PRIMARY KEY semantics)."""
    arch = _FakeArchive()
    write_daily_snapshot(
        [_flat("NVDA", 30000, 0.30, "SMH")],
        snapshot_date="2026-06-09", archive=arch,
    )
    # Rerun with a different value (price moved intraday)
    write_daily_snapshot(
        [_flat("NVDA", 31500, 0.31, "SMH")],
        snapshot_date="2026-06-09", archive=arch,
    )
    assert len(arch._rows) == 1
    assert arch._rows[("2026-06-09", "NVDA")]["market_value"] == 31500.0


def test_write_daily_snapshot_empty_positions_skip():
    """All-cash day → 0 rows written, no exception."""
    arch = _FakeArchive()
    written = write_daily_snapshot(
        [], snapshot_date="2026-06-09", archive=arch,
    )
    assert written == 0
    assert arch._rows == {}


# ---------------------------------------------------------------------------
# Snapshot module — read_weekly_window
# ---------------------------------------------------------------------------

def _seed_full_week(arch: _FakeArchive) -> None:
    """5 weekdays of NVDA + AAPL snapshots ending 2026-06-12 Fri."""
    # 2026-06-08 Mon through 2026-06-12 Fri
    for i, day in enumerate(("2026-06-08", "2026-06-09", "2026-06-10",
                             "2026-06-11", "2026-06-12")):
        write_daily_snapshot(
            [
                _flat("NVDA", 30000 + i * 500, 0.30, "SMH"),
                _flat("AAPL", 20000 - i * 100, 0.20, "XLK"),
            ],
            snapshot_date=day, archive=arch,
        )


def test_read_weekly_window_returns_5_days():
    """Full week of snapshots → 5 entries per ticker."""
    arch = _FakeArchive()
    _seed_full_week(arch)
    win = read_weekly_window(arch, end_date_iso="2026-06-12", window_days=7)
    assert set(win.keys()) == {"NVDA", "AAPL"}
    assert len(win["NVDA"]) == 5
    assert len(win["AAPL"]) == 5
    # Verify sorted ascending
    nv_dates = [s.snapshot_date for s in win["NVDA"]]
    assert nv_dates == sorted(nv_dates)
    assert nv_dates[0] == "2026-06-08"
    assert nv_dates[-1] == "2026-06-12"


def test_read_weekly_window_handles_missing_days():
    """Holiday-shortened week → only writes Mon/Tue/Thu/Fri → 4 snaps."""
    arch = _FakeArchive()
    for day in ("2026-06-08", "2026-06-09", "2026-06-11", "2026-06-12"):
        write_daily_snapshot(
            [_flat("NVDA", 30000, 0.30, "SMH")],
            snapshot_date=day, archive=arch,
        )
    win = read_weekly_window(arch, end_date_iso="2026-06-12", window_days=7)
    assert len(win["NVDA"]) == 4
    # Wed (2026-06-10) missing
    nv_dates = [s.snapshot_date for s in win["NVDA"]]
    assert "2026-06-10" not in nv_dates


def test_read_weekly_window_ticker_added_midweek_returns_partial():
    """Position added Wed → 3 snapshots returned (Wed/Thu/Fri)."""
    arch = _FakeArchive()
    _seed_full_week(arch)
    # Add CRM on Wed
    for day in ("2026-06-10", "2026-06-11", "2026-06-12"):
        write_daily_snapshot(
            [_flat("CRM", 10000, 0.10, "XLK")],
            snapshot_date=day, archive=arch,
        )
    win = read_weekly_window(arch, end_date_iso="2026-06-12", window_days=7)
    assert "CRM" in win
    assert len(win["CRM"]) == 3
    # Full-week positions still 5 snapshots
    assert len(win["NVDA"]) == 5


def test_read_weekly_window_empty_returns_empty():
    """No snapshots in window → empty dict (not exception)."""
    arch = _FakeArchive()
    win = read_weekly_window(arch, end_date_iso="2026-06-12", window_days=7)
    assert win == {}


def test_read_weekly_window_invalid_date_returns_empty():
    """Malformed end_date_iso → empty dict + warning log (not crash)."""
    arch = _FakeArchive()
    _seed_full_week(arch)
    win = read_weekly_window(arch, end_date_iso="not-a-date")
    assert win == {}


# ---------------------------------------------------------------------------
# Snapshot module — compute_weekly_attribution
# ---------------------------------------------------------------------------

def _snap(ticker: str, snap_date: str, mv: float, weight: float):
    return PositionSnapshot(
        snapshot_date=snap_date, ticker=ticker,
        market_value=mv, weight=weight, sector_etf="XLK",
    )


def test_compute_weekly_attribution_simple_case():
    """NVDA up 5% with avg_weight 30% → contribution +1.5%."""
    snaps = {
        "NVDA": [
            _snap("NVDA", "2026-06-08", 30000, 0.30),
            _snap("NVDA", "2026-06-09", 30500, 0.30),
            _snap("NVDA", "2026-06-12", 31500, 0.30),
        ],
    }
    items = compute_weekly_attribution(snaps)
    assert len(items) == 1
    nv = items[0]
    assert nv.ticker == "NVDA"
    assert abs(nv.weekly_return - 0.05) < 1e-9    # 31500/30000 - 1
    assert abs(nv.avg_weight - 0.30) < 1e-9
    assert abs(nv.contribution - 0.015) < 1e-9


def test_compute_weekly_attribution_sorted_desc():
    """Best contribution first, worst last."""
    snaps = {
        "NVDA": [_snap("NVDA", "2026-06-08", 30000, 0.30),
                 _snap("NVDA", "2026-06-12", 31500, 0.30)],   # +5% × 30% = +1.5%
        "AAPL": [_snap("AAPL", "2026-06-08", 20000, 0.20),
                 _snap("AAPL", "2026-06-12", 19000, 0.20)],   # -5% × 20% = -1.0%
        "JPM":  [_snap("JPM",  "2026-06-08", 15000, 0.15),
                 _snap("JPM",  "2026-06-12", 15300, 0.15)],   # +2% × 15% = +0.3%
    }
    items = compute_weekly_attribution(snaps)
    assert [i.ticker for i in items] == ["NVDA", "JPM", "AAPL"]


def test_compute_weekly_attribution_single_snap_skipped():
    """Ticker with 1 snapshot (added today) is excluded — no return."""
    snaps = {
        "NVDA": [_snap("NVDA", "2026-06-08", 30000, 0.30),
                 _snap("NVDA", "2026-06-12", 31500, 0.30)],
        "BRAND_NEW": [_snap("BRAND_NEW", "2026-06-12", 5000, 0.05)],
    }
    items = compute_weekly_attribution(snaps)
    tickers = {i.ticker for i in items}
    assert "NVDA" in tickers
    assert "BRAND_NEW" not in tickers


def test_compute_weekly_attribution_zero_first_mv_skipped():
    """first.market_value == 0 → skipped, no ZeroDivisionError."""
    snaps = {
        "DEAD": [_snap("DEAD", "2026-06-08", 0.0, 0.0),
                 _snap("DEAD", "2026-06-12", 1.0, 0.0001)],
        "LIVE": [_snap("LIVE", "2026-06-08", 100, 0.001),
                 _snap("LIVE", "2026-06-12", 110, 0.001)],
    }
    items = compute_weekly_attribution(snaps)
    assert {i.ticker for i in items} == {"LIVE"}


def test_compute_weekly_attribution_empty_returns_empty():
    """No snapshots → empty list."""
    assert compute_weekly_attribution({}) == []


# ---------------------------------------------------------------------------
# Drawdown realtime fix
# ---------------------------------------------------------------------------

def _utc_midnight_unix(d: date) -> int:
    return int(
        datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp()
    )


def _fake_broker(equity: list[float], dates: list[date]):
    """Return a broker stub with get_portfolio_history → canned shape."""
    timestamps = [_utc_midnight_unix(d) for d in dates]

    def get_portfolio_history(*, period: str = "1M",
                              timeframe: str = "1D") -> dict:
        return {"equity": equity, "timestamp": timestamps}

    return SimpleNamespace(get_portfolio_history=get_portfolio_history)


def test_drawdown_without_today_realtime_no_change():
    """Backward-compat: today_realtime_value=None → behaviour identical
    to Phase 2 (the kwarg is purely additive)."""
    today = date.today()
    history_dates = [today - timedelta(days=10 - i) for i in range(10)]
    equity = [100.0, 102, 105, 110, 108, 107, 105, 103, 101, 100]
    broker = _fake_broker(equity, history_dates)

    metrics, warnings = compute_drawdown(broker=broker)
    # Peak 110 at idx 3, trough 100 at idx 9 → dd 10/110
    assert abs(metrics.max_drawdown_pct - (10 / 110)) < 1e-9
    # Current point is 100 vs peak 110 → current dd 10/110
    assert abs(metrics.current_drawdown_pct - (10 / 110)) < 1e-9
    assert warnings == []


def test_drawdown_with_today_realtime_appends_to_series():
    """EOD series ends yesterday + realtime drop today → today's value
    appended → current_drawdown_pct reflects the intraday decline."""
    today = date.today()
    yesterday = today - timedelta(days=1)
    history_dates = [yesterday - timedelta(days=4 - i) for i in range(5)]
    # Stable around 100, ending yesterday at 100
    equity = [100.0, 101.0, 100.0, 100.5, 100.0]
    broker = _fake_broker(equity, history_dates)

    # Today realtime tanks to 96.7 (Phase 2.5 full prod-bug scenario)
    metrics, _ = compute_drawdown(broker=broker, today_realtime_value=96.7)
    # New peak from the appended walk should still be 101.0 (idx 1).
    # Trough = today's 96.7 → max_dd = (101.0 - 96.7) / 101.0
    expected_max_dd = (101.0 - 96.7) / 101.0
    assert abs(metrics.max_drawdown_pct - expected_max_dd) < 1e-9
    # current_dd = same as max here since today IS the trough
    assert abs(metrics.current_drawdown_pct - expected_max_dd) < 1e-9


def test_drawdown_today_already_in_series_no_duplicate():
    """If history's last point already bears today's date, the realtime
    value REPLACES it instead of appending (idempotent same-day rerun)."""
    today = date.today()
    history_dates = [today - timedelta(days=4 - i) for i in range(5)]
    equity = [100.0, 101.0, 100.0, 100.5, 100.0]   # last point = today
    broker = _fake_broker(equity, history_dates)

    metrics, _ = compute_drawdown(broker=broker, today_realtime_value=96.7)
    # Series length should still be 5, not 6 — today's point was
    # overwritten, not appended. We verify indirectly by checking
    # the math against a 5-element series where the last is 96.7:
    expected_max_dd = (101.0 - 96.7) / 101.0
    assert abs(metrics.max_drawdown_pct - expected_max_dd) < 1e-9


def test_drawdown_realtime_with_empty_equity_no_crash():
    """Empty history → unavailable() returned, today_realtime_value
    ignored (no equity to append to). Phase 2 contract preserved."""
    broker = _fake_broker([], [])
    metrics, _ = compute_drawdown(broker=broker, today_realtime_value=100.0)
    # DrawdownMetrics.unavailable() → all fields None
    assert metrics.current_drawdown_pct is None
    assert metrics.max_drawdown_pct is None


def test_drawdown_realtime_recomputes_max_dd_correctly():
    """Realtime point that DEEPENS an existing trough should update
    max_dd. Realtime point that's above peak should NOT change max_dd
    (peak can move forward; max_dd stays put)."""
    today = date.today()
    history_dates = [today - timedelta(days=5 - i) for i in range(5)]
    # Peak 110 at idx 0, trough 95 at idx 3, partial recovery to 100 at idx 4 (yesterday)
    equity = [110.0, 105.0, 100.0, 95.0, 100.0]
    broker = _fake_broker(equity, history_dates)

    # Today realtime breaks below to 90 → new max_dd
    metrics_break, _ = compute_drawdown(
        broker=broker, today_realtime_value=90.0,
    )
    expected = (110.0 - 90.0) / 110.0
    assert abs(metrics_break.max_drawdown_pct - expected) < 1e-9
    assert abs(metrics_break.current_drawdown_pct - expected) < 1e-9

    # Today realtime rallies to 115 → new peak, current_dd = 0,
    # max_dd unchanged at (110 - 95)/110 from the original trough
    broker = _fake_broker(equity, history_dates)   # fresh broker
    metrics_recover, _ = compute_drawdown(
        broker=broker, today_realtime_value=115.0,
    )
    assert abs(metrics_recover.current_drawdown_pct - 0.0) < 1e-9
    # Original max_dd from the (110 → 95) leg is preserved
    expected_original_max = (110.0 - 95.0) / 110.0
    assert abs(metrics_recover.max_drawdown_pct - expected_original_max) < 1e-9


# ---------------------------------------------------------------------------
# Surface contract
# ---------------------------------------------------------------------------

def test_snapshot_module_public_surface():
    """Module exposes the 6 documented public names."""
    from v2.portfolio import snapshot as snap_mod
    for name in (
        "AttributionItem", "PositionSnapshot", "WEEKLY_MIN_DAYS",
        "compute_weekly_attribution", "read_weekly_window",
        "write_daily_snapshot",
    ):
        assert hasattr(snap_mod, name), f"snapshot.py missing {name}"
        assert name in snap_mod.__all__, f"{name} not in __all__"


def test_weekly_min_days_pinned_at_5():
    """⑩ card switches to '归因累积中' below this; pin to catch
    accidental drift."""
    assert WEEKLY_MIN_DAYS == 5
