"""Bot responder tests for /macro and /cpi / /fomc / /yields
(Phase 4 Stage 4).

Sandbox-runnable via sys.modules stubbing of v2.data + v2.broker +
v2.bot.state — same harness as v2/sec/test_bot_responders.py. The
build_macro_snapshot + build_release_event functions are patched via
``monkeypatch.setattr`` so tests don't touch FRED / yfinance / DeepSeek.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---- Module-level edgartools stub (same reason as v2/sec tests) ----
if "edgar" not in sys.modules:
    _edgar_stub = types.ModuleType("edgar")
    _edgar_stub.Company = type("Company", (), {})
    _edgar_stub.set_identity = lambda *a, **kw: None
    sys.modules["edgar"] = _edgar_stub


@pytest.fixture(autouse=True)
def stub_v2_bot_imports(monkeypatch):
    """Mirror of the v2/sec/test_bot_responders.py harness — evict
    cached partial v2.reporting / v2.bot.responders from sys.modules
    (Phase 2 portfolio byte-equal test pollutes these), then install
    minimal v2.data / v2.broker / v2.bot.state stubs."""
    for cached in (
        "v2.reporting",
        "v2.reporting._portfolio_formatters",
        "v2.bot.responders",
    ):
        monkeypatch.delitem(sys.modules, cached, raising=False)

    # ---- v2.data shell + submodules ----
    v2_data = types.ModuleType("v2.data")
    v2_data.__path__ = []

    class _FakeFD:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def get_earnings(self, t): return None
        def get_earnings_history(self, t, limit=4): return []

    v2_data.CachedFDClient = _FakeFD
    v2_data.FDClient = _FakeFD
    monkeypatch.setitem(sys.modules, "v2.data", v2_data)

    v2_data_client = types.ModuleType("v2.data.client")
    v2_data_client.FDClient = _FakeFD
    monkeypatch.setitem(sys.modules, "v2.data.client", v2_data_client)

    v2_data_models = types.ModuleType("v2.data.models")
    v2_data_models.EarningsData = type("EarningsData", (), {})
    v2_data_models.EarningsRecord = type("EarningsRecord", (), {})
    v2_data_models.Price = type("Price", (), {})
    monkeypatch.setitem(sys.modules, "v2.data.models", v2_data_models)

    v2_data_news = types.ModuleType("v2.data.news_provider")
    v2_data_news.NewsProvider = type("NewsProvider", (), {})
    v2_data_news.default_news_provider = lambda: None
    monkeypatch.setitem(sys.modules, "v2.data.news_provider", v2_data_news)

    v2_data_yf = types.ModuleType("v2.data.yfinance_client")
    v2_data_yf.KNOWN_ADRS = frozenset()
    v2_data_yf.YFinanceClient = type("YFinanceClient", (), {})
    monkeypatch.setitem(sys.modules, "v2.data.yfinance_client", v2_data_yf)

    # ---- v2.broker ----
    v2_broker = types.ModuleType("v2.broker")
    v2_broker.AlpacaUnavailable = RuntimeError
    v2_broker.get_portfolio = lambda: {"positions": []}
    v2_broker.get_pnl = lambda: {"intraday_pl": 0.0, "intraday_pl_pct": 0.0}
    v2_broker.get_portfolio_history = lambda **kw: {"equity": [], "timestamp": []}
    monkeypatch.setitem(sys.modules, "v2.broker", v2_broker)

    # ---- v2.bot package + state ----
    v2_bot_pkg = types.ModuleType("v2.bot")
    v2_bot_pkg.__path__ = [str(_REPO_ROOT / "v2" / "bot")]
    v2_bot_state = types.ModuleType("v2.bot.state")
    v2_bot_state.watchlist_list = lambda: []
    v2_bot_pkg.state = v2_bot_state
    monkeypatch.setitem(sys.modules, "v2.bot", v2_bot_pkg)
    monkeypatch.setitem(sys.modules, "v2.bot.state", v2_bot_state)


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

def _full_snapshot():
    """Stage 5 dry-run analogue — all fields populated."""
    from v2.macro.models import MacroSnapshot
    return MacroSnapshot(
        snapshot_date="2026-06-12",
        vix=14.2, vix_pct_change_1d=0.008,
        dxy=99.5, wti_crude=78.4, gold=2650.5,
        fed_funds_upper=5.50, fed_funds_lower=5.25,
        dgs2=4.21, dgs10=4.42, t10y2y=0.21, t10y2y_prior=0.18,
    )


def _spike_snapshot():
    from v2.macro.models import MacroSnapshot
    return MacroSnapshot(
        snapshot_date="2026-06-12",
        vix=28.5, vix_pct_change_1d=0.25, vix_spike=True, vix_elevated=True,
        dxy=99.5, wti_crude=78.4, gold=2650.5,
        fed_funds_upper=5.50, fed_funds_lower=5.25,
        dgs2=4.21, dgs10=4.42, t10y2y=-0.10, t10y2y_prior=0.05,
        curve_flip=True,
    )


def _partial_snapshot_with_warnings():
    """yfinance partial failure case: VIX missing, DXY present, several
    warnings aggregated."""
    from v2.macro.models import MacroSnapshot
    return MacroSnapshot(
        snapshot_date="2026-06-12",
        vix=None, vix_pct_change_1d=None,
        dxy=99.5, wti_crude=None, gold=2650.5,
        fed_funds_upper=5.50, fed_funds_lower=5.25,
        dgs2=4.21, dgs10=4.42, t10y2y=0.21, t10y2y_prior=0.18,
        warnings=["yfinance VIX: HTTPError", "yfinance WTI: TimeoutException"],
    )


def _cpi_release():
    from v2.macro.models import MacroRelease
    return MacroRelease(
        release_type="CPI", release_date="2026-05-13",
        period="CPI April 2026",
        headline=320.5, core=315.2,
        mom_pct=0.002, yoy_pct=0.029,
        consensus=0.002, surprise_sigma=0.5,
        surprise_label="in_line",
        trailing_3mo_trend="decelerating",
        bull_takeaway="核心 YoY 连续 3 月放缓",
        bear_takeaway="MoM 仍高于 Fed 目标",
        narrative="通胀压力分化",
        tone="neutral",
    )


def _fomc_event():
    from v2.macro.models import FOMCEvent
    return FOMCEvent(
        meeting_date="2026-05-07",
        statement_text="...",
        statement_diff={
            "added_phrases": ["elevated"],
            "removed_phrases": ["modest"],
            "unchanged_phrases": [],
        },
        has_sep=False,
        sep_median_dots=None,
        sep_dot_plot_change="no_change",
        sell_side_sentiment="hawkish",
        sell_side_sources=["reuters.com", "bloomberg.com", "wsj.com"],
    )


# ---------------------------------------------------------------------------
# /macro view tests
# ---------------------------------------------------------------------------

class TestMacroView:

    def test_macro_view_full_data(self, monkeypatch):
        """All snapshot + calendar data populated → multi-section dashboard."""
        from v2.bot import responders

        monkeypatch.setattr(
            "v2.macro.build_macro_snapshot",
            lambda iso: _full_snapshot(),
        )
        monkeypatch.setattr(
            "v2.macro.release_calendar.get_releases_in_window",
            lambda start, end: {
                "2026-06-10": [("CPI", "CPI May 2026", "BLS")],
                "2026-06-17": [("FOMC", "Jun FOMC + SEP", "Fed")],
                "2026-06-25": [
                    ("PCE", "PCE May 2026", "BEA"),
                    ("GDP", "GDP Q1 2026", "BEA"),
                ],
            },
        )

        out = responders.macro_view({})
        assert "宏观 dashboard" in out
        assert "市场状态" in out
        assert "收益率" in out
        assert "VIX" in out
        # Yields band rendered
        assert "5.25% – 5.50%" in out
        # Spread shown (10-2 spread positive 0.21 → +21bp)
        assert "+21bp" in out
        # Calendar entries
        assert "CPI" in out
        assert "FOMC" in out

    def test_macro_view_spike_flag_visible(self, monkeypatch):
        """vix_spike → 🚨 +20% tag in card body."""
        from v2.bot import responders

        monkeypatch.setattr(
            "v2.macro.build_macro_snapshot",
            lambda iso: _spike_snapshot(),
        )
        monkeypatch.setattr(
            "v2.macro.release_calendar.get_releases_in_window",
            lambda start, end: {},
        )

        out = responders.macro_view({})
        assert "🚨 +20%" in out
        assert "今日翻转" in out         # curve_flip

    def test_macro_view_partial_data_shows_warnings(self, monkeypatch):
        """yfinance VIX failure → warnings list visible, dashboard
        still renders the remaining fields."""
        from v2.bot import responders

        monkeypatch.setattr(
            "v2.macro.build_macro_snapshot",
            lambda iso: _partial_snapshot_with_warnings(),
        )
        monkeypatch.setattr(
            "v2.macro.release_calendar.get_releases_in_window",
            lambda start, end: {},
        )

        out = responders.macro_view({})
        # Warning section rendered
        assert "数据不全" in out
        assert "yfinance VIX" in out
        # Non-VIX fields still present
        assert "99.5" in out          # DXY
        assert "2,650.5" in out       # Gold

    def test_macro_view_snapshot_exception_graceful(self, monkeypatch):
        """build_macro_snapshot raising → friendly error string,
        no exception propagated up."""
        from v2.bot import responders

        def boom(_iso):
            raise RuntimeError("FRED 503")

        monkeypatch.setattr("v2.macro.build_macro_snapshot", boom)

        out = responders.macro_view({})
        assert "宏观快照失败" in out
        assert "FRED 503" in out


# ---------------------------------------------------------------------------
# /cpi / /pce / /nfp / release_check tests
# ---------------------------------------------------------------------------

class TestReleaseCheck:

    def _patch_calendar(self, monkeypatch, releases_by_date):
        """Patch the underlying _2026_RELEASES dict so the past-date
        walker finds the test fixture."""
        import v2.macro.release_calendar as cal_mod
        monkeypatch.setattr(cal_mod, "_2026_RELEASES", releases_by_date)

    def test_release_check_cpi(self, monkeypatch):
        """release_check(release_type='cpi') → renders CPI card with
        Python-computed numerics + LLM labels."""
        from v2.bot import responders

        self._patch_calendar(monkeypatch, {
            "2020-01-01": [("CPI", "CPI Dec 2019", "BLS")],
            "2020-02-01": [("CPI", "CPI Jan 2020", "BLS")],
        })

        from v2.macro.models import MacroReport
        report = MacroReport(report_date="2020-02-01")
        report.today_releases = [_cpi_release()]
        monkeypatch.setattr(
            "v2.macro.build_release_event",
            lambda iso: report,
        )

        out = responders.release_check({"release_type": "cpi"})
        assert "CPI" in out
        # Python-computed numerics
        assert "MoM" in out
        assert "+0.20%" in out
        assert "YoY" in out
        assert "+2.90%" in out
        # LLM qualitative labels
        assert "核心 YoY 连续 3 月放缓" in out
        assert "MoM 仍高于 Fed 目标" in out

    def test_release_check_fomc_with_diff(self, monkeypatch):
        """release_check(release_type='fomc') → FOMC card with Python
        statement diff + Tavily aggregate. No LLM hawkish/dovish verdict."""
        from v2.bot import responders

        self._patch_calendar(monkeypatch, {
            "2020-05-07": [("FOMC", "May FOMC", "Fed")],
        })

        from v2.macro.models import MacroReport
        report = MacroReport(report_date="2020-05-07")
        report.fomc_event = _fomc_event()
        monkeypatch.setattr(
            "v2.macro.build_release_event",
            lambda iso: report,
        )

        out = responders.release_check({"release_type": "fomc"})
        assert "FOMC" in out
        # Statement diff visible
        assert "新增措辞" in out
        assert "elevated" in out
        assert "移除措辞" in out
        assert "modest" in out
        # Tavily aggregate visible (canonical formatter uses "卖方读数")
        assert "卖方读数" in out
        assert "hawkish" in out
        assert "reuters.com" in out

    def test_release_check_invalid_type_defaults_to_cpi(self, monkeypatch):
        """release_type outside enum → falls back to CPI."""
        from v2.bot import responders

        self._patch_calendar(monkeypatch, {
            "2020-01-01": [("CPI", "CPI Dec 2019", "BLS")],
        })

        from v2.macro.models import MacroReport
        report = MacroReport(report_date="2020-01-01")
        report.today_releases = [_cpi_release()]
        monkeypatch.setattr(
            "v2.macro.build_release_event",
            lambda iso: report,
        )

        out = responders.release_check({"release_type": "bogus_release"})
        # Renders the CPI card (default fallback)
        assert "CPI" in out

    def test_release_check_no_calendar_match_friendly(self, monkeypatch):
        """release_type with no past entries in calendar → friendly
        '未找到' message instead of crash."""
        from v2.bot import responders

        self._patch_calendar(monkeypatch, {})   # empty calendar

        out = responders.release_check({"release_type": "cpi"})
        assert "未找到" in out

    def test_release_check_fred_failure_friendly(self, monkeypatch):
        """build_release_event raising → friendly FRED failure card."""
        from v2.bot import responders

        self._patch_calendar(monkeypatch, {
            "2020-01-01": [("CPI", "CPI Dec 2019", "BLS")],
        })

        def boom(_iso):
            raise RuntimeError("FRED 502")

        monkeypatch.setattr("v2.macro.build_release_event", boom)

        out = responders.release_check({"release_type": "cpi"})
        assert "FRED" in out
        assert "失败" in out
