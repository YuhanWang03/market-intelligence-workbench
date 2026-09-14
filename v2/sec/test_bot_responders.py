"""Bot responder tests for /8k and /insiders (Phase 3 Stage 4).

Sandbox-runnable via sys.modules stubbing of v2.data + v2.broker +
v2.bot.state (same pattern as Phase 1/2 cron-integration tests).
Mocks edgartools Filing objects directly — verified shape in Phase 3
Stage 0 task 2.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---- Module-level edgartools stub ------------------------------------
# Pytest collection of this file imports v2/sec/__init__.py which pulls
# v2.sec.client → `from edgar import Company, set_identity`. The sandbox
# does not ship edgartools. Install a minimal stub so collection works.
# Production VPS has the real `edgar` package and is unaffected.
if "edgar" not in sys.modules:
    _edgar_stub = types.ModuleType("edgar")
    _edgar_stub.Company = type("Company", (), {})
    _edgar_stub.set_identity = lambda *a, **kw: None
    sys.modules["edgar"] = _edgar_stub


@pytest.fixture(autouse=True)
def stub_v2_bot_imports(monkeypatch):
    """Pre-stub v2.data / v2.broker / v2.bot.state so v2.bot.responders
    imports cleanly in sandbox (real modules transitively need v2.data).

    The v2.bot.responders import chain pulls v2.screening +
    v2.monitoring + v2.lateral + v2.institutional which all reach into
    v2.data.{client,models,news_provider,yfinance_client}. We stub
    each submodule explicitly with the symbols those callers import.
    """
    # ---- Evict caches polluted by other tests --------------------------
    # v2/portfolio/test_formatters_byte_equal.py installs a partial
    # v2.reporting stub (only the 4 portfolio formatters) via
    # sys.modules.setdefault, which persists across tests. If we let
    # that stub stay, `from v2.reporting import format_alert_list, ...`
    # inside v2.bot.responders breaks. Drop it here so the real package
    # loads when v2.bot.responders is imported below.
    for _cached in (
        "v2.reporting",
        "v2.reporting._portfolio_formatters",
        "v2.bot.responders",
    ):
        monkeypatch.delitem(sys.modules, _cached, raising=False)
    # ---- v2.data (package + 4 submodules) ----
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

    # ---- v2.broker (responders' _user_universe imports get_portfolio) ----
    v2_broker = types.ModuleType("v2.broker")
    v2_broker.AlpacaUnavailable = RuntimeError
    v2_broker.get_portfolio = lambda: {"positions": []}
    v2_broker.get_pnl = lambda: {"intraday_pl": 0.0, "intraday_pl_pct": 0.0}
    v2_broker.get_portfolio_history = lambda **kw: {"equity": [], "timestamp": []}
    monkeypatch.setitem(sys.modules, "v2.broker", v2_broker)

    # ---- v2.bot package + .state submodule ----
    v2_bot_pkg = types.ModuleType("v2.bot")
    v2_bot_pkg.__path__ = [str(_REPO_ROOT / "v2" / "bot")]
    v2_bot_state = types.ModuleType("v2.bot.state")
    v2_bot_state.watchlist_list = lambda: []
    v2_bot_pkg.state = v2_bot_state
    monkeypatch.setitem(sys.modules, "v2.bot", v2_bot_pkg)
    monkeypatch.setitem(sys.modules, "v2.bot.state", v2_bot_state)


# ---------------------------------------------------------------------------
# Mocked edgartools filing builders
# ---------------------------------------------------------------------------

def _fake_8k(*, acc="acc-1", filing_date="2026-06-04", items=None,
             text="", form="8-K"):
    items = items if items is not None else ["ITEM 5.02: Departure"]
    return SimpleNamespace(
        accession_number=acc, accession_no=acc,
        filing_date=filing_date, cik="0001645590", form=form,
        obj=lambda: SimpleNamespace(items=list(items), text=text),
    )


def _fake_form4(*, acc="f4-1", filing_date="2026-06-04",
                insider_name="Jane Doe", insider_title="Director",
                transactions=None):
    transactions = transactions if transactions is not None else [
        {"Security": "Common", "Date": filing_date, "Shares": 1000.0,
         "Remaining": 10000.0, "Price": 100.0, "AcquiredDisposed": "A",
         "DirectIndirect": "D", "form": "4", "Code": "P", "footnotes": ""},
    ]
    df = pd.DataFrame(transactions)
    relationship = SimpleNamespace(
        officer_title=insider_title,
        is_officer=insider_title not in ("", "Director"),
        is_director=insider_title == "Director",
        is_ten_percent_owner=False,
    )
    owner = SimpleNamespace(relationship=relationship)
    owners = SimpleNamespace(owners=[owner])
    form4_obj = SimpleNamespace(
        insider_name=insider_name,
        reporting_owners=owners,
        to_dataframe=lambda: df.copy(),
    )
    return SimpleNamespace(
        accession_number=acc, accession_no=acc,
        filing_date=filing_date, cik="0001973239", form="4",
        obj=lambda: form4_obj,
    )


# ---------------------------------------------------------------------------
# /8k responder tests
# ---------------------------------------------------------------------------

class TestEightKView:

    def test_valid_ticker_with_filings(self, monkeypatch):
        """Multi-item HPE filing + standalone 5.02 → card lists both."""
        from v2.bot import responders

        # Patch client + LLM extractor
        filings = [
            _fake_8k(acc="acc-multi", filing_date="2026-06-04",
                     items=["ITEM 1.01: Material Agreement",
                            "ITEM 2.02: Earnings",
                            "ITEM 5.02: Departure of CEO",
                            "ITEM 9.01: Exhibits"],
                     text=(
                         "Item 1.01 ...\nMaterial agreement entered.\n"
                         "Item 2.02 ...\nResults of operations.\n"
                         "Item 5.02 ...\nJohn Smith resigned as CEO. "
                         "Jane Doe appointed Interim CEO.\n"
                         "Item 9.01 ...\nExhibits.\n"
                     )),
        ]
        monkeypatch.setattr(
            "v2.sec.client.get_recent_filings",
            lambda ticker, form, since, until: filings,
        )
        # Mock LLM extractor — returns canned NER result
        monkeypatch.setattr(
            "v2.sec.ner_5_02.extract_5_02",
            lambda text, **kw: {
                "departures": [{"name": "John Smith", "title": "Chief Executive Officer"}],
                "appointments": [{"name": "Jane Doe", "title": "Interim CEO"}],
                "has_senior_exec": True,
            },
        )

        out = responders.eight_k_view({"ticker": "HPE"})

        # Card structure
        assert "SEC 8-K · HPE · 过去 30 天" in out
        assert "共 <b>1</b> 个 filings" in out
        # All 4 items rendered
        assert "1.01" in out
        assert "2.02" in out
        assert "5.02" in out
        assert "9.01" in out
        # Tier emojis present
        assert "🚨" in out          # P0 from 5.02 (per priority table)
        # 2.02 annotated as routed to ⑧
        assert "⑧ 处理" in out
        # 5.02 LLM extraction visible
        assert "John Smith" in out
        assert "Jane Doe" in out

    def test_empty_filings_returns_friendly_card(self, monkeypatch):
        """No filings in the window → '无 8-K 申报' card."""
        from v2.bot import responders
        monkeypatch.setattr(
            "v2.sec.client.get_recent_filings",
            lambda ticker, form, since, until: [],
        )
        out = responders.eight_k_view({"ticker": "AAPL"})
        assert "SEC 8-K · AAPL · 过去 30 天" in out
        assert "无 8-K 申报" in out

    def test_invalid_ticker_friendly_error(self):
        """Non-ascii / wrong-length ticker → friendly rejection."""
        from v2.bot import responders
        out = responders.eight_k_view({"ticker": "INVALID123"})
        assert "无效 ticker" in out
        # Empty ticker
        out2 = responders.eight_k_view({"ticker": ""})
        assert "无效 ticker" in out2

    def test_5_02_llm_failure_shows_placeholder(self, monkeypatch):
        """LLM raises → card still renders with '(姓名待解析)' placeholder."""
        from v2.bot import responders

        monkeypatch.setattr(
            "v2.sec.client.get_recent_filings",
            lambda ticker, form, since, until: [
                _fake_8k(items=["ITEM 5.02: Departure"], text="..."),
            ],
        )
        # LLM raises
        def boom(text, **kw):
            raise RuntimeError("DeepSeek 503")

        monkeypatch.setattr("v2.sec.ner_5_02.extract_5_02", boom)

        out = responders.eight_k_view({"ticker": "XYZ"})
        assert "5.02" in out
        assert "姓名待解析" in out


# ---------------------------------------------------------------------------
# /insiders responder tests
# ---------------------------------------------------------------------------

class TestInsiderView:

    def test_purchase_block_renders(self, monkeypatch):
        """5 P-code transactions → purchase block with total + max."""
        from v2.bot import responders

        purchases = [
            _fake_form4(
                acc=f"p-{i}", filing_date=f"2026-06-{i:02d}",
                insider_name=f"Insider {i}",
                insider_title="CEO" if i == 0 else "Director",
                transactions=[{
                    "Security": "Common", "Date": f"2026-06-{i:02d}",
                    "Shares": 1000.0 + i * 100, "Remaining": 50000.0,
                    "Price": 100.0, "AcquiredDisposed": "A",
                    "DirectIndirect": "D", "form": "4", "Code": "P",
                    "footnotes": "",
                }],
            )
            for i in range(1, 6)
        ]
        monkeypatch.setattr(
            "v2.sec.client.get_recent_filings",
            lambda ticker, form, since, until: purchases,
        )
        out = responders.insider_view({"ticker": "NVDA"})
        # Purchase block visible
        assert "P (Purchase): 5 笔" in out
        assert "最大:" in out
        # Sales empty
        assert "S (Sale):</b> 0 笔" in out

    def test_AMF_aggregated_not_listed_individually(self, monkeypatch):
        """A/M/F transactions show counts, NOT individual rows."""
        from v2.bot import responders

        filings = []
        # 8 awards, 3 exercises, 2 tax
        for i in range(8):
            filings.append(_fake_form4(
                acc=f"a-{i}", filing_date="2026-06-04",
                insider_name="Officer X",
                transactions=[{
                    "Security": "Common", "Date": "2026-06-04",
                    "Shares": 100.0, "Remaining": 1000.0, "Price": 100.0,
                    "AcquiredDisposed": "A", "DirectIndirect": "D",
                    "form": "4", "Code": "A", "footnotes": "",
                }],
            ))
        for i in range(3):
            filings.append(_fake_form4(
                acc=f"m-{i}", filing_date="2026-06-04",
                insider_name="Officer X",
                transactions=[{
                    "Security": "Common", "Date": "2026-06-04",
                    "Shares": 100.0, "Remaining": 1000.0, "Price": 100.0,
                    "AcquiredDisposed": "A", "DirectIndirect": "D",
                    "form": "4", "Code": "M", "footnotes": "",
                }],
            ))
        for i in range(2):
            filings.append(_fake_form4(
                acc=f"f-{i}", filing_date="2026-06-04",
                insider_name="Officer X",
                transactions=[{
                    "Security": "Common", "Date": "2026-06-04",
                    "Shares": 100.0, "Remaining": 1000.0, "Price": 100.0,
                    "AcquiredDisposed": "D", "DirectIndirect": "D",
                    "form": "4", "Code": "F", "footnotes": "",
                }],
            ))

        monkeypatch.setattr(
            "v2.sec.client.get_recent_filings",
            lambda ticker, form, since, until: filings,
        )
        out = responders.insider_view({"ticker": "NVDA"})
        # Code counts shown
        assert ">A</b>: 8 笔" in out
        assert ">M</b>: 3 笔" in out
        assert ">F</b>: 2 笔" in out
        # Individual rows for these codes should NOT appear (no "Officer X · $...K · ...")
        assert "Officer X · " not in out

    def test_cluster_detected_in_window(self, monkeypatch):
        """3 distinct insiders P same day → cluster block shown."""
        from v2.bot import responders

        filings = []
        for i, name in enumerate(["Alice", "Bob", "Carol"]):
            filings.append(_fake_form4(
                acc=f"clu-{i}",
                filing_date="2026-06-04",
                insider_name=name,
                insider_title="Director",
                transactions=[{
                    "Security": "Common", "Date": "2026-06-04",
                    "Shares": 1000.0, "Remaining": 50000.0,
                    "Price": 100.0, "AcquiredDisposed": "A",
                    "DirectIndirect": "D", "form": "4", "Code": "P",
                    "footnotes": "",
                }],
            ))

        monkeypatch.setattr(
            "v2.sec.client.get_recent_filings",
            lambda ticker, form, since, until: filings,
        )
        out = responders.insider_view({"ticker": "ARM"})
        # Cluster section present and lists the 3 names
        assert "集群:</b> 1 个" in out
        assert "Alice" in out
        assert "Bob" in out
        assert "Carol" in out
        assert "买入" in out

    def test_no_cluster_message(self, monkeypatch):
        """Only 2 distinct insiders → no cluster → friendly message."""
        from v2.bot import responders

        filings = [
            _fake_form4(
                acc=f"x-{i}", filing_date="2026-06-04",
                insider_name=name,
                insider_title="Director",
                transactions=[{
                    "Security": "Common", "Date": "2026-06-04",
                    "Shares": 1000.0, "Remaining": 50000.0,
                    "Price": 100.0, "AcquiredDisposed": "A",
                    "DirectIndirect": "D", "form": "4", "Code": "P",
                    "footnotes": "",
                }],
            )
            for i, name in enumerate(["A", "B"])
        ]
        monkeypatch.setattr(
            "v2.sec.client.get_recent_filings",
            lambda ticker, form, since, until: filings,
        )
        out = responders.insider_view({"ticker": "NVDA"})
        assert "无同日 ≥3 distinct insiders 集群" in out

    def test_days_back_param_bounded(self, monkeypatch):
        """days_back=10 within [7, 365] respected; reflects in card header."""
        from v2.bot import responders

        captured: dict = {}

        def spy_fetch(ticker, form, since, until):
            captured["since"] = since
            captured["until"] = until
            return []

        monkeypatch.setattr("v2.sec.client.get_recent_filings", spy_fetch)
        out = responders.insider_view({"ticker": "AAPL", "days_back": 10})
        assert "过去 10 天" in out
        # Verify since/until span ~10 days (allowing slop for calendar math)
        from datetime import date
        since_d = date.fromisoformat(captured["since"])
        until_d = date.fromisoformat(captured["until"])
        assert (until_d - since_d).days == 10

    def test_days_back_clamped_to_bounds(self, monkeypatch):
        """days_back=2 < min 7 → clamped to 7; days_back=999 > max 365 → 365."""
        from v2.bot import responders

        captured: dict = {}

        def spy_fetch(ticker, form, since, until):
            captured["since"] = since
            captured["until"] = until
            return []

        monkeypatch.setattr("v2.sec.client.get_recent_filings", spy_fetch)
        # Too small → clamps to 7
        out = responders.insider_view({"ticker": "AAPL", "days_back": 2})
        from datetime import date
        delta = (date.fromisoformat(captured["until"])
                 - date.fromisoformat(captured["since"])).days
        assert delta == 7
        assert "过去 7 天" in out

        # Too large → clamps to 365
        responders.insider_view({"ticker": "AAPL", "days_back": 999})
        delta = (date.fromisoformat(captured["until"])
                 - date.fromisoformat(captured["since"])).days
        assert delta == 365

    def test_invalid_ticker_friendly_error(self):
        from v2.bot import responders
        out = responders.insider_view({"ticker": "INVALID123"})
        assert "无效 ticker" in out
