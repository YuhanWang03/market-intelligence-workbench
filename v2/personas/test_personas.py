"""Tests for the persona package. No network, no API key, no LLM."""

from __future__ import annotations

import json

import pytest

from v2.agent_common.llm import LLMResponse, ScriptedLLM
from v2.personas import PERSONAS, get_persona, list_personas
from v2.personas.base import apply_margin_of_safety, classic_verdict, confidence_from_ratio
from v2.personas.committee import analyze_snapshot, run_committee, tally
from v2.personas.data import ALL_LINE_ITEMS, FinancialDatasetsClient, adapt_client
from v2.personas.fixtures import distressed_snapshot, empty_snapshot, quality_snapshot
from v2.personas.models import Evaluation, PersonaSignal, Record, SubScore, as_records
from v2.personas.narrate import build_messages, narrate, parse_reply
from v2.personas.snapshot import PersonaSnapshot, build_snapshot


# --------------------------------------------------------------------------- models

def test_record_missing_field_reads_none_and_supports_mapping_protocol():
    r = Record({"a": 1}, b=None)
    assert r.a == 1 and r.b is None and r.zzz is None
    assert hasattr(r, "zzz")  # documented: never rely on AttributeError
    assert "a" in r and len(r) == 2 and r.get("q", 7) == 7
    assert r.model_dump() == {"a": 1, "b": None}
    r.c = 3
    assert r.to_dict()["c"] == 3


def test_as_records_accepts_dicts_records_and_objects():
    class Obj:
        def __init__(self):
            self.x = 1
            self._hidden = 2

    class Model:
        def model_dump(self):
            return {"y": 2}

    rows = as_records([{"a": 1}, Record(b=2), Obj(), Model()])
    assert [r.to_dict() for r in rows] == [{"a": 1}, {"b": 2}, {"x": 1}, {"y": 2}]


def test_evaluation_ratio_and_signal_serialization():
    ev = Evaluation(parts=[SubScore("a", 3, 5), SubScore("b", 1, 5)])
    assert ev.score == 4 and ev.max_score == 10 and ev.ratio == 0.4
    sig = PersonaSignal(persona="p", ticker="T", as_of="2026-01-01", signal="bullish", confidence=80, score=4, max_score=10, parts=ev.parts)
    body = json.loads(json.dumps(sig.to_dict()))
    assert body["ratio"] == 0.4 and body["parts"][0]["name"] == "a" and sig.direction == 1


# ----------------------------------------------------------------------------- base

def test_classic_verdict_and_confidence_are_monotone():
    assert classic_verdict(0.7) == "bullish" and classic_verdict(0.3) == "bearish" and classic_verdict(0.5) == "neutral"
    assert confidence_from_ratio(1.0, "bullish") > confidence_from_ratio(0.7, "bullish") >= 55
    assert confidence_from_ratio(0.0, "bearish") > confidence_from_ratio(0.3, "bearish") >= 55
    assert confidence_from_ratio(0.5, "neutral") == 50
    assert 10 <= confidence_from_ratio(0.31, "neutral") < 50


def test_margin_of_safety_gate():
    assert apply_margin_of_safety("bullish", 80, None) == ("bullish", 80)
    assert apply_margin_of_safety("bullish", 80, -0.1)[0] == "neutral"
    assert apply_margin_of_safety("neutral", 45, -0.4)[0] == "bearish"
    assert apply_margin_of_safety("bullish", 80, 0.5) == ("bullish", 85)


# ------------------------------------------------------------------------- snapshot

def test_snapshot_hash_depends_on_data_not_fetch_time():
    a, b = quality_snapshot(), quality_snapshot()
    b.fetched_at = "2030-01-01T00:00:00"
    assert a.content_hash == b.content_hash
    b.market_cap = 1.0
    assert a.content_hash != b.content_hash


def test_snapshot_round_trips_through_dict():
    snap = quality_snapshot()
    again = PersonaSnapshot.from_dict(json.loads(json.dumps(snap.to_dict())))
    assert again.content_hash == snap.content_hash
    assert again.metrics("ttm", 3)[0].return_on_equity == snap.metrics_ttm[0].return_on_equity
    assert len(again.prices) == len(snap.prices)


class _FakeClient:
    """Speaks the persona protocol and counts what was asked of it."""

    def __init__(self, snap: PersonaSnapshot):
        self.snap = snap
        self.calls: list[str] = []

    def get_financial_metrics(self, ticker, end_date, *, period="ttm", limit=10):
        self.calls.append(f"metrics:{period}")
        return [r.to_dict() for r in self.snap.metrics(period, limit)]

    def search_line_items(self, ticker, line_items, end_date, *, period="ttm", limit=10):
        self.calls.append(f"items:{period}")
        assert set(line_items) == set(ALL_LINE_ITEMS)
        return [r.to_dict() for r in self.snap.line_items(period, limit)]

    def get_market_cap(self, ticker, end_date):
        self.calls.append("market_cap")
        return self.snap.market_cap

    def get_insider_trades(self, ticker, end_date, *, start_date=None, limit=1000):
        self.calls.append("insiders")
        return list(self.snap.insider_trades)

    def get_company_news(self, ticker, end_date, *, start_date=None, limit=100):
        self.calls.append("news")
        return list(self.snap.news)

    def get_prices(self, ticker, start_date, end_date):
        self.calls.append("prices")
        return list(reversed(self.snap.prices))  # deliberately unsorted


def test_build_snapshot_fetches_once_and_only_what_is_needed():
    client = _FakeClient(quality_snapshot())
    snap = build_snapshot("qlty", "2026-06-30", client, need=("prices",))
    assert snap.ticker == "QLTY" and snap.market_cap == 250_000e6
    assert client.calls.count("metrics:ttm") == 1 and client.calls.count("items:annual") == 1
    assert "insiders" not in client.calls and "news" not in client.calls and "prices" in client.calls
    assert snap.prices[0].time < snap.prices[-1].time  # re-sorted ascending
    assert snap.gaps == []


def test_build_snapshot_records_gaps_instead_of_raising():
    class Broken:
        def get_financial_metrics(self, ticker, end_date, *, period="ttm", limit=10):
            raise RuntimeError("boom")

        def search_line_items(self, *a, **k):
            return []

        def get_market_cap(self, *a):
            return None

        def get_insider_trades(self, *a, **k):
            return []

        def get_company_news(self, *a, **k):
            return []

        def get_prices(self, *a):
            return []

    snap = build_snapshot("X", "2026-06-30", Broken())
    assert "financial_metrics" not in snap.requests and snap.requests["line_items"] == 2  # rejected requests are not billed, so not counted
    assert any(g.startswith("metrics_ttm: RuntimeError") for g in snap.gaps)
    assert any(g.startswith("fundamentals:") for g in snap.gaps)
    assert not snap.has_fundamentals


def test_adapt_client_wraps_a_production_style_client(monkeypatch):
    monkeypatch.delenv("FINANCIAL_DATASETS_API_KEY", raising=False)

    class ProdLike:
        """Positional signature like the VPS FDClient, no line-item support."""

        def get_financial_metrics(self, ticker, end_date, limit=1):
            return [{"market_cap": 42.0, "return_on_equity": 0.2}]

        def get_prices(self, ticker, start, end):
            return [{"time": "2026-01-02", "close": 1.0}]

    fd = adapt_client(ProdLike())
    assert fd.get_financial_metrics("T", "2026-06-30", period="ttm", limit=5)[0].return_on_equity == 0.2
    assert fd.get_market_cap("T", "2026-06-30") == 42.0
    with pytest.raises(NotImplementedError):
        fd.search_line_items("T", ["revenue"], "2026-06-30")
    snap = build_snapshot("T", "2026-06-30", ProdLike(), need=("prices",))
    assert snap.market_cap == 42.0 and snap.metrics_ttm and snap.prices
    assert any(g.startswith("line_items_ttm") for g in snap.gaps)


def test_adapt_client_returns_protocol_clients_unchanged():
    client = _FakeClient(quality_snapshot())
    assert adapt_client(client) is client
    http = FinancialDatasetsClient("k")
    assert adapt_client(http) is http and http.api_key == "k"


# ------------------------------------------------------------------------- personas

@pytest.mark.parametrize("key", PERSONAS)
def test_every_persona_runs_deterministically(key):
    p = get_persona(key)
    assert p.key == key and p.name and p.name_zh and p.style and p.system_prompt
    for snap in (quality_snapshot(), distressed_snapshot()):
        a = p.analyze(snap)
        b = p.analyze(snap)
        assert a.signal in ("bullish", "bearish", "neutral") and 0 <= a.confidence <= 100
        assert a.to_dict() == b.to_dict(), "persona output must be deterministic"
        assert a.max_score > 0 and a.parts and a.snapshot_hash == snap.content_hash
        assert abs(sum(part.score for part in a.parts) - a.score) < 1e-9
        json.dumps(a.to_dict())  # serializable end to end


@pytest.mark.parametrize("key", PERSONAS)
def test_every_persona_abstains_without_data(key):
    s = get_persona(key).analyze(empty_snapshot())
    assert s.abstained and s.confidence == 0 and s.signal == "neutral"
    assert s.reasoning.startswith("abstain")


@pytest.mark.parametrize("key", PERSONAS)
def test_no_persona_is_bullish_on_the_distressed_company(key):
    s = get_persona(key).analyze(distressed_snapshot())
    assert s.signal != "bullish", s.reasoning


def test_buffett_needs_a_margin_of_safety_to_be_bullish():
    buffett = get_persona("warren_buffett")
    rich = quality_snapshot()
    rich_signal = buffett.analyze(rich)
    assert rich_signal.margin_of_safety is not None and rich_signal.margin_of_safety < 0
    assert rich_signal.signal == "neutral"
    cheap = quality_snapshot()
    cheap.market_cap = 60_000e6
    cheap_signal = buffett.analyze(cheap)
    assert cheap_signal.margin_of_safety > 0 and cheap_signal.signal == "bullish"
    assert cheap_signal.confidence > rich_signal.confidence
    assert "moat" in {p.name for p in cheap_signal.parts}


def test_list_personas_orders_and_caches():
    people = list_personas(["ben_graham", "warren_buffett"])
    assert [p.key for p in people] == ["ben_graham", "warren_buffett"]
    assert get_persona("ben_graham") is people[0]
    with pytest.raises(KeyError):
        get_persona("nobody")


# ------------------------------------------------------------------------ committee

def _sig(persona, signal, conf, abstained=False):
    return PersonaSignal(persona=persona, ticker="T", as_of="2026-06-30", signal=signal, confidence=conf, score=1, max_score=2, abstained=abstained)


def test_tally_weights_votes_by_confidence_and_ignores_abstentions():
    v = tally([_sig("a", "bullish", 80), _sig("b", "bullish", 60), _sig("c", "bearish", 40), _sig("d", "neutral", 50), _sig("e", "neutral", 0, abstained=True)])
    assert (v.bullish, v.bearish, v.neutral, v.abstained) == (2, 1, 1, 1)
    assert v.net_votes == 1 and v.voters == 4
    assert v.consensus == pytest.approx((80 + 60 - 40) / 400)
    assert v.agreement == 0.5 and v.avg_confidence == pytest.approx(57.5)
    assert v.stance == "bullish"
    assert tally([]).stance == "abstain"


def test_run_committee_on_prebuilt_snapshots_ranks_quality_over_distress():
    snaps = {s.ticker: s for s in (quality_snapshot(), distressed_snapshot())}
    result = run_committee(["dstr", "QLTY", "qlty"], personas=["warren_buffett", "ben_graham", "michael_burry"], snapshots=snaps)
    assert [v.ticker for v in result.verdicts] == ["QLTY", "DSTR"]
    assert result.verdicts[0].rank == 1 and result.verdicts[0].consensus > result.verdicts[1].consensus
    assert result.verdict("DSTR").stance == "bearish"
    grid = result.matrix()
    assert set(grid) == {"warren_buffett", "ben_graham", "michael_burry"} and set(grid["ben_graham"]) == {"QLTY", "DSTR"}
    assert [v.ticker for v in result.top(1)] == ["QLTY"]
    json.dumps(result.to_dict())


def test_run_committee_uses_client_for_missing_snapshots_and_reports_errors():
    client = _FakeClient(quality_snapshot())
    result = run_committee(["QLTY"], client, personas=["warren_buffett"], max_workers=2)
    assert result.verdicts and result.verdicts[0].ticker == "QLTY" and not result.errors
    assert "insiders" not in client.calls  # Buffett does not read insider trades

    class Exploding:
        def get_financial_metrics(self, *a, **k):
            raise ValueError("no")

    result = run_committee(["ZZZ"], Exploding(), personas=["warren_buffett"])
    # An exploding client degrades to gaps, never to an exception.
    assert result.verdicts[0].signals[0].abstained


def test_a_crashing_persona_abstains_instead_of_sinking_the_vote():
    class Broken(type(get_persona("warren_buffett"))):
        key = "broken"

        def evaluate(self, snap):
            raise ZeroDivisionError("bad math")

    v = analyze_snapshot(quality_snapshot(), [get_persona("warren_buffett"), Broken()])
    assert v.abstained == 1 and v.voters == 1
    assert "ZeroDivisionError" in v.signals[1].reasoning


# --------------------------------------------------------------------------- narrate

def _bullish_signal():
    cheap = quality_snapshot()
    cheap.market_cap = 60_000e6
    return get_persona("warren_buffett").analyze(cheap)


def test_parse_reply_handles_fences_and_prose():
    assert parse_reply('```json\n{"signal":"bullish","confidence":70,"reasoning":"ok"}\n```')["reasoning"] == "ok"
    assert parse_reply('Sure: {"signal":"bullish","confidence":70,"reasoning":"ok"}.')["confidence"] == 70
    assert parse_reply("no json here") is None


def test_narrate_keeps_a_grounded_reply_and_flags_invented_numbers():
    sig = _bullish_signal()
    roe = sig.parts[0].details  # "Strong ROE of 24.0%; ..."
    good = json.dumps({"signal": sig.signal, "confidence": sig.confidence, "reasoning": f"护城河扎实，{roe.split(';')[0]}，安全边际为正。"})
    llm = ScriptedLLM([LLMResponse(text=good)])
    out = narrate(sig, llm=llm)
    assert out.narrative and out.narrative_grounded is True
    messages = llm.calls[0]
    assert messages[0]["role"] == "system" and "Warren Buffett" in messages[0]["content"]
    assert str(sig.confidence) in messages[1]["content"]

    sig2 = _bullish_signal()
    invented = json.dumps({"signal": sig2.signal, "confidence": sig2.confidence, "reasoning": "ROE 高达 87.5%，市盈率仅 3.14 倍。"})
    narrate(sig2, llm=ScriptedLLM([LLMResponse(text=invented)]))
    assert sig2.narrative and sig2.narrative_grounded is False


def test_narrate_discards_replies_that_change_the_verdict_or_fail():
    sig = _bullish_signal()
    flipped = json.dumps({"signal": "bearish", "confidence": sig.confidence, "reasoning": "no"})
    assert narrate(sig, llm=ScriptedLLM([LLMResponse(text=flipped)])).narrative is None
    changed = json.dumps({"signal": sig.signal, "confidence": 1, "reasoning": "no"})
    assert narrate(sig, llm=ScriptedLLM([LLMResponse(text=changed)])).narrative is None
    assert narrate(sig, llm=ScriptedLLM([LLMResponse(text="garbage")])).narrative is None

    class Dead:
        def complete(self, messages, tools=None):
            raise RuntimeError("offline")

    assert narrate(sig, llm=Dead()).narrative is None
    skipped = get_persona("warren_buffett").analyze(empty_snapshot())
    assert narrate(skipped, llm=Dead()).narrative is None  # abstentions are never narrated


def test_build_messages_mentions_language_and_facts():
    sig = _bullish_signal()
    zh = build_messages(sig, get_persona("warren_buffett"), language="zh")
    en = build_messages(sig, get_persona("warren_buffett"), language="en")
    assert "Simplified Chinese" in zh[0]["content"] and "English" in en[0]["content"]
    assert '"ticker":"QLTY"' in zh[1]["content"]


# -------------------------------------------------------------------------------- cli

def test_cli_demo_runs_without_network(capsys):
    from v2.personas.__main__ import main

    assert main(["--demo", "--personas", "warren_buffett,ben_graham", "--json"]) == 0
    body = json.loads(capsys.readouterr().out)
    assert [v["ticker"] for v in body["verdicts"]] == ["QLTY", "DSTR"]
    assert main(["--demo", "--personas", "warren_buffett"]) == 0
    assert "consensus" in capsys.readouterr().out


@pytest.mark.parametrize("key", PERSONAS)
def test_every_persona_abstains_when_line_items_are_missing(key):
    """Ratios without line items must read as 'no data', never as 'scores zero → bearish'."""
    snap = quality_snapshot()
    snap.line_items_ttm = []
    snap.line_items_annual = []
    snap.gaps.append("line_items_ttm: RuntimeError: HTTP 402 for /financials/search/line-items: payment required")
    snap.gaps.append("line_items_annual: RuntimeError: HTTP 402 for /financials/search/line-items: payment required")
    s = get_persona(key).analyze(snap)
    assert s.abstained and s.confidence == 0, s.reasoning
    assert "line items" in s.reasoning and "HTTP 402" in s.reasoning


@pytest.mark.parametrize("key", PERSONAS)
def test_every_persona_abstains_on_too_short_history(key):
    snap = quality_snapshot()
    snap.line_items_ttm = snap.line_items_ttm[:1]
    snap.line_items_annual = snap.line_items_annual[:1]
    s = get_persona(key).analyze(snap)
    assert s.abstained and "need 3" in s.reasoning


def test_annual_series_is_derived_from_ttm_when_the_provider_has_too_few_years():
    """FD only carries ~2 fiscal years at historical dates; TTM rows a year apart stand in."""
    from v2.personas.snapshot import derive_annual

    snap = quality_snapshot()                      # 10 quarterly TTM rows, 10 real annual rows
    real = snap.metrics("annual")
    assert real == snap.metrics_annual and not snap.annual_derived

    snap.metrics_annual = snap.metrics_annual[:2]  # what FD returns as of mid-2025
    snap.line_items_annual = snap.line_items_annual[:2]
    derived = snap.metrics("annual")
    assert len(derived) == 3 and all(r.period == "annual" and r.derived_from == "ttm" for r in derived)
    assert [r.report_period for r in derived] == [snap.metrics_ttm[i].report_period for i in (0, 4, 8)]
    assert len(snap.line_items("annual")) == 3 and snap.annual_derived
    for key in PERSONAS:                           # nobody abstains for "need 3" any more
        sig = get_persona(key).analyze(snap)
        assert "need 3" not in sig.reasoning, (key, sig.reasoning)

    # the derived series is only used when it is longer than the real one
    snap.metrics_ttm = snap.metrics_ttm[:5]        # 5 quarters → 2 derived points, no better than 2 real
    assert snap.metrics("annual") == snap.metrics_annual
    assert derive_annual([]) == [] and derive_annual([Record(ticker="X", revenue=1)]) == []


def test_buffett_survives_a_loss_making_latest_period():
    """INTC / CRWD: a negative newest net income made (new/old) ** (1/years) complex and crashed the vote."""
    snap = quality_snapshot()
    for row in snap.line_items_ttm[:2]:
        row.net_income = -2_000e6
    sig = get_persona("warren_buffett").analyze(snap)
    assert not sig.abstained and "TypeError" not in sig.reasoning
    assert sig.signal in ("bearish", "neutral", "bullish")


def test_price_readers_abstain_without_prices():
    snap = quality_snapshot()
    snap.prices = []
    assert get_persona("nassim_taleb").analyze(snap).abstained
    assert get_persona("stanley_druckenmiller").analyze(snap).abstained
    assert not get_persona("warren_buffett").analyze(snap).abstained


def test_store_refuses_to_cache_snapshots_with_core_gaps(tmp_path):
    from v2.personas.store import PersonaStore

    store = PersonaStore(tmp_path / "p.db")
    good = quality_snapshot()
    good.fetched_at = ""  # fixture timestamp is months old; let the store stamp now
    store.save_snapshot(good)
    assert store.cached_snapshot("QLTY", good.as_of) is not None
    broken = distressed_snapshot()
    broken.gaps.append("line_items_ttm: RuntimeError: HTTP 402")
    store.save_snapshot(broken)
    assert store.cached_snapshot("DSTR", broken.as_of) is None


def test_http_client_sends_a_real_user_agent_and_explains_cloudflare_403(monkeypatch):
    import io
    import urllib.error
    import urllib.request

    from v2.personas.data import USER_AGENT

    seen: list[urllib.request.Request] = []

    def fake_urlopen(request, timeout=0):
        seen.append(request)
        body = b'{"type":"https://developers.cloudflare.com/waf","title":"blocked"}'
        raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", {}, io.BytesIO(body))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = FinancialDatasetsClient("k", max_retries=0)
    with pytest.raises(RuntimeError) as exc:
        client.search_line_items("AAPL", ["revenue"], "2026-06-30")
    assert "HTTP 403" in str(exc.value) and "Cloudflare" in str(exc.value)
    assert seen[0].get_header("User-agent") == USER_AGENT
    assert seen[0].get_header("X-api-key") == "k"


def test_adapter_survives_a_broken_production_get_market_cap():
    class Prod:
        def get_market_cap(self, ticker):
            raise AttributeError("'CompanyFacts' object has no attribute 'market_cap'")

        def get_financial_metrics(self, ticker, end_date, limit=1):
            return [{"market_cap": 123.0}]

    fd = adapt_client(Prod())
    assert fd.get_market_cap("AAPL", "2026-06-30") == 123.0


def test_adapter_caps_news_limit():
    class Prod:
        def get_news(self, ticker, end, start, limit):
            assert limit <= 100
            return [{"title": "x"}] * 3

    assert len(adapt_client(Prod()).get_company_news("AAPL", "2026-06-30", start_date="2025-06-30", limit=250)) == 3



def test_http_client_drops_line_items_the_api_rejects(monkeypatch):
    import io
    import json as _json
    import urllib.error
    import urllib.request

    bodies: list[list[str]] = []

    def fake_urlopen(request, timeout=0):
        items = _json.loads(request.data)["line_items"]
        bodies.append(items)
        bad = [n for n in items if n in ("intangible_assets", "made_up")]
        if bad:
            body = _json.dumps({"error": f"Invalid line items: {', '.join(bad)}", "message": "x"}).encode()
            raise urllib.error.HTTPError(request.full_url, 400, "Bad Request", {}, io.BytesIO(body))
        return io.BytesIO(_json.dumps({"search_results": [{"ticker": "AAPL", "revenue": 1.0}]}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    rows = FinancialDatasetsClient("k", max_retries=0).search_line_items("AAPL", ["revenue", "intangible_assets", "made_up"], "2026-06-30")
    assert rows[0].revenue == 1.0
    assert bodies[-1] == ["revenue"] and len(bodies) == 2


def test_news_limit_steps_down_until_accepted():
    from v2.personas.data import _with_smaller_news_limit

    tried: list[int] = []

    def fetch(limit):
        tried.append(limit)
        if limit > 20:
            raise RuntimeError('HTTP 400 for /news/: {"error":"Invalid limit"}')
        return ["n"] * limit

    assert len(_with_smaller_news_limit(fetch, 100)) == 20 and tried == [100, 50, 20]
    with pytest.raises(ValueError):
        _with_smaller_news_limit(lambda n: (_ for _ in ()).throw(ValueError("unrelated")), 100)


def test_cached_snapshot_rejects_rows_saved_with_core_gaps(tmp_path):
    import json as _json
    import sqlite3

    from v2.personas.store import PersonaStore, utc_now

    store = PersonaStore(tmp_path / "p.db")
    snap = quality_snapshot()
    snap.gaps.append("line_items_ttm: RuntimeError: HTTP 403")
    with sqlite3.connect(store.path) as conn:  # simulate a row written before the rule existed
        conn.execute("INSERT INTO snapshots (ticker, as_of, content_hash, fetched_at, payload_json) VALUES (?,?,?,?,?)",
                     (snap.ticker, snap.as_of, snap.content_hash, utc_now(), _json.dumps(snap.to_dict(), default=str)))
    assert store.cached_snapshot("QLTY", snap.as_of) is None



def test_cagr_uses_report_dates_not_row_count():
    from v2.personas.base import Persona

    snap = quality_snapshot()  # ~12%/yr growth, quarterly TTM rows 91 days apart, annual rows a year apart
    ttm = Persona.cagr(snap.line_items_ttm, "revenue")
    annual = Persona.cagr(snap.line_items_annual, "revenue")
    assert ttm is not None and annual is not None
    assert abs(ttm[0] - 0.12) < 0.01 and abs(annual[0] - 0.12) < 0.01
    assert 2.1 < ttm[1] < 2.4 and 8.9 < annual[1] < 9.1
    # the old arithmetic would have called ten TTM rows nine years: ~3%/yr
    assert Persona.cagr(snap.line_items_ttm[:2], "revenue") is None  # one quarter is too short a span
    undated = [Record(period="ttm", revenue=v) for v in (121.0, 118.0, 115.0, 112.0, 109.0)]
    fallback = Persona.cagr(undated, "revenue")
    assert fallback is not None and fallback[1] == 1.0 and abs(fallback[0] - 0.11) < 0.01
    assert Persona.cagr([Record(revenue=-1.0), Record(revenue=2.0)], "revenue") is None


def test_jhunjhunwala_and_damodaran_no_longer_read_ttm_rows_as_years():
    snap = quality_snapshot()
    rj = get_persona("rakesh_jhunjhunwala").analyze(snap)
    details = " ".join(p.details for p in rj.parts)
    assert "revenue CAGR: 12." in details and "EPS CAGR: 12." in details, details
    assert "Low EPS CAGR" not in details
    # with growth read as ~3%/yr the DCF landed near $103B; at the true 12% it clears $120B
    assert rj.facts["intrinsic_value"] > 120e9
    ad = get_persona("aswath_damodaran").analyze(snap)
    growth = ad.parts[0].details
    assert "12." in growth, growth



def test_consistency_and_trend_loops_read_newest_first_order():
    grow, shrink = quality_snapshot(), distressed_snapshot()
    rj_grow = " ".join(p.details for p in get_persona("rakesh_jhunjhunwala").analyze(grow).parts)
    rj_shrink = " ".join(p.details for p in get_persona("rakesh_jhunjhunwala").analyze(shrink).parts)
    assert "Consistent growth pattern (100% of periods)" in rj_grow, rj_grow
    assert "Inconsistent growth pattern" in rj_shrink or "Insufficient" in rj_shrink, rj_shrink
    munger_grow = get_persona("charlie_munger").analyze(grow).parts[0].details
    munger_shrink = get_persona("charlie_munger").analyze(shrink).parts[0].details
    assert "Gross margins consistently improving" in munger_grow, munger_grow
    assert "consistently improving" not in munger_shrink, munger_shrink


def test_wikipedia_constituents_parser_is_nesting_safe_and_picks_the_best_table():
    import runpy

    ns = runpy.run_path("v2/screening/universes.py", run_name="not_main")  # avoid importing the production-only screener package
    html = """
    <table><tr><th>Year</th><th>Return</th></tr><tr><td>2024</td><td>+25%</td></tr></table>
    <table class="wikitable"><caption>Constituents</caption>
    <tr><th>Company<span>sort</span></th><th>Ticker</th><th>GICS Sector</th></tr>
    <tr><td>Adobe<table><tr><td>inner</td></tr></table></td><td><a href="x">ADBE</a></td><td>IT</td></tr>
    <tr><td>Alphabet</td><td>GOOGL</td><td>Comm</td></tr>
    <tr><td>Berkshire</td><td>BRK.B</td><td>Fin</td></tr>
    <tr><td>Bad</td><td>not a ticker</td><td>x</td></tr>
    </table>
    <table><tr><th>Symbol</th></tr><tr><td>ONLY</td></tr></table>"""
    assert ns["parse_constituents"](html, ("ticker", "symbol")) == ["ADBE", "GOOGL", "BRK.B"]
    assert len(ns["describe_tables"](html)) == 4
    sp500, as_of = ns["load_universe"]("sp500")
    assert len(sp500) > 450 and len(set(sp500)) == len(sp500) and as_of
