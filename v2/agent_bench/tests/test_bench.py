"""The bench's own plumbing, offline: cases, bank, replay, grading, runner, pairwise blinding, report."""
from __future__ import annotations

import json
from collections import Counter

import pytest

from v2.agent_bench import cases as case_module
from v2.agent_bench.bank import Bank, Replay, Recorder, canonical, fault_envelope
from v2.agent_bench.cases import BenchCase, all_cases, by_id, dev_cases, select
from v2.agent_bench.judge import Score, assign_sides, compare_pair, grade, scrub
from v2.agent_bench.report import fold_attempts, pair_summary, render, summarize, write_report
from v2.agent_bench.runner import Run, read_ledger
from v2.agent_v2.models import EvidenceItem, ResultStatus, ToolEnvelope


# --- cases --------------------------------------------------------------------------

def test_case_set_is_consistent():
    cases = all_cases()
    table = by_id(cases)  # raises on duplicate ids
    assert len(table) >= 100
    assert all(c.criteria for c in cases), "every case has a rubric"
    assert all(c.category in case_module.CATEGORIES for c in cases)
    assert all(c.set in {"dev", "holdout"} for c in cases)
    origins = Counter(c.origin for c in cases)
    assert origins["quality_v2"] >= 56 and origins["evaluation_v3"] == 18 and origins["seed"] >= 30
    assert all(c.fault is None or c.fault["mode"] in {"error", "empty", "timeout"} for c in cases)
    assert all(c.frozen_only for c in cases if c.fault or c.fixtures), "injected conditions need the frozen bank"
    v2_only = [c for c in cases if c.expectations.get("v2")]
    assert v2_only and all("route" in c.expectations["v2"] or "agents" in c.expectations["v2"] for c in v2_only)


def test_selection_hides_frozen_only_cases_outside_frozen_mode():
    live = select(dev_cases(), mode="live")
    frozen = select(dev_cases(), mode="frozen")
    assert len(frozen) > len(live) and not any(c.frozen_only for c in live)
    assert [c.id for c in select(all_cases(), sets=("holdout",), categories=("safety",))]


# --- bank ---------------------------------------------------------------------------

def _envelope(capability="market.performance", subject="NVDA"):
    return ToolEnvelope(capability, ResultStatus.COMPLETED, subject=subject, evidence=[EvidenceItem("demo-price", subject, f"{subject} 收盘价为 120 美元。", metric="close", value=120, unit="USD", source_id="market_data")])


def test_bank_records_replays_and_reports_misses(tmp_path):
    bank = Bank(tmp_path / "bank")
    recorder = Recorder()
    recorder.use()
    wrapped = recorder.wrap("market.performance", lambda args, ctx: _envelope())
    assert wrapped({"ticker": "NVDA"}, None).status is ResultStatus.COMPLETED
    path = bank.save("c1", recorder.records)
    assert json.loads(path.read_text(encoding="utf-8"))["records"][0]["arguments"] == {"ticker": "NVDA"}
    replay = Replay()
    replay.use(bank.load("c1"))
    handler = replay.handler("market.performance")
    hit = handler({"ticker": "NVDA"}, None)
    miss = handler({"ticker": "AMD"}, None)
    assert hit.evidence[0].value == 120 and hit.elapsed_ms == 0
    assert miss.status is ResultStatus.PARTIAL_DATA and miss.metadata["fixture_missing"] is True
    assert [c["fixture_match"] for c in replay.calls] == [True, False] and len(replay.missing()) == 1
    assert canonical("x", {"b": 1, "a": 2}) == canonical("x", {"a": 2, "b": 1})
    assert bank.sha() and bank.sha() != Bank(tmp_path / "empty").sha()


def test_fault_injection_and_synthetic_fixtures_take_precedence(tmp_path):
    replay = Replay()
    replay.use([], fault={"capability": "market.performance", "mode": "timeout"}, fixtures=({"capability": "web.research", "arguments": None, "result": {"capability": "web.research", "status": "completed", "subject": "AAPL", "evidence": [{"id": "w1", "entity": "AAPL", "claim": "injected", "source_id": "web:news"}], "limitations": [], "errors": [], "metrics": {}, "findings": [], "metadata": {}}},))
    failed = replay.handler("market.performance")({"ticker": "NVDA"}, None)
    assert failed.status is ResultStatus.FAILED and "timeout" in failed.errors[0]
    served = replay.handler("web.research")({"query": "anything at all"}, None)
    assert served.status is ResultStatus.COMPLETED and served.evidence[0].claim == "injected"
    assert fault_envelope({"mode": "empty"}, "x").status is ResultStatus.PARTIAL_DATA


# --- grading ------------------------------------------------------------------------

def _result(answer, *, status="completed", evidence=(), results=(), route="research", sub_agents=()):
    return {"answer": answer, "status": status, "evidence": list(evidence), "results": list(results), "route": {"kind": route}, "sub_agents": list(sub_agents), "verification": {"ok": True, "warnings": []}, "synthesis": {}}


def test_grade_combines_judge_and_deterministic_checks():
    case = BenchCase("t", "market", "q", ("说了价格",), forbidden=("编造",), must_cite=("market_data",), expectations={"v2": {"route": "research", "agents": ["debater"]}})
    good = _result("收盘 120 美元 [demo-price]", evidence=[{"id": "demo-price", "source_id": "market_data"}], sub_agents=[{"name": "debater"}])
    verdict = {"criteria": [{"index": 0, "met": True, "quote": "120"}], "forbidden": [{"index": 0, "asserted": False}]}
    assert grade(case, "v2", good, verdict).passed
    assert grade(case, "v3", _result("收盘 120 美元 [demo-price]", evidence=[{"id": "demo-price", "source_id": "market_data"}]), verdict).passed, "V2-only expectations are not applied to V3"
    no_cite = grade(case, "v3", _result("收盘 120 美元"), verdict)
    assert not no_cite.passed and not no_cite.sources_ok
    unjudged = grade(case, "v3", good, None)
    assert not unjudged.passed and not unjudged.judged and "未经裁判评分" in unjudged.problems
    hit = grade(case, "v3", good, {"criteria": [{"index": 0, "met": True}], "forbidden": [{"index": 0, "asserted": True, "quote": "x"}]})
    assert hit.forbidden_hit == 1 and not hit.passed
    missing_agent = grade(case, "v2", _result("收盘 120 美元 [demo-price]", evidence=[{"id": "demo-price", "source_id": "market_data"}]), verdict)
    assert not missing_agent.route_ok


def test_grade_confirmation_cases_require_the_waiting_status_and_no_write():
    case = BenchCase("t", "command", "加入关注", ("要求确认",), expect_status=("waiting_confirmation",), forbid_capabilities=("state.mutate",))
    verdict = {"criteria": [{"index": 0, "met": True}], "forbidden": []}
    assert grade(case, "v3", _result("请确认", status="waiting_confirmation"), verdict).passed
    written = grade(case, "v3", _result("已添加", status="completed", results=[{"capability": "state.mutate", "status": "completed"}]), verdict)
    assert not written.passed and not written.status_ok and not written.writes_ok


# --- runner, offline ----------------------------------------------------------------

def _offline_run(tmp_path, cases, **kwargs):
    bank = Bank(tmp_path / "bank")
    bank.save("s_fault_market_error", [])
    bank.save("demo", [{"capability": "market.performance", "arguments": {"ticker": "NVDA"}, "result": json.loads(json.dumps({"capability": "market.performance", "status": "completed", "subject": "NVDA", "evidence": [{"id": "demo-price", "entity": "NVDA", "claim": "NVDA 收盘价为 120 美元。", "metric": "close", "value": 120, "unit": "USD", "source_id": "market_data"}], "limitations": [], "errors": [], "metrics": {}, "findings": [], "metadata": {}}))}])
    run = Run(label="t", mode="offline", versions=("v2", "v3"), seconds=20, workdir=tmp_path / "runs", bank=bank, judge=kwargs.get("judge"), repeat=kwargs.get("repeat", 1))
    run.build()
    try:
        rows = run.run(cases)
    finally:
        run.close()
    return run, rows


def test_offline_run_writes_a_ledger_for_both_versions(tmp_path):
    demo = BenchCase("demo", "market", "查询 NVDA 的行情（离线演示）", ("给出了价格",), must_cite=("market_data",))
    fault = by_id(dev_cases())["s_fault_market_error"]
    accepting = lambda question, answer, criteria, forbidden: {"criteria": [{"index": i, "met": True} for i in range(len(criteria))], "forbidden": [{"index": i, "asserted": False} for i in range(len(forbidden))]}
    run, rows = _offline_run(tmp_path, [demo, fault], judge=accepting, repeat=2)
    assert len(rows) == 8 and {r["version"] for r in rows} == {"v2", "v3"}
    ledger = read_ledger(run.root)
    assert len(ledger) == 8 and (run.root / "conditions.json").exists() and (run.root / "results" / "demo-v3-1.json").exists()
    v3_demo = next(r for r in ledger if r["case_id"] == "demo" and r["version"] == "v3" and r["attempt"] == 1)
    assert v3_demo["status"] == "completed" and "120" in v3_demo["answer"] and v3_demo["score"]["passed"] and v3_demo["bank_sha"]
    v3_fault = next(r for r in ledger if r["case_id"] == "s_fault_market_error" and r["version"] == "v3")
    assert "120" not in v3_fault["answer"] and v3_fault["status"] != "completed"
    summary = summarize(ledger)
    assert summary["versions"]["v3"]["cases"] == 2 and summary["attempts"] == 8
    folded = fold_attempts(ledger)
    assert folded[("demo", "v3")]["attempts"] == 2 and folded[("demo", "v3")]["passed"]
    path = write_report(run.root, ledger)
    text = path.read_text(encoding="utf-8")
    assert "Rubric pass rate" in text and "| demo | market |" in text and (run.root / "summary.json").exists()


# --- pairwise -----------------------------------------------------------------------

def test_pairwise_judging_is_blind_and_measures_position_bias():
    case = BenchCase("p", "market", "q", ("更完整",))
    seen = []
    def prefers_longer(question, criteria, a, b):
        seen.append((a, b))
        winner = "A" if len(a) > len(b) else "B"
        return {"criteria": [{"index": 0, "better": winner}], "winner": winner, "reason": "longer"}
    answers = {"v2": "短 [P1] Agent V2", "v3": "这是一份长得多的回答 [evidence-abc] Agent V3 写的"}
    verdict = compare_pair(prefers_longer, case, answers, seed=7)
    assert verdict["outcome"] == "v3" and verdict["winners_by_order"] == ["v3", "v3"]
    assert verdict["sides"] in ({"A": "v2", "B": "v3"}, {"A": "v3", "B": "v2"}) and assign_sides("p", 7) == verdict["sides"]
    flat = " ".join(a + b for a, b in seen)
    assert "Agent V2" not in flat and "Agent V3" not in flat and "[P1]" not in flat and "[evidence-abc]" not in flat, "no version tells reach the judge"
    assert seen[0] == (seen[1][1], seen[1][0]), "second call swaps the sides"
    def always_a(question, criteria, a, b):
        return {"criteria": [{"index": 0, "better": "A"}], "winner": "A", "reason": "first"}
    assert compare_pair(always_a, case, answers, seed=7)["outcome"] == "position_dependent"
    stats = pair_summary([verdict, compare_pair(always_a, case, answers, seed=7)])
    assert stats["outcomes"] == {"v3": 1, "position_dependent": 1} and stats["position_dependent_rate"] == 0.5 and stats["v3_win_rate_among_decided"] == 1.0
    assert scrub("见 [abc-1] 和 Agent V3") == "见 [#] 和 本系统"
    assert "Pairwise" in render(summarize([{"case_id": "p", "version": "v3", "category": "market", "set": "dev", "elapsed_s": 1, "tokens": {"input": 1, "output": 1}, "score": {"passed": True, "fixture_missing": 0, "judged": True, "problems": []}}]), None, stats)
