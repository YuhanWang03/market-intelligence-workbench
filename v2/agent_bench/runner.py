"""Run cases through both agents and keep one ledger row per attempt.

Ledger: ``<workdir>/<label>/ledger.jsonl`` (append-only, one JSON object per
attempt) plus ``results/<case>-<version>-<attempt>.json`` with the full
serialised result.  ``conditions.json`` records the model, budget, mode and
bank digest so two labels can be compared honestly.
"""
from __future__ import annotations

import json
import logging
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from v2.agent_bench import agents as agent_builders
from v2.agent_bench.bank import Bank, Recorder, Replay
from v2.agent_bench.cases import BenchCase
from v2.agent_bench.judge import RubricJudgeFn, Score, grade, verdict_is_valid

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKDIR = _PROJECT_ROOT / "data" / "agent_bench" / "runs"


def _preference(turn: str) -> str:
    try:
        from v2.agent_v2.memory import parse_feedback
    except ImportError:  # pragma: no cover
        return ""
    parsed = parse_feedback(turn)
    return str(parsed.get("note") or "") if parsed and parsed.get("kind") == "preference" else ""


def _tokens(result: dict[str, Any]) -> dict[str, int]:
    usage = (result.get("synthesis") or {}).get("usage") or []
    total = {"input": 0, "output": 0}
    for row in usage:
        if isinstance(row, dict):
            total["input"] += int(row.get("input_tokens") or 0)
            total["output"] += int(row.get("output_tokens") or 0)
    return total


class Run:
    def __init__(self, *, label: str, mode: str, versions: tuple[str, ...], seconds: float = 180, workdir: Path | None = None, bank: Bank | None = None,
                 judge: RubricJudgeFn | None = None, debate: bool = False, repeat: int = 1, progress: Callable[[str], None] | None = None, seed_watchlist: tuple[str, ...] = ()) -> None:
        if mode not in agent_builders.MODES:
            raise ValueError(f"unknown mode: {mode}")
        self.label, self.mode, self.versions, self.seconds, self.repeat, self.debate = label, mode, tuple(versions), seconds, max(1, repeat), debate
        self.seed_watchlist = tuple(seed_watchlist)
        self.root = Path(workdir or DEFAULT_WORKDIR) / label
        self.bank = bank or Bank()
        self.judge = judge
        self.progress = progress or (lambda message: None)
        self.agents: dict[str, Any] = {}
        self.replay: dict[str, Replay] = {}
        self.recorder: dict[str, Recorder] = {}

    # -- setup ---------------------------------------------------------------------------
    def build(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "results").mkdir(exist_ok=True)
        settings = agent_builders.configure_model() if self.mode != "offline" else {"model": "(offline)", "base_url": "", "thinking": "", "temperature": 0}
        if self.seed_watchlist and self.mode in {"live", "record"}:
            agent_builders.seed_watchlist(self.root / "agents", self.seed_watchlist)
        for version in self.versions:
            agent = agent_builders.build_agent(version, mode=self.mode, seconds=self.seconds, workdir=self.root / "agents" / version, debate=self.debate)
            if self.mode in {"frozen", "offline"}:
                self.replay[version] = Replay(version)
                agent_builders.attach_replay(agent, version, self.replay[version])
            elif self.mode == "record":
                self.recorder[version] = Recorder(version)
                agent_builders.attach_recorder(agent, version, self.recorder[version])
            self.agents[version] = agent
        conditions = {**settings, "label": self.label, "mode": self.mode, "versions": list(self.versions), "max_seconds": self.seconds, "repeat": self.repeat, "debate": self.debate,
                      "web": True, "bank_sha": self.bank.sha() if self.mode in {"frozen", "offline"} else None, "seed_watchlist": list(self.seed_watchlist), "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                      "limitations": ["Same model, budget and web allowance for both agents; native planners, synthesizers and tool adapters differ by design.",
                                      "frozen: tool responses come from the bank; a call the bank lacks is answered as fixture_missing, never from a provider.",
                                      "live/record: data changes between the two agents' runs; alternate the version order across repeats."]}
        (self.root / "conditions.json").write_text(json.dumps(conditions, ensure_ascii=False, indent=1), encoding="utf-8")

    def close(self) -> None:
        for agent in self.agents.values():
            agent_builders.close_agent(agent)

    # -- one attempt ----------------------------------------------------------------------
    def attempt(self, case: BenchCase, version: str, attempt: int) -> dict[str, Any]:
        agent = self.agents[version]
        session_id = f"bench-{self.label}-{case.id}-{version}-{attempt}"
        if self.mode in {"frozen", "offline"}:
            self.replay[version].use(self.bank.load(case.id), fault=case.fault, fixtures=case.fixtures)
        elif self.mode == "record":
            self.recorder[version].use(fault=case.fault)
        started = time.monotonic()
        error = ""
        result: dict[str, Any] = {}
        try:
            for turn in case.preceding:
                note = _preference(turn)
                if note and agent_builders.set_preference(agent, version, session_id, note):
                    continue
                agent.run(turn, session_id=session_id, allow_web=case.allow_web)
            result = agent.run(case.question, session_id=session_id, allow_web=case.allow_web).to_dict()
        except Exception as exc:  # noqa: BLE001 — a crashed attempt is a failed row, not a stopped run
            error = f"{type(exc).__name__}: {str(exc)[:300]}"
            result = {"status": "failed", "answer": "", "error": error, "route": {"kind": ""}, "results": [], "evidence": [], "verification": {"ok": False, "warnings": []}, "traceback": traceback.format_exc()[-2000:]}
        elapsed = time.monotonic() - started
        fixture_missing = len(self.replay[version].missing()) if version in self.replay else 0
        fixture_approximate = len(self.replay[version].approximate()) if version in self.replay else 0
        verdict = None
        if self.judge is not None and result.get("answer"):
            for _ in range(2):  # one retry for a malformed verdict, as V2's grader does
                try:
                    verdict = self.judge(case.question, str(result.get("answer") or ""), list(case.criteria), list(case.forbidden))
                except Exception as exc:  # noqa: BLE001 — an unjudged row is recorded as such
                    verdict = None
                    error = error or f"judge: {type(exc).__name__}: {str(exc)[:200]}"
                    break
                if verdict_is_valid(verdict):
                    break
            if verdict is not None and not verdict_is_valid(verdict):
                error = error or "judge: malformed verdict twice"
        graded_case = case
        if not self.debate and "debater" in (case.expectations.get(version) or {}).get("agents", []):
            from dataclasses import replace as _replace
            scoped = {**case.expectations, version: {**case.expectations[version], "agents": [a for a in case.expectations[version]["agents"] if a != "debater"]}}
            graded_case = _replace(case, expectations=scoped)
        score = grade(graded_case, version, result, verdict, fixture_missing=fixture_missing)
        if self.mode == "record":
            self.bank.save(case.id, self.recorder[version].records)
        row = self._row(case, version, attempt, result, score, elapsed, error)
        row["fixture_approximate"] = fixture_approximate
        self._write(row, result)
        return row

    def _row(self, case: BenchCase, version: str, attempt: int, result: dict[str, Any], score: Score, elapsed: float, error: str) -> dict[str, Any]:
        return {
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "label": self.label, "mode": self.mode, "bank_sha": self.bank.sha() if self.mode in {"frozen", "offline"} else None,
            "case_id": case.id, "category": case.category, "set": case.set, "tags": list(case.tags), "origin": case.origin, "question": case.question,
            "version": version, "attempt": attempt, "status": str(result.get("status") or ""), "route": str((result.get("route") or {}).get("kind") or ""),
            "capabilities": [row.get("capability") for row in result.get("results") or []], "evidence_count": len(result.get("evidence") or []),
            "verification_ok": bool((result.get("verification") or {}).get("ok")), "synthesis": str((result.get("synthesis") or {}).get("outcome") or ""),
            "debate": (result.get("synthesis") or {}).get("debate") or {}, "tokens": _tokens(result), "elapsed_s": round(elapsed, 1), "run_id": result.get("run_id"),
            "answer": str(result.get("answer") or ""), "error": error, "score": score.to_dict(),
        }

    def _write(self, row: dict[str, Any], result: dict[str, Any]) -> None:
        with (self.root / "ledger.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        (self.root / "results" / f"{row['case_id']}-{row['version']}-{row['attempt']}.json").write_text(json.dumps({"row": row, "result": result}, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    # -- the loop ------------------------------------------------------------------------------
    def run(self, cases: list[BenchCase]) -> list[dict[str, Any]]:
        """Every (case, version, attempt) not already in this label's ledger; rerunning a label resumes it."""
        rows: list[dict[str, Any]] = []
        total = len(cases) * len(self.versions) * self.repeat
        done = 0
        finished = {(r["case_id"], r["version"], r["attempt"]) for r in read_ledger(self.root)}
        if finished:
            self.progress(f"RESUME {self.label}: {len(finished)} attempts already in the ledger will be skipped")
        for attempt in range(1, self.repeat + 1):
            for index, case in enumerate(cases):
                if case.frozen_only and self.mode not in {"frozen", "offline"}:
                    self.progress(f"SKIP  {case.id}: frozen-only case in {self.mode} mode")
                    continue
                # Alternate the order per case and attempt so a shared cache warms for each version equally often.
                order = list(self.versions) if (index + attempt) % 2 == 0 else list(reversed(self.versions))
                for version in order:
                    done += 1
                    if (case.id, version, attempt) in finished:
                        continue
                    self.progress(f"START [{done}/{total}] {case.id} {version} #{attempt}")
                    row = self.attempt(case, version, attempt)
                    rows.append(row)
                    score = row["score"]
                    verdict = "PASS" if score["passed"] else ("RECORDED" if self.mode == "record" and self.judge is None else ("UNJUDGED" if not score["judged"] else "FAIL"))
                    problems = [p for p in score["problems"] if p != "未经裁判评分"]
                    self.progress(f"END   [{done}/{total}] {case.id} {version} #{attempt} {verdict} {row['status']} {row['elapsed_s']}s"
                                  + (f" missing={score['fixture_missing']}" if score["fixture_missing"] else "") + (f" approx={row['fixture_approximate']}" if row.get("fixture_approximate") else "") + (f" :: {'; '.join(problems[:2])}" if problems else ""))
        return rows


def drop_unjudged(root: Path) -> int:
    """Remove rows a dead network left ungraded (judge or model unreachable) so a resume runs them again."""
    rows = read_ledger(root)
    keep = [row for row in rows if row["score"]["judged"] or not row.get("error")]
    if len(keep) != len(rows):
        (Path(root) / "ledger.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in keep), encoding="utf-8")
    return len(rows) - len(keep)


def read_ledger(root: Path) -> list[dict[str, Any]]:
    path = Path(root) / "ledger.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
