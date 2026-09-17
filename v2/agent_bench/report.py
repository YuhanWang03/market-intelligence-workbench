"""Turn a ledger (and optional pair verdicts) into numbers a reader can act on."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _rate(passed: int, total: int) -> str:
    return f"{passed}/{total} ({passed / total:.0%})" if total else "0/0"


def fold_attempts(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """Per (case, version): majority pass over attempts, mean elapsed, mean tokens, max fixture_missing."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["case_id"], row["version"])].append(row)
    folded: dict[tuple[str, str], dict[str, Any]] = {}
    for key, items in groups.items():
        passes = sum(1 for r in items if r["score"]["passed"])
        folded[key] = {
            "case_id": key[0], "version": key[1], "category": items[0]["category"], "set": items[0]["set"], "attempts": len(items), "passes": passes,
            "passed": passes * 2 > len(items), "flaky": 0 < passes < len(items),
            "elapsed_s": round(sum(r["elapsed_s"] for r in items) / len(items), 1),
            "tokens_in": round(sum(r["tokens"]["input"] for r in items) / len(items)), "tokens_out": round(sum(r["tokens"]["output"] for r in items) / len(items)),
            "fixture_missing": max(r["score"]["fixture_missing"] for r in items), "fixture_approximate": max(r.get("fixture_approximate", 0) for r in items), "judged": all(r["score"]["judged"] for r in items),
            "problems": sorted({p for r in items for p in r["score"]["problems"]})[:4],
        }
    return folded


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    folded = fold_attempts(rows)
    versions = sorted({r["version"] for r in rows})
    out: dict[str, Any] = {"versions": {}, "cases": len({r["case_id"] for r in rows}), "attempts": len(rows)}
    for version in versions:
        mine = [f for f in folded.values() if f["version"] == version]
        by_category: dict[str, dict[str, int]] = defaultdict(lambda: {"passed": 0, "total": 0})
        by_set: dict[str, dict[str, int]] = defaultdict(lambda: {"passed": 0, "total": 0})
        for f in mine:
            for table, key in ((by_category, f["category"]), (by_set, f["set"])):
                table[key]["total"] += 1
                table[key]["passed"] += int(f["passed"])
        problems = Counter()
        for f in mine:
            for p in f["problems"]:
                problems[p.split("：")[0].split("（")[0][:24]] += 1
        out["versions"][version] = {
            "cases": len(mine), "passed": sum(f["passed"] for f in mine), "flaky": sum(f["flaky"] for f in mine), "unjudged": sum(not f["judged"] for f in mine),
            "fixture_missing_cases": sum(1 for f in mine if f["fixture_missing"]), "fixture_approximate_cases": sum(1 for f in mine if f["fixture_approximate"]),
            "mean_elapsed_s": round(sum(f["elapsed_s"] for f in mine) / len(mine), 1) if mine else 0,
            "mean_tokens_in": round(sum(f["tokens_in"] for f in mine) / len(mine)) if mine else 0,
            "mean_tokens_out": round(sum(f["tokens_out"] for f in mine) / len(mine)) if mine else 0,
            "by_category": dict(sorted(by_category.items())), "by_set": dict(by_set), "top_problems": problems.most_common(6),
        }
    return out


def pair_summary(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    outcomes = Counter(p["outcome"] for p in pairs)
    by_category: dict[str, Counter] = defaultdict(Counter)
    for p in pairs:
        by_category[p["category"]][p["outcome"]] += 1
    decided = outcomes.get("v2", 0) + outcomes.get("v3", 0)
    return {"pairs": len(pairs), "outcomes": dict(outcomes), "decided": decided, "position_dependent_rate": round(outcomes.get("position_dependent", 0) / len(pairs), 2) if pairs else 0,
            "v3_win_rate_among_decided": round(outcomes.get("v3", 0) / decided, 2) if decided else None, "by_category": {k: dict(v) for k, v in sorted(by_category.items())}}


def render(summary: dict[str, Any], conditions: dict[str, Any] | None = None, pairs: dict[str, Any] | None = None, folded: dict[tuple[str, str], dict[str, Any]] | None = None) -> str:
    lines = ["# Agent bench report", ""]
    if conditions:
        lines += [f"- label: `{conditions.get('label')}` · mode: `{conditions.get('mode')}` · model: `{conditions.get('model')}` · thinking: {conditions.get('thinking')} · budget: {conditions.get('max_seconds')}s · repeat: {conditions.get('repeat')} · bank: `{conditions.get('bank_sha')}`", ""]
    lines += ["## Rubric pass rate (majority over attempts)", "", "| version | cases | passed | flaky | unjudged | fixture-missing cases | approximate-fixture cases | mean s | mean tokens in/out |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for version, v in summary["versions"].items():
        lines.append(f"| {version} | {v['cases']} | {_rate(v['passed'], v['cases'])} | {v['flaky']} | {v['unjudged']} | {v['fixture_missing_cases']} | {v.get('fixture_approximate_cases', 0)} | {v['mean_elapsed_s']} | {v['mean_tokens_in']}/{v['mean_tokens_out']} |")
    categories = sorted({c for v in summary["versions"].values() for c in v["by_category"]})
    if categories:
        lines += ["", "### By category", "", "| category | " + " | ".join(summary["versions"]) + " |", "| --- |" + " --- |" * len(summary["versions"])]
        for category in categories:
            lines.append(f"| {category} | " + " | ".join(_rate(v["by_category"].get(category, {}).get("passed", 0), v["by_category"].get(category, {}).get("total", 0)) for v in summary["versions"].values()) + " |")
    sets = sorted({s for v in summary["versions"].values() for s in v["by_set"]})
    if len(sets) > 1:
        lines += ["", "### By set", "", "| set | " + " | ".join(summary["versions"]) + " |", "| --- |" + " --- |" * len(summary["versions"])]
        for s in sets:
            lines.append(f"| {s} | " + " | ".join(_rate(v["by_set"].get(s, {}).get("passed", 0), v["by_set"].get(s, {}).get("total", 0)) for v in summary["versions"].values()) + " |")
    for version, v in summary["versions"].items():
        if v["top_problems"]:
            lines += ["", f"### Most common problems · {version}", ""] + [f"- {count} × {text}" for text, count in v["top_problems"]]
    if pairs:
        lines += ["", "## Pairwise blind comparison", "", f"- pairs: {pairs['pairs']} · outcomes: {pairs['outcomes']} · position-dependent: {pairs['position_dependent_rate']:.0%} · V3 win rate among decided: {pairs['v3_win_rate_among_decided']}", ""]
        if pairs["by_category"]:
            lines += ["| category | v2 | v3 | tie | position_dependent |", "| --- | --- | --- | --- | --- |"]
            for category, counts in pairs["by_category"].items():
                lines.append(f"| {category} | {counts.get('v2', 0)} | {counts.get('v3', 0)} | {counts.get('tie', 0)} | {counts.get('position_dependent', 0)} |")
    if folded:
        lines += ["", "## Cases", "", "| case | category | " + " | ".join(summary["versions"]) + " |", "| --- | --- |" + " --- |" * len(summary["versions"])]
        for case_id in sorted({k[0] for k in folded}):
            cells = []
            for version in summary["versions"]:
                f = folded.get((case_id, version))
                cells.append("-" if f is None else (("PASS" if f["passed"] else "FAIL") + ("~" if f["flaky"] else "") + (f" · {f['problems'][0][:40]}" if f["problems"] and not f["passed"] else "")))
            category = next(f["category"] for k, f in folded.items() if k[0] == case_id)
            lines.append(f"| {case_id} | {category} | " + " | ".join(cells) + " |")
    lines += ["", "_Rubric pass rates and pairwise outcomes are judge-based; sample a share of cases for human review before quoting them._"]
    return "\n".join(lines)


def write_report(root: Path, rows: list[dict[str, Any]], pairs: list[dict[str, Any]] | None = None) -> Path:
    root = Path(root)
    conditions = json.loads((root / "conditions.json").read_text(encoding="utf-8")) if (root / "conditions.json").exists() else None
    summary = summarize(rows)
    pair_stats = pair_summary(pairs) if pairs else None
    (root / "summary.json").write_text(json.dumps({"summary": summary, "pairs": pair_stats, "conditions": conditions}, ensure_ascii=False, indent=1), encoding="utf-8")
    path = root / "report.md"
    path.write_text(render(summary, conditions, pair_stats, fold_attempts(rows)), encoding="utf-8")
    return path
