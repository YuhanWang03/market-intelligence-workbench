"""Pydantic models for lateral-expansion (玩法 ③) results."""

from __future__ import annotations

from pydantic import BaseModel, Field

from v2.screening.models import ScreenCandidate


CATEGORIES: list[str] = ["supplier", "customer", "smaller_peer", "beneficiary"]

CATEGORY_LABEL_CN: dict[str, str] = {
    "supplier": "供应商",
    "customer": "客户",
    "smaller_peer": "同业小市值",
    "beneficiary": "间接受益方",
}


class Label(BaseModel):
    """One (seed, category, reason) annotation for a neighbor.

    Same ticker can have many labels (e.g. AMD shows up as both a
    "smaller_peer" of NVDA and a "beneficiary" of AAPL's M-chip momentum).
    """

    seed: str
    category: str          # one of CATEGORIES
    reason: str            # LLM's one-line rationale, ≤ 30 chars
    evidence_status: str = 'UNCHECKED'
    evidence_url: str | None = None
    evidence_text: str = ''
    evidence_title: str = ''


class Neighbor(BaseModel):
    """One unique ticker discovered as a neighbor of one or more seeds."""

    ticker: str
    labels: list[Label] = Field(default_factory=list)

    # Filled by verify.py
    exists: bool = False
    sector: str | None = None
    already_in_universe: bool = False

    # Filled by orchestrator (only for tickers that are real & new)
    candidate: ScreenCandidate | None = None
    passed_filter: bool = False
    failed_reason: str = ""           # e.g. "毛利 26% < 50%"

    # Filled by narrator (only for tickers that pass the filter)
    bull: str = ""
    bear: str = ""

    # Phase Resume-3: Tavily relation verification
    # True if Tavily search confirms the seed-neighbor relationship exists
    relation_verified: bool = False
    relation_evidence_url: str | None = None
    relation_checked: bool = False    # was a check attempted (vs not yet checked)


class LateralResult(BaseModel):
    """Top-level result returned by run_lateral_expansion()."""

    date: str
    seeds: list[str]
    neighbors: list[Neighbor] = Field(default_factory=list)

    llm_tokens: int = 0
    api_calls: int = 0          # FD + yfinance
    tavily_calls: int = 0       # relation verification (Step 3)
    warnings: list[str] = Field(default_factory=list)
    candidate_errors: list[dict[str, str]] = Field(default_factory=list)
    api_call_counts: dict[str, int] = Field(default_factory=dict)
