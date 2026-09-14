"""Pydantic models for institutional 13F tracking (玩法 ④b)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ChangeType = Literal["new", "exit", "increase", "decrease"]


class Position(BaseModel):
    """A single equity position in one 13F filing."""

    cik: str
    accession: str
    quarter: str                # e.g. "2026-Q1"
    cusip: str
    ticker: str | None = None
    issuer_name: str
    shares: int
    market_value: float         # in dollars (13F reports in thousands; we convert)


class Filing(BaseModel):
    """Metadata for one 13F-HR filing."""

    cik: str
    manager_name: str
    accession: str
    quarter: str
    filing_date: str
    period_of_report: str       # quarter end date YYYY-MM-DD
    portfolio_value: float      # sum of all positions' market_value
    n_positions: int


class PositionChange(BaseModel):
    """One QoQ change for a single ticker held by a manager."""

    cik: str
    manager_name: str
    quarter: str                # the NEW quarter being reported
    change_type: ChangeType

    ticker: str | None = None
    issuer_name: str
    cusip: str

    # Current quarter state
    current_shares: int = 0
    current_value: float = 0
    current_pct: float = 0      # of current portfolio

    # Previous quarter state
    prev_shares: int = 0
    prev_value: float = 0
    prev_pct: float = 0

    # Flag if this ticker is in user's monitored universe (TECH_30 etc.)
    in_universe: bool = False

    # Filled by summarizer (LLM)
    interpretation: str = ""


class InstitutionalReport(BaseModel):
    """Top-level result returned per pipeline run."""

    date: str
    new_filings: list[Filing] = Field(default_factory=list)
    changes: list[PositionChange] = Field(default_factory=list)
    api_calls: int = 0
    llm_tokens: int = 0
