"""Pydantic models for the anomaly monitoring pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ScoredReason(BaseModel):
    """One attribution reason annotated with the Verifier LLM's confidence (Phase B ① C).

    confidence: 高 = direct, company-level + magnitude-matched cause
                中 = relevant but weak/indirect link
                低 = market-wide noise, unverified rumor, magnitude-mismatched
    """

    text: str
    confidence: Literal["高", "中", "低"] = "中"
    note: str = ""   # short Verifier comment (optional, may be empty)


class HistoricalAnomaly(BaseModel):
    """A past anomaly retrieved by ChromaDB RAG (Phase C)."""

    date: str
    flags: str       # comma-separated flag list (e.g. "52w_high,volume_spike")
    doc: str         # the embedded document text — useful for showing context
    metadata: dict = {}  # governance fields on a retro record: confidence, version, written_at, top_reason


class MonitorConfig(BaseModel):
    """Thresholds for anomaly detection."""

    volume_spike_threshold: float = 3.0       # today's vol / 30d avg >= this
    high_52w_threshold: float = 0.99          # close >= 52w_high * this fires "new high"
    low_52w_threshold: float = 1.01           # close <= 52w_low * this fires "new low"
    sparkline_days: int = 7                   # how many recent closes to chart

    # Insider trading (Phase B 3a)
    insider_lookback_days: int = 30           # how far back to scan Form 4 filings
    insider_buy_min_value: float = 1_000_000.0    # net buying ≥ this fires "insider_buying"
    insider_sell_min_value: float = 5_000_000.0   # net selling ≥ this fires "insider_selling"


class InsiderExec(BaseModel):
    """One notable executive trade for display in the alert card."""

    name: str
    title: str
    direction: str          # "buy" or "sell"
    value: float            # USD


class InsiderActivity(BaseModel):
    """Aggregated insider activity over the lookback window."""

    net_value: float                          # buy_value - sell_value
    buy_value: float
    sell_value: float
    trade_count: int                          # open-market trades only
    executives: list[InsiderExec] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Options unusual activity (Phase B ③b)
# ---------------------------------------------------------------------------


class OptionsSnapshot(BaseModel):
    """One-day options-chain snapshot (nearest expiration)."""

    ticker: str
    date: str
    call_oi: int
    call_volume: int
    put_oi: int
    put_volume: int
    expiration: str | None = None


class OptionsBurst(BaseModel):
    """Result of comparing today's chain to the 14-day baseline."""

    side: Literal["call", "put"]
    ratio: float                              # current_oi / baseline_avg
    current_oi: int
    baseline_avg_oi: int
    baseline_days: int                        # how many days the baseline covers


class NewsSource(BaseModel):
    """One news item referenced in the attribution."""

    title: str
    url: str


class Anomaly(BaseModel):
    """A ticker exhibiting one or more anomalous behaviors today.

    Multiple flags can co-fire (volume spike + 52w high is common and
    especially informative). Attribution fields are filled later.
    """

    ticker: str
    date: str                                 # YYYY-MM-DD — anomaly observation date

    # Price snapshot
    price: float
    price_change_pct: float                   # vs prior close

    # Volume snapshot
    volume_today: int
    volume_avg_30d: float
    volume_ratio: float                       # volume_today / volume_avg_30d

    # 52-week reference
    high_52w: float
    low_52w: float

    # Which signals fired (subset of {"volume_spike", "52w_high", "52w_low"})
    flags: list[str] = Field(default_factory=list)

    # Recent price closes for the sparkline chart (oldest -> newest)
    recent_prices: list[float] = Field(default_factory=list)

    # Phase B 3a: insider trading info (None if no significant activity)
    insider: InsiderActivity | None = None

    # Phase B 3b: options burst info (None during cold-start or no burst)
    options_burst: OptionsBurst | None = None
    options_snapshot: OptionsSnapshot | None = None   # always captured if available

    # Phase C: similar past anomalies retrieved from ChromaDB (None on cold start)
    historical_context: list[HistoricalAnomaly] = Field(default_factory=list)

    # ETF benchmarking — relative-strength vs sector ETF (None if ETF prices
    # unavailable or were not pre-fetched for this run).
    sector_etf: str | None = None
    sector_return_1d: float | None = None
    relative_1d_pp: float | None = None        # ticker_return - sector_return, in pp
    contrarian: bool = False                   # True when ticker and sector moved in opposite directions ≥ 1.5pp

    # Filled by attributor (Phase B ① C: reasons now scored by Verifier LLM)
    reasons: list[ScoredReason] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)   # actionable follow-ups
    sources: list[NewsSource] = Field(default_factory=list)
    tavily_calls: int = 0
    llm_tokens: int = 0       # combined Generator + Verifier tokens
    filtered_count: int = 0   # how many Tavily results entity-filter rejected
