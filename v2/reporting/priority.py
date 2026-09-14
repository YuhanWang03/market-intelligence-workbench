"""Unified push importance scoring.

Every notifier.send_text caller computes a PriorityResult and passes it
down so the notifier can:

  P0 (≥ 80)  immediate Telegram + 🚨🚨🚨 prefix + dashboard red chip
  P1 (60-79) immediate Telegram + dashboard blue chip
  P2 (40-59) archive only (rolled up by the daily digest cron)
  P3 (< 40)  archive only, dashboard hides by default

The base score for each kind of push is in BASE_SCORES; metadata
adjustments (price moves, holdings, surprise, etc.) compose on top.
Pure Python rules — no LLM involved.

Public surface used by callers:
  - compute_importance(event_kind, metadata) -> PriorityResult
  - tier_emoji_prefix(tier) -> str   (for Telegram message decoration)
  - tier_chip_color(tier)   -> str   (Tailwind class string for dashboard)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


PriorityTier = Literal["P0", "P1", "P2", "P3"]


# Neutral base by event kind. Callers should use these stable keys
# (don't free-form). Unrecognized kinds fall back to "default" / P2.
BASE_SCORES: dict[str, int] = {
    "alert_fire":          85,  # user /alert triggered — always P0
    "intraday_anomaly":    65,  # mid-session anomaly — P1
    "anomaly_attribution": 60,  # post-close anomaly — base P1, adjustable
    "screen_result":       55,  # daily screen — P2 by default
    "lateral_expansion":   55,  # supply-chain expansion — P2
    "etf_daily":           50,  # ARK daily snapshot — P2
    "etf_significant":     75,  # ARK significant rebalance — P1
    "institutional_13f":   65,  # 13F push — P1
    "earnings_summary":    70,  # earnings recap — P1
    "earnings_reminder":   45,  # generic earnings reminder — P2 (legacy)
    "earnings_reminder_d3":45,  # D-3 reminder — P2
    "earnings_reminder_d1":60,  # D-1 reminder — P1 (upgraded urgency)
    "earnings_reminder_d0":60,  # D-0 release day — P1
    "earnings_pending":    45,  # FD hasn't ingested yet — P2 placeholder
    "portfolio_risk":      55,  # portfolio risk daily — P2 base
    "portfolio_alert":     85,  # portfolio drawdown / concentration — P0
    "macro_event":         70,  # macro data print — P1
    "macro_critical":      90,  # FOMC / CPI — P0
    "regime_change":       80,  # market regime shift — P0
    "sec_filing":          50,  # routine filing — P2 (legacy, kept for compat)
    "sec_critical":        85,  # 8-K material event — P0 (legacy)
    # Phase 3 SEC monitoring — graduated per-tier 8-K + Form 4 kinds
    "sec_8k_p0":           85,  # 1.03/1.05/2.04/3.01/4.02/5.01/5.02(senior)
    "sec_8k_p1":           65,  # 1.01/1.02/2.01/2.03/2.05/2.06/4.01/5.02(other)
    "sec_8k_p2":           55,  # 3.02/3.03/5.08/7.01/8.01 (2.02 routed to ⑧)
    "sec_8k_p3":           35,  # 1.04/5.03/5.04/5.05/5.07/9.01
    "sec_form4_purchase":  75,  # insider P-code base P1
    "sec_form4_sale":      50,  # insider S-code base P2 (default noise)
    "sec_form4_cluster":   75,  # ≥3 distinct insiders same-day same-direction
    "sec_insider_digest":  55,  # ⑫b Fri 19:15 ET — P2 floor, P1 on unusual ≥3
    # Phase 5a ⑬ ARK Alerts — graduated base ladder; cron primarily emits
    # at ark_alert_p1 and lets metadata adjustments escalate. p0/p2 entries
    # reserved for future direct-tier callers.
    "ark_alert_p2":        55,
    "ark_alert_p1":        65,
    "ark_alert_p0":        85,
    # Phase 4 Macro Agent — daily snapshot + release-driven kinds
    "macro_release_p2":    55,  # in-line print (CPI/PCE/NFP/GDP/PPI/FOMC)
    "macro_release_p1":    65,  # 1-2σ surprise → P1
    "macro_release_p0":    85,  # ≥3σ surprise OR FOMC SEP shift → P0
    "macro_snapshot_p3":   35,  # ⑭ daily ambient (VIX/yields/DXY/WTI/gold)
    "macro_vix_spike":     85,  # VIX +20% single-day → P0
    "macro_curve_flip":    65,  # T10Y2Y sign change today → P1
    "macro_weekly":        65,  # ⑰ Fri 19:30 ET — operator-visibility floor
    # ⑱ Money-flow divergence — P2 base (rolls into daily digest); a
    # "strong" verdict or a held/watchlist hit escalates it to P1.
    "moneyflow_divergence": 55,
    "scheduler_status":    30,  # scheduler startup message — P3
    "error_alert":         75,  # error reported by @notify_on_error — P1
    "p2_digest":           65,  # the digest itself is P1
    "default":             55,  # unrecognized → P2
}


@dataclass(frozen=True)
class PriorityResult:
    score: int
    tier: PriorityTier
    reasons: list[str]   # human-readable adjustment trail, for trace + debug


def compute_importance(
    event_kind: str,
    metadata: dict | None = None,
) -> PriorityResult:
    """Score a push from 0 to 100 and bucket it into P0–P3.

    metadata fields the scorer reads (all optional):
      - reasons_count: int           — Tavily-Verifier reasons that survived
      - flags: list[str]              — anomaly chips (e.g. "contrarian_move")
      - price_change_pct: float       — fraction, abs value matters
      - surprise_pct: float           — earnings surprise fraction
      - daily_pnl_pct: float          — portfolio daily PnL fraction (negative = loss)
      - top_1_pct: float              — share of largest single position [0, 1]
      - max_drawdown_pct: float       — 1M peak-trough drawdown (signed; abs used)
      - n_earnings_next_7d: int       — held-position earnings in next 7 days
      - is_held_position: bool        — ticker is in the user's broker account
      - is_watchlist: bool            — ticker is on the user's watchlist
      - action: str                   — ETF rebalance action ("exit", "buy", ...)
      - ticker_held_by_user: bool     — ETF rebalance touches a held position
      - sec_item: str                 — 8-K item code ("2.02", "5.02", ...)
    """
    md = metadata or {}
    base = BASE_SCORES.get(event_kind, BASE_SCORES["default"])
    adjustments: list[tuple[int, str]] = []

    # ---- anomaly attribution ----
    if event_kind == "anomaly_attribution":
        reasons = int(md.get("reasons_count") or 0)
        if reasons == 0:
            adjustments.append((-25, "no_tavily_reasons"))
        elif reasons >= 3:
            adjustments.append((+10, "rich_attribution"))

        flags = md.get("flags") or []
        if "contrarian_move" in flags:
            adjustments.append((+15, "contrarian"))

        pct = abs(float(md.get("price_change_pct") or 0.0))
        if pct >= 0.05:
            adjustments.append((+10, f"big_move_{pct:.2%}"))

    # ---- earnings ----
    if event_kind == "earnings_summary":
        surprise = abs(float(md.get("surprise_pct") or 0.0))
        if surprise >= 0.10:
            # +30 puts even a no-holdings BEAT/MISS firmly into P0
            # (70 base → 100 capped). Big surprises are the whole reason
            # to wake the user at 21:00 ET.
            adjustments.append((+30, f"big_surprise_{surprise:.1%}"))
        if md.get("guidance_lowered"):
            adjustments.append((+10, "guidance_lowered"))
        # Phase 3.5 — 10-Q auditor / regulator flags. ⑧ surfaces these
        # via the optional ten_q_delta block; the cron threads the
        # flags into compute_importance metadata so the priority
        # escalation is auditable in the trace.
        if md.get("has_going_concern"):
            adjustments.append((+20, "going_concern_in_10q"))
        if md.get("has_material_weakness"):
            adjustments.append((+15, "material_weakness_in_10q"))

    # ---- portfolio ----
    if event_kind == "portfolio_risk":
        # daily_pnl_pct — single-day loss is the only factor that on its
        # own pushes to P0. Everything else (concentration / drawdown /
        # earnings density) is "concerning, not urgent" and sits in P1.
        pnl_pct = float(md.get("daily_pnl_pct") or 0.0)
        if pnl_pct <= -0.05:
            adjustments.append((+30, f"daily_loss_{pnl_pct:.1%}"))   # → P0
        elif pnl_pct <= -0.02:
            adjustments.append((+10, "moderate_loss"))

        # Single-ticker concentration — graduated ladder.
        top_1 = float(md.get("top_1_pct") or 0.0)
        if top_1 >= 0.30:
            adjustments.append((+20, f"top1_{top_1:.0%}"))
        elif top_1 >= 0.20:
            adjustments.append((+10, f"top1_{top_1:.0%}"))

        # 1-month drawdown — a single significant DD bumps to P1.
        # (abs so callers can pass either signed or unsigned values.)
        max_dd = abs(float(md.get("max_drawdown_pct") or 0.0))
        if max_dd >= 0.10:
            adjustments.append((+15, f"drawdown_{max_dd:.0%}"))

        # Event density — multiple earnings in a week = elevated
        # cluster risk (a single bad print whipsaws the whole book).
        n_earnings = int(md.get("n_earnings_next_7d") or 0)
        if n_earnings >= 3:
            adjustments.append((+10, f"earnings_density_{n_earnings}"))

    # ---- Phase 6 ⑱ money-flow divergence ----
    if event_kind == "moneyflow_divergence":
        # "strong" (RSI in a confirming zone or an RSI divergence) is the
        # only thing that lifts a no-holdings signal out of the digest into
        # an immediate P1 push. "moderate" stays P2 by design.
        if md.get("strength") == "strong":
            adjustments.append((+10, "strong_divergence"))
        if md.get("rsi_divergence") in ("bullish", "bearish"):
            adjustments.append((+5, f"rsi_{md['rsi_divergence']}_divergence"))

    # ---- universal: holdings / watchlist matter for everything ----
    if md.get("is_held_position"):
        adjustments.append((+15, "held_position"))
    elif md.get("is_watchlist"):
        adjustments.append((+10, "watchlist"))

    # ---- ETF significant rebalance ----
    if event_kind == "etf_significant":
        if md.get("action") == "exit":
            adjustments.append((+10, "etf_exit"))
        if md.get("ticker_held_by_user"):
            adjustments.append((+15, "etf_touches_holding"))

    # ---- Phase 3 SEC 8-K (graduated kinds) ----
    if event_kind.startswith("sec_8k"):
        if md.get("is_amendment"):
            adjustments.append((-5, "amendment"))
        if event_kind == "sec_8k_p0" and md.get("has_senior_exec"):
            # 5.02 + LLM-confirmed senior exec → small extra nudge
            adjustments.append((+5, "ceo_cfo_5_02_confirmed"))

    # ---- Phase 3 Form 4 individual purchase ----
    if event_kind == "sec_form4_purchase":
        usd = float(md.get("transaction_usd") or 0)
        if usd >= 1_000_000:
            adjustments.append((+25, f"big_purchase_${usd/1e6:.1f}M"))
        elif usd >= 100_000:
            adjustments.append((+10, "moderate_purchase"))
        role = md.get("insider_role")
        if role in {"CEO", "CFO"}:
            adjustments.append((+10, f"{role}_purchase"))
        if md.get("is_10b5_1"):
            # Pre-planned purchases mute the signal (rare — most P-code
            # purchases are discretionary)
            adjustments.append((-10, "10b5_1_plan_purchase"))

    # ---- Phase 3 Form 4 individual sale ----
    if event_kind == "sec_form4_sale":
        usd = abs(float(md.get("transaction_usd") or 0))
        if usd >= 10_000_000 and not md.get("is_10b5_1"):
            adjustments.append((+15, f"big_discretionary_sale_${usd/1e6:.1f}M"))
        elif usd >= 1_000_000 and md.get("is_10b5_1"):
            adjustments.append((-5, "10b5_1_plan_sale"))

    # ---- Phase 5a ARK Alerts (⑬ Mon-Fri 08:30 ET) ----
    # Adjustments don't depend on is_held_position / is_watchlist (those
    # already fire via the universal block above for any kind that carries
    # them). The dedicated is_in_user_universe flag mirrors the
    # held-or-watchlist semantics specifically for the ARK alert flow
    # — kept as a separate reason so the audit trail is readable
    # (`held_or_watchlist_ark` vs the generic `held_position`).
    if event_kind.startswith("ark_alert"):
        if md.get("is_in_user_universe"):
            adjustments.append((+10, "held_or_watchlist_ark"))
        if md.get("is_multi_fund"):
            adjustments.append((+15, "multi_fund_coordination"))
        action = md.get("action")
        # today_weight + yesterday_weight metadata are in DECIMAL
        # fraction units (0.02 = 2%) per the cron's metadata-build
        # contract — see scripts/ark_alerts_to_telegram.py. The cron
        # divides ArkAlert.today_weight (CSV pct unit) by 100 before
        # passing here so the `:.1%` reason format renders cleanly.
        if action == "new_position":
            tw = float(md.get("today_weight") or 0.0)
            if tw >= 0.02:
                adjustments.append((+10, f"large_new_position_{tw:.1%}"))
        elif action == "liquidated":
            yw = float(md.get("yesterday_weight") or 0.0)
            if yw >= 0.02:
                adjustments.append((+10, f"large_liquidation_{yw:.1%}"))

    # ---- Phase 3.5 weekly insider digest (⑫b Fri 19:15 ET) ----
    if event_kind == "sec_insider_digest":
        # P2 (55) base. Bump to P1 when ≥3 tickers exceeded the
        # _UNUSUAL_PUSH_THRESHOLD this week — that's a coordinated
        # signal worth pulling out of the digest roll-up.
        n_unusual = int(md.get("unusual_ticker_count") or 0)
        if n_unusual >= 3:
            adjustments.append((+10, f"unusual_tickers_{n_unusual}"))

    # ---- Phase 3 Form 4 cluster (≥3 distinct insiders same day) ----
    if event_kind == "sec_form4_cluster":
        n = int(md.get("transaction_count") or 0)
        if n >= 5:
            adjustments.append((+15, f"large_cluster_{n}"))
        elif n >= 3:
            adjustments.append((+5, f"cluster_{n}"))
        if md.get("direction") == "purchase":
            adjustments.append((+10, "cluster_buy"))

    # ---- SEC 8-K item codes ----
    if event_kind == "sec_critical":
        item = str(md.get("sec_item") or "")
        if item.startswith("2."):
            adjustments.append((+5, "earnings_item"))
        elif item.startswith("5."):
            adjustments.append((+5, "exec_change"))

    # ---- Phase 4 Macro release (CPI / PCE / NFP / GDP / PPI / FOMC) ----
    if event_kind.startswith("macro_release"):
        sigma = abs(float(md.get("surprise_sigma") or 0))
        if sigma >= 3.0:
            adjustments.append((+20, f"extreme_surprise_{sigma:.1f}sigma"))
        elif sigma >= 2.0:
            adjustments.append((+10, f"big_surprise_{sigma:.1f}sigma"))
        elif sigma >= 1.0:
            adjustments.append((+5, f"moderate_surprise_{sigma:.1f}sigma"))

        if md.get("is_fomc") and md.get("sep_shift") in (
            "hawkish_shift", "dovish_shift",
        ):
            adjustments.append((+15, f"sep_{md['sep_shift']}"))

        if md.get("sell_side_consensus") == "hawkish_unexpected":
            adjustments.append((+10, "sell_side_hawkish"))

    # ---- Phase 4 VIX spike (⑭ snapshot) ----
    if event_kind == "macro_vix_spike":
        vix_pct = float(md.get("vix_pct_change_1d") or 0)
        if vix_pct >= 0.30:
            adjustments.append((+20, f"vix_extreme_+{vix_pct:.0%}"))
        elif vix_pct >= 0.20:
            adjustments.append((+10, f"vix_strong_+{vix_pct:.0%}"))

    # ---- Phase 4 yield curve flip ----
    if event_kind == "macro_curve_flip":
        adjustments.append((+10, "yield_curve_inverted"))

    raw = base + sum(d for d, _ in adjustments)
    score = max(0, min(100, raw))
    tier: PriorityTier = (
        "P0" if score >= 80 else
        "P1" if score >= 60 else
        "P2" if score >= 40 else
        "P3"
    )

    reasons_log = [f"base={base}"] + [
        f"{d:+d}_{label}" for d, label in adjustments
    ]
    return PriorityResult(score=score, tier=tier, reasons=reasons_log)


def tier_emoji_prefix(tier: PriorityTier) -> str:
    """Telegram message prefix for visual emphasis.

    P0 gets a 🚨 cluster so it's unmistakable on the lock screen.
    P1 stays clean — most pushes are P1, prefixes would dilute.
    P2 prefix only ever shows up on the daily digest header itself.
    P3 never reaches Telegram so its prefix is moot.
    """
    return {
        "P0": "🚨🚨🚨 ",
        "P1": "",
        "P2": "📋 ",
        "P3": "",
    }[tier]


def tier_chip_color(tier: PriorityTier) -> str:
    """Tailwind class string for the dashboard's priority chip."""
    return {
        "P0": "bg-rose-500 text-white",
        "P1": "bg-blue-500 text-white",
        "P2": "bg-amber-400 text-amber-900",
        "P3": "bg-slate-300 text-slate-600",
    }[tier]


__all__ = [
    "PriorityTier",
    "PriorityResult",
    "BASE_SCORES",
    "compute_importance",
    "tier_emoji_prefix",
    "tier_chip_color",
]
