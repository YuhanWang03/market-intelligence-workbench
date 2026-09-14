"""Nassim Taleb — antifragility, tail risk and convex payoffs.

Ported from ``src/agents/nassim_taleb.py`` at virattt/ai-hedge-fund
v2026.5.14 (MIT).  The seven scoring functions (tail risk, antifragility,
convexity, fragility via negativa, skin in the game, volatility regime and
the black-swan sentinel) are kept rule-for-rule, including the pandas/numpy
arithmetic over the daily price series; the LangGraph state, progress bar
and LLM call are gone.  Upstream derived no signal in code — the LLM read
``score / max_score`` — so the verdict is the shared base rule over the
same ratio.  ``safe_float`` and ``prices_to_df`` are reproduced privately
here so the module has no upstream dependency.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from v2.personas.base import Persona
from v2.personas.models import Evaluation
from v2.personas.snapshot import PersonaSnapshot


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Upstream ``safe_float``: float or ``default`` for NaN / non-numeric."""
    try:
        if pd.isna(value) or np.isnan(value):
            return default
        return float(value)
    except (ValueError, TypeError, OverflowError):
        return default


def _prices_to_df(prices: list) -> pd.DataFrame:
    """Upstream ``prices_to_df``: DatetimeIndex from ``time``, numeric OHLCV, ascending."""
    if not prices:
        return pd.DataFrame()
    df = pd.DataFrame([p.to_dict() for p in prices])
    if "time" not in df.columns:
        return pd.DataFrame()
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    for col in ("open", "close", "high", "low", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan
    df.sort_index(inplace=True)
    return df


class NassimTaleb(Persona):
    key = "nassim_taleb"
    name = "Nassim Taleb"
    name_zh = "纳西姆·塔勒布"
    style = "focuses on tail risk, antifragility, and asymmetric payoffs"
    period = "ttm"
    lookback = 10
    #: upstream fetched line items with ``limit=5`` while metrics used 10
    line_item_lookback = 5
    #: upstream fetched at most 100 news rows
    news_limit = 100
    needs = frozenset({"insiders", "news", "prices"})
    requires = frozenset({"metrics", "line_items", "prices"})
    system_prompt = (
        "You are Nassim Taleb. Decide bullish, bearish, or neutral using only the provided facts.\n"
        "Checklist: antifragility (benefits from disorder); tail risk (fat tails, skewness); convexity "
        "(asymmetric payoff); fragility via negativa; skin in the game (insiders); volatility regime (low vol = danger).\n"
        "Signal rules: bullish = antifragile with convex payoff AND not fragile; bearish = fragile (high leverage, "
        "thin margins, volatile earnings) OR no skin in the game; neutral = mixed or insufficient data.\n"
        "Confidence: 90-100 truly antifragile, strong convexity, skin in the game; 70-89 low fragility, decent "
        "optionality; 50-69 mixed fragility; 30-49 some fragility, weak insider alignment; 10-29 clearly fragile "
        "or dangerous vol regime.\n"
        "Use Taleb's vocabulary: antifragile, convexity, via negativa, barbell, turkey problem, Lindy effect. "
        "Keep reasoning under 150 characters. Do not invent data."
    )

    # ------------------------------------------------------------------ rules

    def evaluate(self, snap: PersonaSnapshot) -> Evaluation:
        metrics = snap.metrics(self.period, self.lookback)
        items = snap.line_items(self.period, self.line_item_lookback)
        market_cap = snap.market_cap
        prices_df = _prices_to_df(snap.prices)
        insiders = list(snap.insider_trades)
        news = list(snap.news)[: self.news_limit]

        tail_risk = self.analyze_tail_risk(prices_df)
        antifragility = self.analyze_antifragility(metrics, items, market_cap)
        convexity = self.analyze_convexity(metrics, items, prices_df, market_cap)
        fragility = self.analyze_fragility(metrics, items)
        skin = self.analyze_skin_in_game(insiders)
        vol_regime = self.analyze_volatility_regime(prices_df)
        black_swan = self.analyze_black_swan_sentinel(news, prices_df)

        ev = Evaluation(parts=[
            self.part("tail_risk", tail_risk["score"], tail_risk["max_score"], tail_risk["details"]),
            self.part("antifragility", antifragility["score"], antifragility["max_score"], antifragility["details"]),
            self.part("convexity", convexity["score"], convexity["max_score"], convexity["details"]),
            self.part("fragility", fragility["score"], fragility["max_score"], fragility["details"]),
            self.part("skin_in_game", skin["score"], skin["max_score"], skin["details"]),
            self.part("volatility_regime", vol_regime["score"], vol_regime["max_score"], vol_regime["details"]),
            self.part("black_swan", black_swan["score"], black_swan["max_score"], black_swan["details"]),
        ])

        latest_item = items[0] if items else None
        latest_metrics = metrics[0] if metrics else None
        fcf = latest_item.free_cash_flow if latest_item else None
        cash = latest_item.cash_and_equivalents if latest_item else None
        fcf_yield = None
        if fcf is not None and market_cap and market_cap > 0:
            fcf_yield = fcf / market_cap
        elif latest_metrics is not None:
            fcf_yield = latest_metrics.free_cash_flow_yield
        ev.facts = {
            "market_cap": market_cap,
            "fcf_yield": fcf_yield,
            "cash_to_market_cap": (cash / market_cap) if (cash is not None and market_cap) else None,
            "debt_to_equity": latest_metrics.debt_to_equity if latest_metrics else None,
            "interest_coverage": latest_metrics.interest_coverage if latest_metrics else None,
            "price_days": int(len(prices_df)),
            "insider_trades": len(insiders),
            "news_items": len(news),
        }
        if not metrics and not items and prices_df.empty:
            ev.insufficient = True
            ev.notes.append("no fundamentals or prices")
        return ev

    # ---------------------------------------------------- upstream functions

    @staticmethod
    def analyze_tail_risk(prices_df: pd.DataFrame) -> dict[str, Any]:
        """Assess fat tails, skewness, tail ratio, and max drawdown."""
        if prices_df.empty or len(prices_df) < 20:
            return {"score": 0, "max_score": 8, "details": "Insufficient price data for tail risk analysis"}

        score = 0
        reasoning = []
        returns = prices_df["close"].pct_change().dropna()

        if len(returns) >= 63:
            kurt = _safe_float(returns.rolling(63).kurt().iloc[-1])
        else:
            kurt = _safe_float(returns.kurt())
        if kurt > 5:
            score += 2
            reasoning.append(f"Extremely fat tails (kurtosis {kurt:.1f})")
        elif kurt > 2:
            score += 1
            reasoning.append(f"Moderate fat tails (kurtosis {kurt:.1f})")
        else:
            reasoning.append(f"Near-Gaussian tails (kurtosis {kurt:.1f}) — suspiciously thin")

        if len(returns) >= 63:
            skew = _safe_float(returns.rolling(63).skew().iloc[-1])
        else:
            skew = _safe_float(returns.skew())
        if skew > 0.5:
            score += 2
            reasoning.append(f"Positive skew ({skew:.2f}) favors long convexity")
        elif skew > -0.5:
            score += 1
            reasoning.append(f"Symmetric distribution (skew {skew:.2f})")
        else:
            reasoning.append(f"Negative skew ({skew:.2f}) — crash-prone")

        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        if len(positive_returns) > 20 and len(negative_returns) > 20:
            right_tail = np.percentile(positive_returns, 95)
            left_tail = abs(np.percentile(negative_returns, 5))
            tail_ratio = right_tail / left_tail if left_tail > 0 else 1.0
            if tail_ratio > 1.2:
                score += 2
                reasoning.append(f"Asymmetric upside (tail ratio {tail_ratio:.2f})")
            elif tail_ratio > 0.8:
                score += 1
                reasoning.append(f"Balanced tails (tail ratio {tail_ratio:.2f})")
            else:
                reasoning.append(f"Asymmetric downside (tail ratio {tail_ratio:.2f})")
        else:
            reasoning.append("Insufficient data for tail ratio")

        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = _safe_float(drawdown.min())
        if max_dd > -0.15:
            score += 2
            reasoning.append(f"Resilient (max drawdown {max_dd:.1%})")
        elif max_dd > -0.30:
            score += 1
            reasoning.append(f"Moderate drawdown ({max_dd:.1%})")
        else:
            reasoning.append(f"Severe drawdown ({max_dd:.1%}) — fragile")

        return {"score": score, "max_score": 8, "details": "; ".join(reasoning)}

    @staticmethod
    def analyze_antifragility(metrics: list, line_items: list, market_cap: float | None) -> dict[str, Any]:
        """Evaluate whether the company benefits from disorder: low debt, high cash, stable margins."""
        if not metrics and not line_items:
            return {"score": 0, "max_score": 10, "details": "Insufficient data for antifragility analysis"}

        score = 0
        reasoning = []
        latest_metrics = metrics[0] if metrics else None
        latest_item = line_items[0] if line_items else None

        cash = latest_item.cash_and_equivalents if latest_item else None
        total_debt = latest_item.total_debt if latest_item else None
        total_assets = latest_item.total_assets if latest_item else None
        if cash is not None and total_debt is not None:
            net_cash = cash - total_debt
            if net_cash > 0 and market_cap and cash > 0.20 * market_cap:
                score += 3
                reasoning.append(f"War chest: net cash ${net_cash:,.0f}, cash is {cash / market_cap:.0%} of market cap")
            elif net_cash > 0:
                score += 2
                reasoning.append(f"Net cash positive (${net_cash:,.0f})")
            elif total_assets and total_debt < 0.30 * total_assets:
                score += 1
                reasoning.append("Net debt but manageable relative to assets")
            else:
                reasoning.append("Leveraged position — not antifragile")
        else:
            reasoning.append("Cash/debt data not available")

        debt_to_equity = latest_metrics.debt_to_equity if latest_metrics else None
        if debt_to_equity is not None:
            if debt_to_equity < 0.3:
                score += 2
                reasoning.append(f"Taleb-approved low leverage (D/E {debt_to_equity:.2f})")
            elif debt_to_equity < 0.7:
                score += 1
                reasoning.append(f"Moderate leverage (D/E {debt_to_equity:.2f})")
            else:
                reasoning.append(f"High leverage (D/E {debt_to_equity:.2f}) — fragile")
        else:
            reasoning.append("Debt-to-equity data not available")

        op_margins = [m.operating_margin for m in metrics if m.operating_margin is not None]
        if len(op_margins) >= 3:
            mean_margin = sum(op_margins) / len(op_margins)
            variance = sum((m - mean_margin) ** 2 for m in op_margins) / len(op_margins)
            std_margin = variance ** 0.5
            cv = std_margin / abs(mean_margin) if mean_margin != 0 else float("inf")
            if cv < 0.15 and mean_margin > 0.15:
                score += 3
                reasoning.append(f"Stable high margins (avg {mean_margin:.1%}, CV {cv:.2f}) — antifragile pricing power")
            elif cv < 0.30 and mean_margin > 0.10:
                score += 2
                reasoning.append(f"Reasonable margin stability (avg {mean_margin:.1%}, CV {cv:.2f})")
            elif cv < 0.30:
                score += 1
                reasoning.append(f"Margins somewhat stable (CV {cv:.2f}) but low (avg {mean_margin:.1%})")
            else:
                reasoning.append(f"Volatile margins (CV {cv:.2f}) — fragile pricing power")
        else:
            reasoning.append("Insufficient margin history for stability analysis")

        fcf_values = [item.free_cash_flow for item in line_items] if line_items else []
        fcf_values = [v for v in fcf_values if v is not None]
        if fcf_values:
            positive_count = sum(1 for v in fcf_values if v > 0)
            if positive_count == len(fcf_values):
                score += 2
                reasoning.append(f"Consistent FCF generation ({positive_count}/{len(fcf_values)} periods positive)")
            elif positive_count > len(fcf_values) / 2:
                score += 1
                reasoning.append(f"Majority positive FCF ({positive_count}/{len(fcf_values)} periods)")
            else:
                reasoning.append(f"Inconsistent FCF ({positive_count}/{len(fcf_values)} periods positive)")
        else:
            reasoning.append("FCF data not available")

        return {"score": score, "max_score": 10, "details": "; ".join(reasoning)}

    @staticmethod
    def analyze_convexity(metrics: list, line_items: list, prices_df: pd.DataFrame, market_cap: float | None) -> dict[str, Any]:
        """Measure asymmetric payoff potential: R&D optionality, upside/downside ratio, cash optionality."""
        if not metrics and not line_items and prices_df.empty:
            return {"score": 0, "max_score": 10, "details": "Insufficient data for convexity analysis"}

        score = 0
        reasoning = []
        latest_item = line_items[0] if line_items else None

        rd = latest_item.research_and_development if latest_item else None
        revenue = latest_item.revenue if latest_item else None
        if rd is not None and revenue and revenue > 0:
            rd_ratio = abs(rd) / revenue
            if rd_ratio > 0.15:
                score += 3
                reasoning.append(f"Significant embedded optionality via R&D ({rd_ratio:.1%} of revenue)")
            elif rd_ratio > 0.08:
                score += 2
                reasoning.append(f"Meaningful R&D investment ({rd_ratio:.1%} of revenue)")
            elif rd_ratio > 0.03:
                score += 1
                reasoning.append(f"Modest R&D ({rd_ratio:.1%} of revenue)")
            else:
                reasoning.append(f"Minimal R&D ({rd_ratio:.1%} of revenue)")
        else:
            reasoning.append("R&D data not available — no penalty for non-R&D sectors")

        if not prices_df.empty and len(prices_df) >= 20:
            returns = prices_df["close"].pct_change().dropna()
            upside = returns[returns > 0]
            downside = returns[returns < 0]
            if len(upside) > 10 and len(downside) > 10:
                avg_up = upside.mean()
                avg_down = abs(downside.mean())
                up_down_ratio = avg_up / avg_down if avg_down > 0 else 1.0
                if up_down_ratio > 1.3:
                    score += 2
                    reasoning.append(f"Convex return profile (up/down ratio {up_down_ratio:.2f})")
                elif up_down_ratio > 1.0:
                    score += 1
                    reasoning.append(f"Slight positive asymmetry (up/down ratio {up_down_ratio:.2f})")
                else:
                    reasoning.append(f"Concave returns (up/down ratio {up_down_ratio:.2f}) — unfavorable")
            else:
                reasoning.append("Insufficient return data for asymmetry analysis")
        else:
            reasoning.append("Insufficient price data for return asymmetry analysis")

        cash = latest_item.cash_and_equivalents if latest_item else None
        if cash is not None and market_cap and market_cap > 0:
            cash_ratio = cash / market_cap
            if cash_ratio > 0.30:
                score += 3
                reasoning.append(f"Cash is a call option on future opportunities ({cash_ratio:.0%} of market cap)")
            elif cash_ratio > 0.15:
                score += 2
                reasoning.append(f"Strong cash position ({cash_ratio:.0%} of market cap)")
            elif cash_ratio > 0.05:
                score += 1
                reasoning.append(f"Moderate cash buffer ({cash_ratio:.0%} of market cap)")
            else:
                reasoning.append(f"Low cash relative to market cap ({cash_ratio:.0%})")
        else:
            reasoning.append("Cash/market cap data not available")

        latest_metrics = metrics[0] if metrics else None
        fcf_yield = None
        if latest_item and market_cap and market_cap > 0:
            fcf = latest_item.free_cash_flow
            if fcf is not None:
                fcf_yield = fcf / market_cap
        if fcf_yield is None and latest_metrics:
            fcf_yield = latest_metrics.free_cash_flow_yield
        if fcf_yield is not None:
            if fcf_yield > 0.10:
                score += 2
                reasoning.append(f"High FCF yield ({fcf_yield:.1%}) provides margin for convex bet")
            elif fcf_yield > 0.05:
                score += 1
                reasoning.append(f"Decent FCF yield ({fcf_yield:.1%})")
            else:
                reasoning.append(f"Low FCF yield ({fcf_yield:.1%})")
        else:
            reasoning.append("FCF yield data not available")

        return {"score": score, "max_score": 10, "details": "; ".join(reasoning)}

    @staticmethod
    def analyze_fragility(metrics: list, line_items: list) -> dict[str, Any]:
        """Via Negativa: detect fragile companies. High score = NOT fragile."""
        if not metrics:
            return {"score": 0, "max_score": 8, "details": "Insufficient data for fragility analysis"}

        score = 0
        reasoning = []
        latest_metrics = metrics[0]

        debt_to_equity = latest_metrics.debt_to_equity
        if debt_to_equity is not None:
            if debt_to_equity > 2.0:
                reasoning.append(f"Extremely fragile balance sheet (D/E {debt_to_equity:.2f})")
            elif debt_to_equity > 1.0:
                score += 1
                reasoning.append(f"Elevated leverage (D/E {debt_to_equity:.2f})")
            elif debt_to_equity > 0.5:
                score += 2
                reasoning.append(f"Moderate leverage (D/E {debt_to_equity:.2f})")
            else:
                score += 3
                reasoning.append(f"Low leverage (D/E {debt_to_equity:.2f}) — not fragile")
        else:
            reasoning.append("Debt-to-equity data not available")

        interest_coverage = latest_metrics.interest_coverage
        if interest_coverage is not None:
            if interest_coverage > 10:
                score += 2
                reasoning.append(f"Interest coverage {interest_coverage:.1f}x — debt is irrelevant")
            elif interest_coverage > 5:
                score += 1
                reasoning.append(f"Comfortable interest coverage ({interest_coverage:.1f}x)")
            else:
                reasoning.append(f"Low interest coverage ({interest_coverage:.1f}x) — fragile to rate changes")
        else:
            reasoning.append("Interest coverage data not available")

        earnings_growth_values = [m.earnings_growth for m in metrics if m.earnings_growth is not None]
        if len(earnings_growth_values) >= 3:
            mean_eg = sum(earnings_growth_values) / len(earnings_growth_values)
            variance = sum((e - mean_eg) ** 2 for e in earnings_growth_values) / len(earnings_growth_values)
            std_eg = variance ** 0.5
            if std_eg < 0.20:
                score += 2
                reasoning.append(f"Stable earnings (growth std {std_eg:.2f}) — robust")
            elif std_eg < 0.50:
                score += 1
                reasoning.append(f"Moderate earnings volatility (growth std {std_eg:.2f})")
            else:
                reasoning.append(f"Highly volatile earnings (growth std {std_eg:.2f}) — fragile")
        else:
            reasoning.append("Insufficient earnings history for volatility analysis")

        net_margin = latest_metrics.net_margin
        if net_margin is not None:
            if net_margin > 0.15:
                score += 1
                reasoning.append(f"Fat margins ({net_margin:.1%}) buffer shocks")
            elif net_margin >= 0.05:
                reasoning.append(f"Moderate margins ({net_margin:.1%})")
            else:
                reasoning.append(f"Paper-thin margins ({net_margin:.1%}) — one shock away from loss")
        else:
            reasoning.append("Net margin data not available")

        score = max(score, 0)
        return {"score": score, "max_score": 8, "details": "; ".join(reasoning)}

    @staticmethod
    def analyze_skin_in_game(insider_trades: list) -> dict[str, Any]:
        """Assess insider alignment: net insider buying signals trust."""
        if not insider_trades:
            return {"score": 1, "max_score": 4, "details": "No insider trade data — neutral assumption"}

        score = 0
        reasoning = []
        shares_bought = sum(t.transaction_shares or 0 for t in insider_trades if (t.transaction_shares or 0) > 0)
        shares_sold = abs(sum(t.transaction_shares or 0 for t in insider_trades if (t.transaction_shares or 0) < 0))
        net = shares_bought - shares_sold

        if net > 0:
            buy_sell_ratio = net / max(shares_sold, 1)
            if buy_sell_ratio > 2.0:
                score = 4
                reasoning.append(f"Strong skin in the game — net insider buying {net:,} shares (ratio {buy_sell_ratio:.1f}x)")
            elif buy_sell_ratio > 0.5:
                score = 3
                reasoning.append(f"Moderate insider conviction — net buying {net:,} shares")
            else:
                score = 2
                reasoning.append(f"Net insider buying of {net:,} shares")
        else:
            reasoning.append(f"Insiders selling — no skin in the game (net {net:,} shares)")

        return {"score": score, "max_score": 4, "details": "; ".join(reasoning)}

    @staticmethod
    def analyze_volatility_regime(prices_df: pd.DataFrame) -> dict[str, Any]:
        """Volatility regime analysis. Key Taleb insight: low vol is dangerous (turkey problem)."""
        if prices_df.empty or len(prices_df) < 30:
            return {"score": 0, "max_score": 6, "details": "Insufficient price data for volatility analysis"}

        score = 0
        reasoning = []
        returns = prices_df["close"].pct_change().dropna()
        hist_vol = returns.rolling(21).std() * math.sqrt(252)

        if len(hist_vol.dropna()) >= 63:
            vol_ma = hist_vol.rolling(63).mean()
            current_vol = _safe_float(hist_vol.iloc[-1])
            avg_vol = _safe_float(vol_ma.iloc[-1])
            vol_regime = current_vol / avg_vol if avg_vol > 0 else 1.0
        elif len(hist_vol.dropna()) >= 21:
            current_vol = _safe_float(hist_vol.iloc[-1])
            avg_vol = _safe_float(hist_vol.mean())
            vol_regime = current_vol / avg_vol if avg_vol > 0 else 1.0
        else:
            return {"score": 0, "max_score": 6, "details": "Insufficient data for volatility regime analysis"}

        if vol_regime < 0.7:
            reasoning.append(f"Dangerously low vol (regime {vol_regime:.2f}) — turkey problem")
        elif vol_regime < 0.9:
            score += 1
            reasoning.append(f"Below-average vol (regime {vol_regime:.2f}) — approaching complacency")
        elif vol_regime <= 1.3:
            score += 3
            reasoning.append(f"Normal vol regime ({vol_regime:.2f}) — fair pricing")
        elif vol_regime <= 2.0:
            score += 4
            reasoning.append(f"Elevated vol (regime {vol_regime:.2f}) — opportunity for the antifragile")
        else:
            score += 2
            reasoning.append(f"Extreme vol (regime {vol_regime:.2f}) — crisis mode")

        if len(hist_vol.dropna()) >= 42:
            vol_of_vol = hist_vol.rolling(21).std()
            vol_of_vol_clean = vol_of_vol.dropna()
            if len(vol_of_vol_clean) > 0:
                current_vov = _safe_float(vol_of_vol_clean.iloc[-1])
                median_vov = _safe_float(vol_of_vol_clean.median())
                if median_vov > 0:
                    if current_vov > 2 * median_vov:
                        score += 2
                        reasoning.append(f"Highly unstable vol (vol-of-vol {current_vov:.4f} vs median {median_vov:.4f}) — regime change likely")
                    elif current_vov > median_vov:
                        score += 1
                        reasoning.append(f"Elevated vol-of-vol ({current_vov:.4f} vs median {median_vov:.4f})")
                    else:
                        reasoning.append(f"Stable vol-of-vol ({current_vov:.4f})")
                else:
                    reasoning.append("Vol-of-vol median is zero — unusual")
            else:
                reasoning.append("Insufficient vol-of-vol data")
        else:
            reasoning.append("Insufficient history for vol-of-vol analysis")

        return {"score": score, "max_score": 6, "details": "; ".join(reasoning)}

    @staticmethod
    def analyze_black_swan_sentinel(news: list, prices_df: pd.DataFrame) -> dict[str, Any]:
        """Monitor for crisis signals: abnormal news sentiment, volume spikes, price dislocations."""
        score = 2  # Default: normal conditions
        reasoning = []

        neg_ratio = 0.0
        if news:
            total = len(news)
            neg_count = sum(1 for n in news if n.sentiment and str(n.sentiment).lower() in ["negative", "bearish"])
            neg_ratio = neg_count / total if total > 0 else 0
        else:
            reasoning.append("No recent news data")

        volume_spike = 1.0
        recent_return = 0.0
        if not prices_df.empty and len(prices_df) >= 10:
            if "volume" in prices_df.columns:
                recent_vol = prices_df["volume"].iloc[-5:].mean()
                avg_vol = prices_df["volume"].iloc[-63:].mean() if len(prices_df) >= 63 else prices_df["volume"].mean()
                volume_spike = recent_vol / avg_vol if avg_vol > 0 else 1.0
                volume_spike = _safe_float(volume_spike, 1.0)
            if len(prices_df) >= 5:
                recent_return = _safe_float(prices_df["close"].iloc[-1] / prices_df["close"].iloc[-5] - 1)

        if neg_ratio > 0.7 and volume_spike > 2.0:
            score = 0
            reasoning.append(f"Black swan warning — {neg_ratio:.0%} negative news, {volume_spike:.1f}x volume spike")
        elif neg_ratio > 0.5 or volume_spike > 2.5:
            score = 1
            reasoning.append(f"Elevated stress signals (neg news {neg_ratio:.0%}, volume {volume_spike:.1f}x)")
        elif neg_ratio > 0.3 and abs(recent_return) > 0.10:
            score = 1
            reasoning.append(f"Moderate stress with price dislocation ({recent_return:.1%} move, {neg_ratio:.0%} negative news)")
        elif neg_ratio < 0.3 and volume_spike < 1.5:
            score = 3
            reasoning.append("No black swan signals detected")
        else:
            reasoning.append(f"Normal conditions (neg news {neg_ratio:.0%}, volume {volume_spike:.1f}x)")

        if neg_ratio > 0.4 and volume_spike < 1.5 and score < 4:
            score = min(score + 1, 4)
            reasoning.append("Contrarian opportunity — negative sentiment without panic selling")

        return {"score": score, "max_score": 4, "details": "; ".join(reasoning)}
