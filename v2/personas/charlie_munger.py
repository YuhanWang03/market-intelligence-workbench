"""Charlie Munger — wonderful businesses at fair prices, judged with mental models.

Ported from ``src/agents/charlie_munger.py`` at virattt/ai-hedge-fund
v2026.5.14 (MIT).  The four scoring functions (moat, management,
predictability, valuation) are kept rule-for-rule, including their 0-10
rescaling and the 35/25/25/15 weighting; the LangGraph state, progress bar
and LLM call are gone.  Upstream already fixed the confidence by code
(``compute_confidence``) and drew a pre-signal from the weighted total
(>= 7.5 bullish, <= 5.5 bearish); both live in :meth:`CharlieMunger.decide`.

One extension: upstream classified insider trades by ``transaction_type``,
a field the data provider never returned, so the insider rule never scored.
Here a trade with no ``transaction_type`` falls back to the sign of
``transaction_shares`` (see :func:`_insider_side`).

Second correction: the gross-margin trend loop compared neighbours as if
rows were oldest-first, crediting "consistently improving" to margins that
were falling. It now reads the newest-first order the data actually has.
"""

from __future__ import annotations

import math
from typing import Any

from v2.personas.base import Persona
from v2.personas.models import Evaluation, Signal
from v2.personas.snapshot import PersonaSnapshot

_WEIGHTS = {"moat": 0.35, "management": 0.25, "predictability": 0.25, "valuation": 0.15}


def _insider_side(trade: Any) -> str | None:
    """'buy' / 'sell' / None for one insider trade.

    Upstream looked only at ``transaction_type``; when that is absent the
    sign of ``transaction_shares`` says the same thing.
    """
    kind = getattr(trade, "transaction_type", None)
    if kind:
        kind = str(kind).lower()
        if kind in ("buy", "purchase"):
            return "buy"
        if kind in ("sell", "sale"):
            return "sell"
        return None
    shares = getattr(trade, "transaction_shares", None)
    if shares is None:
        return None
    try:
        shares = float(shares)
    except (TypeError, ValueError):
        return None
    if shares > 0:
        return "buy"
    if shares < 0:
        return "sell"
    return None


def _finite(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return value


class CharlieMunger(Persona):
    key = "charlie_munger"
    name = "Charlie Munger"
    name_zh = "查理·芒格"
    style = "only buys wonderful businesses at fair prices"
    period = "annual"
    lookback = 10
    needs = frozenset({"insiders", "news"})
    system_prompt = (
        "You are Charlie Munger. Decide bullish, bearish, or neutral using only the facts; keep reasoning "
        "under 120 characters and use the provided confidence exactly.\n"
        "Mental models: moat (consistent ROIC >15%, pricing power, low capex, R&D/intangibles); management "
        "(cash conversion, low debt, sensible cash, insider buying, shrinking share count); predictability "
        "(steady revenue, positive operating income, stable margins, reliable FCF); valuation (normalized FCF "
        "yield, margin of safety vs 15x FCF, FCF trend).\n"
        "Quality outweighs price: total = 0.35 moat + 0.25 management + 0.25 predictability + 0.15 valuation. "
        "Signal rules: bullish >= 7.5; bearish <= 5.5; neutral otherwise. Invert, always invert."
    )

    # ------------------------------------------------------------------ rules

    def evaluate(self, snap: PersonaSnapshot) -> Evaluation:
        metrics = snap.metrics(self.period, self.lookback)
        items = snap.line_items(self.period, self.lookback)
        market_cap = snap.market_cap
        insider_trades = snap.insider_trades
        news = snap.news

        moat = self.analyze_moat_strength(metrics, items)
        management = self.analyze_management_quality(items, insider_trades)
        predictability = self.analyze_predictability(items)
        valuation = self.calculate_munger_valuation(items, market_cap)

        parts = []
        for name, result in (("moat", moat), ("management", management), ("predictability", predictability), ("valuation", valuation)):
            weight = _WEIGHTS[name]
            parts.append(self.part(name, result["score"] * weight, 10 * weight, result["details"], raw_score=result["score"], raw_max=10, weight=weight))
        ev = Evaluation(parts=parts)

        mos = valuation.get("margin_of_safety_vs_fair_value")
        ev.margin_of_safety = mos
        ivr = valuation.get("intrinsic_value_range") or {}
        ev.facts = {
            "moat_score": moat["score"],
            "management_score": management["score"],
            "predictability_score": predictability["score"],
            "valuation_score": valuation["score"],
            "market_cap": market_cap,
            "fcf_yield": valuation.get("fcf_yield"),
            "normalized_fcf": valuation.get("normalized_fcf"),
            "intrinsic_value_conservative": ivr.get("conservative"),
            "intrinsic_value_reasonable": ivr.get("reasonable"),
            "intrinsic_value_optimistic": ivr.get("optimistic"),
            "margin_of_safety_vs_fair_value": mos,
            "insider_buy_ratio": management.get("insider_buy_ratio"),
            "recent_de_ratio": _finite(management.get("recent_de_ratio")),
            "cash_to_revenue": management.get("cash_to_revenue"),
            "share_count_trend": management.get("share_count_trend"),
            "news_sentiment": self.analyze_news_sentiment(news) if news else "No news data available",
        }
        if not metrics and not items:
            ev.insufficient = True
            ev.notes.append("no fundamentals")
        return ev

    def decide(self, ev: Evaluation) -> tuple[Signal, int]:
        total = ev.score  # already the upstream weighted 0-10 total
        if total >= 7.5:  # Munger has very high standards
            signal: Signal = "bullish"
        elif total <= 5.5:
            signal = "bearish"
        else:
            signal = "neutral"
        confidence = self.compute_confidence(
            moat=ev.facts.get("moat_score") or 0,
            mgmt=ev.facts.get("management_score") or 0,
            pred=ev.facts.get("predictability_score") or 0,
            val=ev.facts.get("valuation_score") or 0,
            mos=ev.margin_of_safety,
            signal=signal,
        )
        return signal, confidence

    # ---------------------------------------------------- upstream functions

    @staticmethod
    def analyze_moat_strength(metrics: list, items: list) -> dict[str, Any]:
        score = 0
        details = []
        if not metrics or not items:
            return {"score": 0, "details": "Insufficient data to analyze moat strength"}

        # 1. ROIC - Munger's favorite metric
        roic_values = [i.return_on_invested_capital for i in items if i.return_on_invested_capital is not None]
        if roic_values:
            high = sum(1 for r in roic_values if r > 0.15)
            if high >= len(roic_values) * 0.8:
                score += 3
                details.append(f"Excellent ROIC: >15% in {high}/{len(roic_values)} periods")
            elif high >= len(roic_values) * 0.5:
                score += 2
                details.append(f"Good ROIC: >15% in {high}/{len(roic_values)} periods")
            elif high > 0:
                score += 1
                details.append(f"Mixed ROIC: >15% in only {high}/{len(roic_values)} periods")
            else:
                details.append("Poor ROIC: Never exceeds 15% threshold")
        else:
            details.append("No ROIC data available")

        # 2. Pricing power - gross margin stability and trend
        gross_margins = [i.gross_margin for i in items if i.gross_margin is not None]
        if gross_margins and len(gross_margins) >= 3:
            # rows are newest-first: 'improving' means newer >= older
            margin_trend = sum(1 for i in range(1, len(gross_margins)) if gross_margins[i - 1] >= gross_margins[i])
            if margin_trend >= len(gross_margins) * 0.7:
                score += 2
                details.append("Strong pricing power: Gross margins consistently improving")
            elif sum(gross_margins) / len(gross_margins) > 0.3:
                score += 1
                details.append(f"Good pricing power: Average gross margin {sum(gross_margins) / len(gross_margins):.1%}")
            else:
                details.append("Limited pricing power: Low or declining gross margins")
        else:
            details.append("Insufficient gross margin data")

        # 3. Capital intensity
        if len(items) >= 3:
            capex_to_revenue = []
            for i in items:
                if i.capital_expenditure is not None and i.revenue is not None and i.revenue > 0:
                    capex_to_revenue.append(abs(i.capital_expenditure) / i.revenue)
            if capex_to_revenue:
                avg = sum(capex_to_revenue) / len(capex_to_revenue)
                if avg < 0.05:
                    score += 2
                    details.append(f"Low capital requirements: Avg capex {avg:.1%} of revenue")
                elif avg < 0.10:
                    score += 1
                    details.append(f"Moderate capital requirements: Avg capex {avg:.1%} of revenue")
                else:
                    details.append(f"High capital requirements: Avg capex {avg:.1%} of revenue")
            else:
                details.append("No capital expenditure data available")
        else:
            details.append("Insufficient data for capital intensity analysis")

        # 4. Intangibles - R&D and goodwill
        r_and_d = [i.research_and_development for i in items if i.research_and_development is not None]
        goodwill = [i.goodwill_and_intangible_assets for i in items if i.goodwill_and_intangible_assets is not None]
        if r_and_d and sum(r_and_d) > 0:
            score += 1
            details.append("Invests in R&D, building intellectual property")
        if goodwill:
            score += 1
            details.append("Significant goodwill/intangible assets, suggesting brand value or IP")

        final_score = min(10, score * 10 / 9)  # max raw score is 9
        return {"score": final_score, "details": "; ".join(details)}

    @staticmethod
    def analyze_management_quality(items: list, insider_trades: list) -> dict[str, Any]:
        score = 0
        details = []
        if not items:
            return {"score": 0, "details": "Insufficient data to analyze management quality"}

        # 1. Capital allocation - FCF to net income
        fcf_values = [i.free_cash_flow for i in items if i.free_cash_flow is not None]
        net_income_values = [i.net_income for i in items if i.net_income is not None]
        if fcf_values and net_income_values and len(fcf_values) == len(net_income_values):
            ratios = [fcf_values[i] / net_income_values[i] for i in range(len(fcf_values)) if net_income_values[i] and net_income_values[i] > 0]
            if ratios:
                avg = sum(ratios) / len(ratios)
                if avg > 1.1:
                    score += 3
                    details.append(f"Excellent cash conversion: FCF/NI ratio of {avg:.2f}")
                elif avg > 0.9:
                    score += 2
                    details.append(f"Good cash conversion: FCF/NI ratio of {avg:.2f}")
                elif avg > 0.7:
                    score += 1
                    details.append(f"Moderate cash conversion: FCF/NI ratio of {avg:.2f}")
                else:
                    details.append(f"Poor cash conversion: FCF/NI ratio of only {avg:.2f}")
            else:
                details.append("Could not calculate FCF to Net Income ratios")
        else:
            details.append("Missing FCF or Net Income data")

        # 2. Debt management
        recent_de_ratio = None
        debt_values = [i.total_debt for i in items if i.total_debt is not None]
        equity_values = [i.shareholders_equity for i in items if i.shareholders_equity is not None]
        if debt_values and equity_values and len(debt_values) == len(equity_values):
            recent_de_ratio = debt_values[0] / equity_values[0] if equity_values[0] > 0 else float("inf")
            if recent_de_ratio < 0.3:
                score += 3
                details.append(f"Conservative debt management: D/E ratio of {recent_de_ratio:.2f}")
            elif recent_de_ratio < 0.7:
                score += 2
                details.append(f"Prudent debt management: D/E ratio of {recent_de_ratio:.2f}")
            elif recent_de_ratio < 1.5:
                score += 1
                details.append(f"Moderate debt level: D/E ratio of {recent_de_ratio:.2f}")
            else:
                details.append(f"High debt level: D/E ratio of {recent_de_ratio:.2f}")
        else:
            details.append("Missing debt or equity data")

        # 3. Cash management efficiency
        cash_to_revenue = None
        cash_values = [i.cash_and_equivalents for i in items if i.cash_and_equivalents is not None]
        revenue_values = [i.revenue for i in items if i.revenue is not None]
        if cash_values and revenue_values:
            ratio = cash_values[0] / revenue_values[0] if revenue_values[0] > 0 else 0
            if 0.1 <= ratio <= 0.25:
                score += 2
                details.append(f"Prudent cash management: Cash/Revenue ratio of {ratio:.2f}")
            elif 0.05 <= ratio < 0.1 or 0.25 < ratio <= 0.4:
                score += 1
                details.append(f"Acceptable cash position: Cash/Revenue ratio of {ratio:.2f}")
            elif ratio > 0.4:
                details.append(f"Excess cash reserves: Cash/Revenue ratio of {ratio:.2f}")
            else:
                details.append(f"Low cash reserves: Cash/Revenue ratio of {ratio:.2f}")
            if revenue_values[0] and revenue_values[0] > 0:
                cash_to_revenue = cash_values[0] / revenue_values[0]
        else:
            details.append("Insufficient cash or revenue data")

        # 4. Insider activity - skin in the game
        insider_buy_ratio = None
        if insider_trades:
            sides = [_insider_side(t) for t in insider_trades]
            buys = sum(1 for s in sides if s == "buy")
            sells = sum(1 for s in sides if s == "sell")
            total = buys + sells
            if total > 0:
                insider_buy_ratio = buys / total
                if insider_buy_ratio > 0.7:
                    score += 2
                    details.append(f"Strong insider buying: {buys}/{total} transactions are purchases")
                elif insider_buy_ratio > 0.4:
                    score += 1
                    details.append(f"Balanced insider trading: {buys}/{total} transactions are purchases")
                elif insider_buy_ratio < 0.1 and sells > 5:
                    score -= 1  # penalty for excessive selling
                    details.append(f"Concerning insider selling: {sells}/{total} transactions are sales")
                else:
                    details.append(f"Mixed insider activity: {buys}/{total} transactions are purchases")
            else:
                details.append("No recorded insider transactions")
        else:
            details.append("No insider trading data available")

        # 5. Share count consistency
        share_count_trend = "unknown"
        share_counts = [i.outstanding_shares for i in items if i.outstanding_shares is not None]
        if share_counts and len(share_counts) >= 3:
            if share_counts[0] < share_counts[-1] * 0.95:
                score += 2
                details.append("Shareholder-friendly: Reducing share count over time")
            elif share_counts[0] < share_counts[-1] * 1.05:
                score += 1
                details.append("Stable share count: Limited dilution")
            elif share_counts[0] > share_counts[-1] * 1.2:
                score -= 1  # penalty for excessive dilution
                details.append("Concerning dilution: Share count increased significantly")
            else:
                details.append("Moderate share count increase over time")
            if share_counts[0] < share_counts[-1] * 0.95:
                share_count_trend = "decreasing"
            elif share_counts[0] > share_counts[-1] * 1.05:
                share_count_trend = "increasing"
            else:
                share_count_trend = "stable"
        else:
            details.append("Insufficient share count data")

        final_score = max(0, min(10, score * 10 / 12))  # max raw score is 12 (3+3+2+2+2)
        return {
            "score": final_score,
            "details": "; ".join(details),
            "insider_buy_ratio": insider_buy_ratio,
            "recent_de_ratio": recent_de_ratio,
            "cash_to_revenue": cash_to_revenue,
            "share_count_trend": share_count_trend,
        }

    @staticmethod
    def analyze_predictability(items: list) -> dict[str, Any]:
        score = 0
        details = []
        if not items or len(items) < 5:
            return {"score": 0, "details": "Insufficient data to analyze business predictability (need 5+ years)"}

        # 1. Revenue stability and growth
        revenues = [i.revenue for i in items if i.revenue is not None]
        if revenues and len(revenues) >= 5:
            growth_rates = [revenues[i] / revenues[i + 1] - 1 for i in range(len(revenues) - 1) if revenues[i + 1] != 0]
            if not growth_rates:
                details.append("Cannot calculate revenue growth: zero revenue values found")
            else:
                avg_growth = sum(growth_rates) / len(growth_rates)
                volatility = sum(abs(r - avg_growth) for r in growth_rates) / len(growth_rates)
                if avg_growth > 0.05 and volatility < 0.1:
                    score += 3
                    details.append(f"Highly predictable revenue: {avg_growth:.1%} avg growth with low volatility")
                elif avg_growth > 0 and volatility < 0.2:
                    score += 2
                    details.append(f"Moderately predictable revenue: {avg_growth:.1%} avg growth with some volatility")
                elif avg_growth > 0:
                    score += 1
                    details.append(f"Growing but less predictable revenue: {avg_growth:.1%} avg growth with high volatility")
                else:
                    details.append(f"Declining or highly unpredictable revenue: {avg_growth:.1%} avg growth")
        else:
            details.append("Insufficient revenue history for predictability analysis")

        # 2. Operating income stability
        op_income = [i.operating_income for i in items if i.operating_income is not None]
        if op_income and len(op_income) >= 5:
            positive = sum(1 for x in op_income if x > 0)
            if positive == len(op_income):
                score += 3
                details.append("Highly predictable operations: Operating income positive in all periods")
            elif positive >= len(op_income) * 0.8:
                score += 2
                details.append(f"Predictable operations: Operating income positive in {positive}/{len(op_income)} periods")
            elif positive >= len(op_income) * 0.6:
                score += 1
                details.append(f"Somewhat predictable operations: Operating income positive in {positive}/{len(op_income)} periods")
            else:
                details.append(f"Unpredictable operations: Operating income positive in only {positive}/{len(op_income)} periods")
        else:
            details.append("Insufficient operating income history")

        # 3. Margin consistency
        op_margins = [i.operating_margin for i in items if i.operating_margin is not None]
        if op_margins and len(op_margins) >= 5:
            avg_margin = sum(op_margins) / len(op_margins)
            volatility = sum(abs(m - avg_margin) for m in op_margins) / len(op_margins)
            if volatility < 0.03:
                score += 2
                details.append(f"Highly predictable margins: {avg_margin:.1%} avg with minimal volatility")
            elif volatility < 0.07:
                score += 1
                details.append(f"Moderately predictable margins: {avg_margin:.1%} avg with some volatility")
            else:
                details.append(f"Unpredictable margins: {avg_margin:.1%} avg with high volatility ({volatility:.1%})")
        else:
            details.append("Insufficient margin history")

        # 4. Cash generation reliability
        fcf_values = [i.free_cash_flow for i in items if i.free_cash_flow is not None]
        if fcf_values and len(fcf_values) >= 5:
            positive = sum(1 for f in fcf_values if f > 0)
            if positive == len(fcf_values):
                score += 2
                details.append("Highly predictable cash generation: Positive FCF in all periods")
            elif positive >= len(fcf_values) * 0.8:
                score += 1
                details.append(f"Predictable cash generation: Positive FCF in {positive}/{len(fcf_values)} periods")
            else:
                details.append(f"Unpredictable cash generation: Positive FCF in only {positive}/{len(fcf_values)} periods")
        else:
            details.append("Insufficient free cash flow history")

        final_score = min(10, score * 10 / 10)  # max raw score is 10 (3+3+2+2)
        return {"score": final_score, "details": "; ".join(details)}

    @staticmethod
    def calculate_munger_valuation(items: list, market_cap: float | None) -> dict[str, Any]:
        score = 0
        details = []
        if not items or market_cap is None:
            return {"score": 0, "details": "Insufficient data to perform valuation"}

        fcf_values = [i.free_cash_flow for i in items if i.free_cash_flow is not None]
        if not fcf_values or len(fcf_values) < 3:
            return {"score": 0, "details": "Insufficient free cash flow data for valuation"}

        # 1. Normalize earnings over the last 3-5 years
        n = min(5, len(fcf_values))
        normalized_fcf = sum(fcf_values[:n]) / n
        if normalized_fcf <= 0:
            return {"score": 0, "details": f"Negative or zero normalized FCF ({normalized_fcf}), cannot value", "intrinsic_value": None}

        # 2. FCF yield
        if market_cap <= 0:
            return {"score": 0, "details": f"Invalid market cap ({market_cap}), cannot value"}
        fcf_yield = normalized_fcf / market_cap

        # 3. Sliding scale on yield
        if fcf_yield > 0.08:
            score += 4
            details.append(f"Excellent value: {fcf_yield:.1%} FCF yield")
        elif fcf_yield > 0.05:
            score += 3
            details.append(f"Good value: {fcf_yield:.1%} FCF yield")
        elif fcf_yield > 0.03:
            score += 1
            details.append(f"Fair value: {fcf_yield:.1%} FCF yield")
        else:
            details.append(f"Expensive: Only {fcf_yield:.1%} FCF yield")

        # 4. Simple intrinsic value range
        conservative_value = normalized_fcf * 10
        reasonable_value = normalized_fcf * 15
        optimistic_value = normalized_fcf * 20

        # 5. Margin of safety vs reasonable value
        mos = (reasonable_value - market_cap) / market_cap
        if mos > 0.3:
            score += 3
            details.append(f"Large margin of safety: {mos:.1%} upside to reasonable value")
        elif mos > 0.1:
            score += 2
            details.append(f"Moderate margin of safety: {mos:.1%} upside to reasonable value")
        elif mos > -0.1:
            score += 1
            details.append(f"Fair price: Within 10% of reasonable value ({mos:.1%})")
        else:
            details.append(f"Expensive: {-mos:.1%} premium to reasonable value")

        # 6. FCF trajectory
        if len(fcf_values) >= 3:
            recent_avg = sum(fcf_values[:3]) / 3
            older_avg = sum(fcf_values[-3:]) / 3 if len(fcf_values) >= 6 else fcf_values[-1]
            if recent_avg > older_avg * 1.2:
                score += 3
                details.append("Growing FCF trend adds to intrinsic value")
            elif recent_avg > older_avg:
                score += 2
                details.append("Stable to growing FCF supports valuation")
            else:
                details.append("Declining FCF trend is concerning")

        final_score = min(10, score * 10 / 10)  # max raw score is 10 (4+3+3)
        return {
            "score": final_score,
            "details": "; ".join(details),
            "intrinsic_value_range": {"conservative": conservative_value, "reasonable": reasonable_value, "optimistic": optimistic_value},
            "fcf_yield": fcf_yield,
            "normalized_fcf": normalized_fcf,
            "margin_of_safety_vs_fair_value": mos,
        }

    @staticmethod
    def analyze_news_sentiment(news_items: list) -> str:
        if not news_items:
            return "No news data available"
        return f"Qualitative review of {len(news_items)} recent news items would be needed"

    @staticmethod
    def compute_confidence(moat: float, mgmt: float, pred: float, val: float, mos: float | None, signal: str) -> int:
        """Upstream ``compute_confidence``: quality-weighted, nudged by margin of safety, bucketed by signal."""
        moat, mgmt, pred, val = float(moat or 0), float(mgmt or 0), float(pred or 0), float(val or 0)
        quality = 0.35 * moat + 0.25 * mgmt + 0.25 * pred  # 0..8.5
        quality_pct = 100 * (quality / 8.5) if quality > 0 else 0
        mos = float(mos) if mos is not None else 0.0
        val_adj = max(-10.0, min(10.0, mos * 100.0 / 3.0))  # ~+/-10pp at +/-30% MOS
        base = 0.85 * quality_pct + 0.15 * (val * 10) + val_adj
        if signal == "bullish":
            upper = 100 if mos > 0 else 69
            lower = 50 if quality_pct >= 55 else 30
        elif signal == "bearish":
            lower = 10 if mos < -0.05 else 30
            upper = 49
        else:
            lower, upper = 50, 69
        conf = int(round(max(lower, min(upper, base))))
        return max(10, min(100, conf))
