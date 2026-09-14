"""Mohnish Pabrai — heads I win, tails I don't lose much.

Ported from ``src/agents/mohnish_pabrai.py`` at virattt/ai-hedge-fund
v2026.5.14 (MIT).  The three scoring functions (downside protection,
FCF-yield valuation, doubling potential) are kept rule-for-rule, including
their 0-10 capping and the 45/35/20 weighting; the LangGraph state,
progress bar and LLM call are gone.  Upstream drew its pre-signal from the
weighted total (>= 7.5 bullish, <= 4.0 bearish) and left the confidence to
the LLM; :meth:`MohnishPabrai.decide` keeps those cut-offs and derives the
confidence from the distance to them.
"""

from __future__ import annotations

from typing import Any

from v2.personas.base import Persona, classic_verdict, confidence_from_ratio
from v2.personas.models import Evaluation, Signal
from v2.personas.snapshot import PersonaSnapshot

_WEIGHTS = {"downside_protection": 0.45, "valuation": 0.35, "double_potential": 0.20}
_BULLISH_AT = 0.75  # upstream: total_score >= 7.5 of 10
_BEARISH_AT = 0.40  # upstream: total_score <= 4.0 of 10


class MohnishPabrai(Persona):
    key = "mohnish_pabrai"
    name = "Mohnish Pabrai"
    name_zh = "莫尼什·帕伯莱"
    style = "looks for doubles at low risk"
    period = "annual"
    lookback = 8
    needs = frozenset()
    system_prompt = (
        "You are Mohnish Pabrai. Decide bullish, bearish, or neutral using only the provided facts, with "
        "candid, checklist-driven reasoning.\n"
        "Philosophy: heads I win, tails I don't lose much - downside protection first (net cash, current "
        "ratio, low D/E, positive stable FCF); simple, understandable businesses with durable moats; demand "
        "a high normalized FCF yield and prefer asset-light models (low capex/revenue); look for rising "
        "intrinsic value at a much lower price; clone proven checklists over novelty; seek a low-risk path "
        "to double in 2-3 years (revenue and FCF growth, or a FCF yield above 8%); avoid leverage, "
        "complexity and fragile balance sheets.\n"
        "Score = 0.45 downside + 0.35 valuation + 0.20 doubling, out of 10. Signal rules: bullish >= 7.5; "
        "bearish <= 4.0; neutral otherwise. Confidence 0-100."
    )

    # ------------------------------------------------------------------ rules

    def evaluate(self, snap: PersonaSnapshot) -> Evaluation:
        metrics = snap.metrics(self.period, self.lookback)
        items = snap.line_items(self.period, self.lookback)
        market_cap = snap.market_cap

        downside = self.analyze_downside_protection(items)
        valuation = self.analyze_pabrai_valuation(items, market_cap)
        double = self.analyze_double_potential(items, market_cap)

        parts = []
        for name, result in (("downside_protection", downside), ("valuation", valuation), ("double_potential", double)):
            weight = _WEIGHTS[name]
            parts.append(self.part(name, result["score"] * weight, 10 * weight, result["details"], raw_score=result["score"], raw_max=10, weight=weight))
        ev = Evaluation(parts=parts)
        ev.facts = {
            "downside_score": downside["score"],
            "valuation_score": valuation["score"],
            "double_potential_score": double["score"],
            "market_cap": market_cap,
            "fcf_yield": valuation.get("fcf_yield"),
            "normalized_fcf": valuation.get("normalized_fcf"),
            "net_cash": downside.get("net_cash"),
        }
        if not metrics and not items:
            ev.insufficient = True
            ev.notes.append("no fundamentals")
        return ev

    def decide(self, ev: Evaluation) -> tuple[Signal, int]:
        # Upstream cut the weighted 0-10 total at 7.5 / 4.0; ratio is total / 10.
        signal = classic_verdict(ev.ratio, bullish_at=_BULLISH_AT, bearish_at=_BEARISH_AT)
        confidence = confidence_from_ratio(ev.ratio, signal, bullish_at=_BULLISH_AT, bearish_at=_BEARISH_AT)
        return signal, confidence

    # ---------------------------------------------------- upstream functions

    @staticmethod
    def analyze_downside_protection(items: list) -> dict[str, Any]:
        if not items:
            return {"score": 0, "details": "Insufficient data"}
        latest = items[0]
        details: list[str] = []
        score = 0

        cash = latest.cash_and_equivalents
        debt = latest.total_debt
        current_assets = latest.current_assets
        current_liabilities = latest.current_liabilities
        equity = latest.shareholders_equity

        # Net cash is a strong downside protector
        net_cash = None
        if cash is not None and debt is not None:
            net_cash = cash - debt
            if net_cash > 0:
                score += 3
                details.append(f"Net cash position: ${net_cash:,.0f}")
            else:
                details.append(f"Net debt position: ${net_cash:,.0f}")

        # Current ratio
        if current_assets is not None and current_liabilities is not None and current_liabilities > 0:
            current_ratio = current_assets / current_liabilities
            if current_ratio >= 2.0:
                score += 2
                details.append(f"Strong liquidity (current ratio {current_ratio:.2f})")
            elif current_ratio >= 1.2:
                score += 1
                details.append(f"Adequate liquidity (current ratio {current_ratio:.2f})")
            else:
                details.append(f"Weak liquidity (current ratio {current_ratio:.2f})")

        # Low leverage
        if equity is not None and equity > 0 and debt is not None:
            de_ratio = debt / equity
            if de_ratio < 0.3:
                score += 2
                details.append(f"Very low leverage (D/E {de_ratio:.2f})")
            elif de_ratio < 0.7:
                score += 1
                details.append(f"Moderate leverage (D/E {de_ratio:.2f})")
            else:
                details.append(f"High leverage (D/E {de_ratio:.2f})")

        # FCF positive and stable
        fcf_values = [i.free_cash_flow for i in items if i.free_cash_flow is not None]
        if fcf_values and len(fcf_values) >= 3:
            recent_avg = sum(fcf_values[:3]) / 3
            older = sum(fcf_values[-3:]) / 3 if len(fcf_values) >= 6 else fcf_values[-1]
            if recent_avg > 0 and recent_avg >= older:
                score += 2
                details.append("Positive and improving/stable FCF")
            elif recent_avg > 0:
                score += 1
                details.append("Positive but declining FCF")
            else:
                details.append("Negative FCF")

        return {"score": min(10, score), "details": "; ".join(details), "net_cash": net_cash}

    @staticmethod
    def analyze_pabrai_valuation(items: list, market_cap: float | None) -> dict[str, Any]:
        if not items or market_cap is None or market_cap <= 0:
            return {"score": 0, "details": "Insufficient data", "fcf_yield": None, "normalized_fcf": None}

        details: list[str] = []
        fcf_values = [i.free_cash_flow for i in items if i.free_cash_flow is not None]
        capex_vals = [abs(i.capital_expenditure or 0) for i in items]

        if not fcf_values or len(fcf_values) < 3:
            return {"score": 0, "details": "Insufficient FCF history", "fcf_yield": None, "normalized_fcf": None}

        n = min(5, len(fcf_values))
        normalized_fcf = sum(fcf_values[:n]) / n
        if normalized_fcf <= 0:
            return {"score": 0, "details": "Non-positive normalized FCF", "fcf_yield": None, "normalized_fcf": normalized_fcf}

        fcf_yield = normalized_fcf / market_cap
        score = 0
        if fcf_yield > 0.10:
            score += 4
            details.append(f"Exceptional value: {fcf_yield:.1%} FCF yield")
        elif fcf_yield > 0.07:
            score += 3
            details.append(f"Attractive value: {fcf_yield:.1%} FCF yield")
        elif fcf_yield > 0.05:
            score += 2
            details.append(f"Reasonable value: {fcf_yield:.1%} FCF yield")
        elif fcf_yield > 0.03:
            score += 1
            details.append(f"Borderline value: {fcf_yield:.1%} FCF yield")
        else:
            details.append(f"Expensive: {fcf_yield:.1%} FCF yield")

        # Asset-light tilt: lower capex intensity preferred
        if capex_vals and len(items) >= 3:
            capex_to_revenue = []
            for i in items:
                revenue = i.revenue
                capex = abs(i.capital_expenditure or 0)
                if revenue and revenue > 0:
                    capex_to_revenue.append(capex / revenue)
            if capex_to_revenue:
                avg_ratio = sum(capex_to_revenue) / len(capex_to_revenue)
                if avg_ratio < 0.05:
                    score += 2
                    details.append(f"Asset-light: Avg capex {avg_ratio:.1%} of revenue")
                elif avg_ratio < 0.10:
                    score += 1
                    details.append(f"Moderate capex: Avg capex {avg_ratio:.1%} of revenue")
                else:
                    details.append(f"Capex heavy: Avg capex {avg_ratio:.1%} of revenue")

        return {"score": min(10, score), "details": "; ".join(details), "fcf_yield": fcf_yield, "normalized_fcf": normalized_fcf}

    @classmethod
    def analyze_double_potential(cls, items: list, market_cap: float | None) -> dict[str, Any]:
        if not items or market_cap is None or market_cap <= 0:
            return {"score": 0, "details": "Insufficient data"}

        details: list[str] = []
        revenues = [i.revenue for i in items if i.revenue is not None]
        fcfs = [i.free_cash_flow for i in items if i.free_cash_flow is not None]

        score = 0
        if revenues and len(revenues) >= 3:
            recent_rev = sum(revenues[:3]) / 3
            older_rev = sum(revenues[-3:]) / 3 if len(revenues) >= 6 else revenues[-1]
            if older_rev > 0:
                rev_growth = (recent_rev / older_rev) - 1
                if rev_growth > 0.15:
                    score += 2
                    details.append(f"Strong revenue trajectory ({rev_growth:.1%})")
                elif rev_growth > 0.05:
                    score += 1
                    details.append(f"Modest revenue growth ({rev_growth:.1%})")

        if fcfs and len(fcfs) >= 3:
            recent_fcf = sum(fcfs[:3]) / 3
            older_fcf = sum(fcfs[-3:]) / 3 if len(fcfs) >= 6 else fcfs[-1]
            if older_fcf != 0:
                fcf_growth = (recent_fcf / older_fcf) - 1
                if fcf_growth > 0.20:
                    score += 3
                    details.append(f"Strong FCF growth ({fcf_growth:.1%})")
                elif fcf_growth > 0.08:
                    score += 2
                    details.append(f"Healthy FCF growth ({fcf_growth:.1%})")
                elif fcf_growth > 0:
                    score += 1
                    details.append(f"Positive FCF growth ({fcf_growth:.1%})")

        # A high FCF yield (>8%) can double capital from cash generation alone
        fcf_yield = cls.analyze_pabrai_valuation(items, market_cap).get("fcf_yield")
        if fcf_yield is not None:
            if fcf_yield > 0.08:
                score += 3
                details.append("High FCF yield can drive doubling via retained cash/Buybacks")
            elif fcf_yield > 0.05:
                score += 1
                details.append("Reasonable FCF yield supports moderate compounding")

        return {"score": min(10, score), "details": "; ".join(details)}
