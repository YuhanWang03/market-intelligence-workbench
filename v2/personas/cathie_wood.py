"""Cathie Wood — the power of innovation and disruption.

Ported from ``src/agents/cathie_wood.py`` at virattt/ai-hedge-fund v2026.5.14
(MIT).  The three scoring functions (disruptive potential, innovation-driven
growth, high-growth valuation) are kept rule-for-rule; the LangGraph state,
progress bar and LLM call are gone.  Upstream normalised the first two parts
to 5 points each, declared ``max_possible_score = 15`` and decided the signal
in code at 70% / 30% of it — so the valuation part is carried at max 5 here
(its rules can award at most 3, as upstream) to keep those cuts exact; that
is :meth:`CathieWood.decide`.  The margin of safety is reported as a fact
but, as upstream, only feeds the valuation score — no value-style gate.
"""

from __future__ import annotations

from typing import Any

from v2.personas.base import Persona, classic_verdict, confidence_from_ratio
from v2.personas.models import Evaluation, Signal
from v2.personas.snapshot import PersonaSnapshot


class CathieWood(Persona):
    key = "cathie_wood"
    name = "Cathie Wood"
    name_zh = "凯茜·伍德"
    style = "believes in the power of innovation and disruption"
    period = "annual"
    lookback = 5
    needs = frozenset()
    system_prompt = (
        "You are Cathie Wood. Explain a bullish, bearish, or neutral verdict using only the provided facts, in her "
        "optimistic, future-focused, conviction-driven voice.\n"
        "Principles: seek disruptive innovation; exponential growth potential and large TAM; technology, healthcare "
        "and other future-facing sectors; multi-year horizons; accept volatility for high returns; management's "
        "vision and willingness to invest in R&D.\n"
        "Checklist: identify the disruptive technology; evidence of accelerating revenue and adoption; ability to "
        "scale in a large market; R&D intensity and pipeline; a growth-biased valuation (20% growth, 15% discount, "
        "25x terminal FCF multiple).\n"
        "Signal rules: score >= 70% of max = bullish; <= 30% = bearish; otherwise neutral. Be specific: cite growth "
        "rates, R&D as % of revenue, margin trends and the margin of safety."
    )

    # ------------------------------------------------------------------ rules

    def evaluate(self, snap: PersonaSnapshot) -> Evaluation:
        metrics = snap.metrics(self.period, self.lookback)
        items = snap.line_items(self.period, self.lookback)
        market_cap = snap.market_cap

        disruptive = self.analyze_disruptive_potential(metrics, items)
        innovation = self.analyze_innovation_growth(metrics, items)
        valuation = self.analyze_cathie_wood_valuation(items, market_cap)

        # Upstream: two parts normalised to 5 each, max_possible_score = 15.
        ev = Evaluation(parts=[
            self.part("disruptive_potential", disruptive["score"], 5, disruptive["details"],
                      raw_score=disruptive.get("raw_score"), raw_max=disruptive.get("max_score")),
            self.part("innovation_growth", innovation["score"], 5, innovation["details"],
                      raw_score=innovation.get("raw_score"), raw_max=innovation.get("max_score")),
            self.part("valuation", valuation["score"], 5, valuation["details"]),
        ])
        ev.margin_of_safety = valuation.get("margin_of_safety")
        ev.facts = {
            "intrinsic_value": valuation.get("intrinsic_value"),
            "market_cap": market_cap,
            "margin_of_safety": ev.margin_of_safety,
            "valuation_assumptions": {"growth_rate": 0.20, "discount_rate": 0.15, "terminal_multiple": 25, "projection_years": 5},
        }
        if not metrics and not items:
            ev.insufficient = True
            ev.notes.append("no fundamentals")
        return ev

    def decide(self, ev: Evaluation) -> tuple[Signal, int]:
        # Upstream: total >= 0.7 * 15 -> bullish, <= 0.3 * 15 -> bearish, else neutral.
        # The margin of safety already shaped the valuation score; no extra gate.
        signal = classic_verdict(ev.ratio)
        return signal, confidence_from_ratio(ev.ratio, signal)

    # ---------------------------------------------------- upstream functions

    @staticmethod
    def analyze_disruptive_potential(metrics: list, items: list) -> dict[str, Any]:
        score = 0
        details = []
        if not metrics or not items:
            return {"score": 0, "details": "Insufficient data to analyze disruptive potential"}

        # 1. Revenue growth — accelerating?
        revenues = [i.revenue for i in items if i.revenue]
        if len(revenues) >= 3:
            growth_rates = []
            for i in range(len(revenues) - 1):
                if revenues[i] and revenues[i + 1]:
                    growth_rate = (revenues[i] - revenues[i + 1]) / abs(revenues[i + 1]) if revenues[i + 1] != 0 else 0
                    growth_rates.append(growth_rate)
            if len(growth_rates) >= 2 and growth_rates[0] > growth_rates[-1]:
                score += 2
                details.append(f"Revenue growth is accelerating: {(growth_rates[0] * 100):.1f}% vs {(growth_rates[-1] * 100):.1f}%")
            latest_growth = growth_rates[0] if growth_rates else 0
            if latest_growth > 1.0:
                score += 3
                details.append(f"Exceptional revenue growth: {(latest_growth * 100):.1f}%")
            elif latest_growth > 0.5:
                score += 2
                details.append(f"Strong revenue growth: {(latest_growth * 100):.1f}%")
            elif latest_growth > 0.2:
                score += 1
                details.append(f"Moderate revenue growth: {(latest_growth * 100):.1f}%")
        else:
            details.append("Insufficient revenue data for growth analysis")

        # 2. Gross margin — expanding?
        gross_margins = [i.gross_margin for i in items if i.gross_margin is not None]
        if len(gross_margins) >= 2:
            margin_trend = gross_margins[0] - gross_margins[-1]
            if margin_trend > 0.05:
                score += 2
                details.append(f"Expanding gross margins: +{(margin_trend * 100):.1f}%")
            elif margin_trend > 0:
                score += 1
                details.append(f"Slightly improving gross margins: +{(margin_trend * 100):.1f}%")
            if gross_margins[0] > 0.50:
                score += 2
                details.append(f"High gross margin: {(gross_margins[0] * 100):.1f}%")
        else:
            details.append("Insufficient gross margin data")

        # 3. Operating leverage
        operating_expenses = [i.operating_expense for i in items if i.operating_expense]
        if len(revenues) >= 2 and len(operating_expenses) >= 2:
            rev_growth = (revenues[0] - revenues[-1]) / abs(revenues[-1])
            opex_growth = (operating_expenses[0] - operating_expenses[-1]) / abs(operating_expenses[-1])
            if rev_growth > opex_growth:
                score += 2
                details.append("Positive operating leverage: Revenue growing faster than expenses")
        else:
            details.append("Insufficient data for operating leverage analysis")

        # 4. R&D intensity
        rd_expenses = [i.research_and_development for i in items if i.research_and_development is not None]
        if rd_expenses and revenues:
            rd_intensity = rd_expenses[0] / revenues[0]
            if rd_intensity > 0.15:
                score += 3
                details.append(f"High R&D investment: {(rd_intensity * 100):.1f}% of revenue")
            elif rd_intensity > 0.08:
                score += 2
                details.append(f"Moderate R&D investment: {(rd_intensity * 100):.1f}% of revenue")
            elif rd_intensity > 0.05:
                score += 1
                details.append(f"Some R&D investment: {(rd_intensity * 100):.1f}% of revenue")
        else:
            details.append("No R&D data available")

        max_possible_score = 12
        normalized_score = (score / max_possible_score) * 5
        return {"score": normalized_score, "details": "; ".join(details), "raw_score": score, "max_score": max_possible_score}

    @staticmethod
    def analyze_innovation_growth(metrics: list, items: list) -> dict[str, Any]:
        score = 0
        details = []
        if not metrics or not items:
            return {"score": 0, "details": "Insufficient data to analyze innovation-driven growth"}

        # 1. R&D investment trends
        rd_expenses = [i.research_and_development for i in items if i.research_and_development]
        revenues = [i.revenue for i in items if i.revenue]
        if rd_expenses and revenues and len(rd_expenses) >= 2:
            rd_growth = (rd_expenses[0] - rd_expenses[-1]) / abs(rd_expenses[-1]) if rd_expenses[-1] != 0 else 0
            if rd_growth > 0.5:
                score += 3
                details.append(f"Strong R&D investment growth: +{(rd_growth * 100):.1f}%")
            elif rd_growth > 0.2:
                score += 2
                details.append(f"Moderate R&D investment growth: +{(rd_growth * 100):.1f}%")
            rd_intensity_start = rd_expenses[-1] / revenues[-1]
            rd_intensity_end = rd_expenses[0] / revenues[0]
            if rd_intensity_end > rd_intensity_start:
                score += 2
                details.append(f"Increasing R&D intensity: {(rd_intensity_end * 100):.1f}% vs {(rd_intensity_start * 100):.1f}%")
        else:
            details.append("Insufficient R&D data for trend analysis")

        # 2. Free cash flow
        fcf_vals = [i.free_cash_flow for i in items if i.free_cash_flow]
        if fcf_vals and len(fcf_vals) >= 2:
            fcf_growth = (fcf_vals[0] - fcf_vals[-1]) / abs(fcf_vals[-1])
            positive_fcf_count = sum(1 for f in fcf_vals if f > 0)
            if fcf_growth > 0.3 and positive_fcf_count == len(fcf_vals):
                score += 3
                details.append("Strong and consistent FCF growth, excellent innovation funding capacity")
            elif positive_fcf_count >= len(fcf_vals) * 0.75:
                score += 2
                details.append("Consistent positive FCF, good innovation funding capacity")
            elif positive_fcf_count > len(fcf_vals) * 0.5:
                score += 1
                details.append("Moderately consistent FCF, adequate innovation funding capacity")
        else:
            details.append("Insufficient FCF data for analysis")

        # 3. Operating efficiency
        op_margin_vals = [i.operating_margin for i in items if i.operating_margin]
        if op_margin_vals and len(op_margin_vals) >= 2:
            margin_trend = op_margin_vals[0] - op_margin_vals[-1]
            if op_margin_vals[0] > 0.15 and margin_trend > 0:
                score += 3
                details.append(f"Strong and improving operating margin: {(op_margin_vals[0] * 100):.1f}%")
            elif op_margin_vals[0] > 0.10:
                score += 2
                details.append(f"Healthy operating margin: {(op_margin_vals[0] * 100):.1f}%")
            elif margin_trend > 0:
                score += 1
                details.append("Improving operating efficiency")
        else:
            details.append("Insufficient operating margin data")

        # 4. Capital allocation
        capex = [i.capital_expenditure for i in items if i.capital_expenditure]
        if capex and revenues and len(capex) >= 2:
            capex_intensity = abs(capex[0]) / revenues[0]
            capex_growth = (abs(capex[0]) - abs(capex[-1])) / abs(capex[-1]) if capex[-1] != 0 else 0
            if capex_intensity > 0.10 and capex_growth > 0.2:
                score += 2
                details.append("Strong investment in growth infrastructure")
            elif capex_intensity > 0.05:
                score += 1
                details.append("Moderate investment in growth infrastructure")
        else:
            details.append("Insufficient CAPEX data")

        # 5. Growth reinvestment
        dividends = [i.dividends_and_other_cash_distributions for i in items if i.dividends_and_other_cash_distributions]
        if dividends and fcf_vals:
            latest_payout_ratio = dividends[0] / fcf_vals[0] if fcf_vals[0] != 0 else 1
            if latest_payout_ratio < 0.2:
                score += 2
                details.append("Strong focus on reinvestment over dividends")
            elif latest_payout_ratio < 0.4:
                score += 1
                details.append("Moderate focus on reinvestment over dividends")
        else:
            details.append("Insufficient dividend data")

        max_possible_score = 15
        normalized_score = (score / max_possible_score) * 5
        return {"score": normalized_score, "details": "; ".join(details), "raw_score": score, "max_score": max_possible_score}

    @staticmethod
    def analyze_cathie_wood_valuation(items: list, market_cap: float | None) -> dict[str, Any]:
        if not items or market_cap is None:
            return {"score": 0, "details": "Insufficient data for valuation"}
        latest = items[0]
        fcf = latest.free_cash_flow if latest.free_cash_flow else 0
        if fcf <= 0:
            return {"score": 0, "details": f"No positive FCF for valuation; FCF = {fcf}", "intrinsic_value": None}

        growth_rate = 0.20
        discount_rate = 0.15
        terminal_multiple = 25
        projection_years = 5

        present_value = 0.0
        for year in range(1, projection_years + 1):
            future_fcf = fcf * (1 + growth_rate) ** year
            present_value += future_fcf / ((1 + discount_rate) ** year)
        terminal_value = (fcf * (1 + growth_rate) ** projection_years * terminal_multiple) / ((1 + discount_rate) ** projection_years)
        intrinsic_value = present_value + terminal_value
        margin_of_safety = (intrinsic_value - market_cap) / market_cap

        score = 0
        if margin_of_safety > 0.5:
            score += 3
        elif margin_of_safety > 0.2:
            score += 1

        details = [f"Calculated intrinsic value: ~{intrinsic_value:,.2f}", f"Market cap: ~{market_cap:,.2f}", f"Margin of safety: {margin_of_safety:.2%}"]
        return {"score": score, "details": "; ".join(details), "intrinsic_value": intrinsic_value, "margin_of_safety": margin_of_safety}
