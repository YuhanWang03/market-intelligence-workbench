"""Bill Ackman — quality brands, financial discipline, activism upside.

Ported from ``src/agents/bill_ackman.py`` at virattt/ai-hedge-fund
v2026.5.14 (MIT).  The four scoring functions (business quality, financial
discipline, activism potential, DCF valuation) are kept rule-for-rule; the
LangGraph state, progress bar and LLM call are gone.  Upstream compared the
total against a hard-coded ``max_possible_score = 20`` (bullish at >= 14,
bearish at <= 6) even though its rules can only award 7 + 4 + 2 + 3 = 16
points; the parts below carry their real maxima and
:meth:`BillAckman.decide` keeps the upstream denominator so the verdict
thresholds are unchanged.  The prompt's "valuation matters: target a margin
of safety" is applied through the shared margin-of-safety gate.
"""

from __future__ import annotations

from typing import Any

from v2.personas.base import Persona, apply_margin_of_safety, classic_verdict, confidence_from_ratio
from v2.personas.models import Evaluation, Signal
from v2.personas.snapshot import PersonaSnapshot


class BillAckman(Persona):
    key = "bill_ackman"
    name = "Bill Ackman"
    name_zh = "比尔·阿克曼"
    style = "takes bold positions and pushes for change"
    period = "annual"
    lookback = 5
    needs = frozenset()
    #: upstream's hard-coded denominator for the 70% / 30% signal thresholds
    upstream_max_score = 20
    system_prompt = (
        "You are a Bill Ackman AI agent. Principles: 1. high-quality businesses with durable moats, often "
        "well-known consumer or service brands; 2. consistent free cash flow and long-term growth; 3. financial "
        "discipline (reasonable leverage, efficient capital allocation); 4. valuation matters — intrinsic value "
        "with a margin of safety; 5. activism where management or operational fixes unlock upside; 6. a few "
        "high-conviction positions.\n"
        "Reasoning: emphasize brand, moat or positioning; review FCF and margin trends; analyze leverage, "
        "buybacks and dividends; give a valuation with numbers (DCF, multiples); name activism catalysts. "
        "Confident, analytic, sometimes confrontational tone.\n"
        "Signal rules: bullish = score >= 70% of max; bearish = score <= 30%; neutral otherwise. Confidence 0-100."
    )

    # ------------------------------------------------------------------ rules

    def evaluate(self, snap: PersonaSnapshot) -> Evaluation:
        metrics = snap.metrics(self.period, self.lookback)
        items = snap.line_items(self.period, self.lookback)
        market_cap = snap.market_cap

        quality = self.analyze_business_quality(metrics, items)
        discipline = self.analyze_financial_discipline(metrics, items)
        activism = self.analyze_activism_potential(items)
        valuation = self.analyze_valuation(items, market_cap)

        ev = Evaluation(parts=[
            self.part("business_quality", quality["score"], 7, quality["details"]),
            self.part("financial_discipline", discipline["score"], 4, discipline["details"]),
            self.part("activism_potential", activism["score"], 2, activism["details"]),
            self.part("valuation", valuation["score"], 3, valuation["details"]),
        ])

        ev.margin_of_safety = valuation.get("margin_of_safety")
        latest = items[0] if items else None
        ev.facts = {
            "intrinsic_value": valuation.get("intrinsic_value"),
            "market_cap": market_cap,
            "margin_of_safety": ev.margin_of_safety,
            "free_cash_flow": latest.free_cash_flow if latest else None,
            "return_on_equity": metrics[0].return_on_equity if metrics else None,
            "dcf_assumptions": valuation.get("assumptions"),
            "valuation_details": valuation.get("details"),
        }
        if not metrics and not items:
            ev.insufficient = True
            ev.notes.append("no fundamentals")
        return ev

    def decide(self, ev: Evaluation) -> tuple[Signal, int]:
        # Upstream: bullish if total >= 0.7 * 20, bearish if total <= 0.3 * 20.
        ratio = ev.score / self.upstream_max_score
        signal = classic_verdict(ratio, bullish_at=0.7, bearish_at=0.3)
        confidence = confidence_from_ratio(ratio, signal, bullish_at=0.7, bearish_at=0.3)
        return apply_margin_of_safety(signal, confidence, ev.margin_of_safety)

    # ---------------------------------------------------- upstream functions

    @staticmethod
    def analyze_business_quality(metrics: list, financial_line_items: list) -> dict[str, Any]:
        """Stable or growing cash flows, durable moats, long-term growth potential."""
        score = 0
        details = []

        if not metrics or not financial_line_items:
            return {"score": 0, "details": "Insufficient data to analyze business quality"}

        revenues = [item.revenue for item in financial_line_items if item.revenue is not None]
        if len(revenues) >= 2:
            initial, final = revenues[-1], revenues[0]
            if initial and final and final > initial:
                growth_rate = (final - initial) / abs(initial)
                if growth_rate > 0.5:
                    score += 2
                    details.append(f"Revenue grew by {(growth_rate * 100):.1f}% over the full period (strong growth).")
                else:
                    score += 1
                    details.append(f"Revenue growth is positive but under 50% cumulatively ({(growth_rate * 100):.1f}%).")
            else:
                details.append("Revenue did not grow significantly or data insufficient.")
        else:
            details.append("Not enough revenue data for multi-period trend.")

        fcf_vals = [item.free_cash_flow for item in financial_line_items if item.free_cash_flow is not None]
        op_margin_vals = [item.operating_margin for item in financial_line_items if item.operating_margin is not None]

        if op_margin_vals:
            above_15 = sum(1 for m in op_margin_vals if m > 0.15)
            if above_15 >= (len(op_margin_vals) // 2 + 1):
                score += 2
                details.append("Operating margins have often exceeded 15% (indicates good profitability).")
            else:
                details.append("Operating margin not consistently above 15%.")
        else:
            details.append("No operating margin data across periods.")

        if fcf_vals:
            positive_fcf_count = sum(1 for f in fcf_vals if f > 0)
            if positive_fcf_count >= (len(fcf_vals) // 2 + 1):
                score += 1
                details.append("Majority of periods show positive free cash flow.")
            else:
                details.append("Free cash flow not consistently positive.")
        else:
            details.append("No free cash flow data across periods.")

        latest_metrics = metrics[0]
        if latest_metrics.return_on_equity and latest_metrics.return_on_equity > 0.15:
            score += 2
            details.append(f"High ROE of {latest_metrics.return_on_equity:.1%}, indicating a competitive advantage.")
        elif latest_metrics.return_on_equity:
            details.append(f"ROE of {latest_metrics.return_on_equity:.1%} is moderate.")
        else:
            details.append("ROE data not available.")

        return {"score": score, "details": "; ".join(details)}

    @staticmethod
    def analyze_financial_discipline(metrics: list, financial_line_items: list) -> dict[str, Any]:
        """Debt ratio trends and capital returned to shareholders over multiple periods."""
        score = 0
        details = []

        if not metrics or not financial_line_items:
            return {"score": 0, "details": "Insufficient data to analyze financial discipline"}

        debt_to_equity_vals = [item.debt_to_equity for item in financial_line_items if item.debt_to_equity is not None]
        if debt_to_equity_vals:
            below_one_count = sum(1 for d in debt_to_equity_vals if d < 1.0)
            if below_one_count >= (len(debt_to_equity_vals) // 2 + 1):
                score += 2
                details.append("Debt-to-equity < 1.0 for the majority of periods (reasonable leverage).")
            else:
                details.append("Debt-to-equity >= 1.0 in many periods (could be high leverage).")
        else:
            liab_to_assets = []
            for item in financial_line_items:
                if item.total_liabilities and item.total_assets and item.total_assets > 0:
                    liab_to_assets.append(item.total_liabilities / item.total_assets)
            if liab_to_assets:
                below_50pct_count = sum(1 for ratio in liab_to_assets if ratio < 0.5)
                if below_50pct_count >= (len(liab_to_assets) // 2 + 1):
                    score += 2
                    details.append("Liabilities-to-assets < 50% for majority of periods.")
                else:
                    details.append("Liabilities-to-assets >= 50% in many periods.")
            else:
                details.append("No consistent leverage ratio data available.")

        dividends_list = [
            item.dividends_and_other_cash_distributions
            for item in financial_line_items
            if item.dividends_and_other_cash_distributions is not None
        ]
        if dividends_list:
            paying_dividends_count = sum(1 for d in dividends_list if d < 0)
            if paying_dividends_count >= (len(dividends_list) // 2 + 1):
                score += 1
                details.append("Company has a history of returning capital to shareholders (dividends).")
            else:
                details.append("Dividends not consistently paid or no data on distributions.")
        else:
            details.append("No dividend data found across periods.")

        shares = [item.outstanding_shares for item in financial_line_items if item.outstanding_shares is not None]
        if len(shares) >= 2:
            if shares[0] < shares[-1]:
                score += 1
                details.append("Outstanding shares have decreased over time (possible buybacks).")
            else:
                details.append("Outstanding shares have not decreased over the available periods.")
        else:
            details.append("No multi-period share count data to assess buybacks.")

        return {"score": score, "details": "; ".join(details)}

    @staticmethod
    def analyze_activism_potential(financial_line_items: list) -> dict[str, Any]:
        """Positive revenue trend but subpar margins may mean operational upside an activist can unlock."""
        if not financial_line_items:
            return {"score": 0, "details": "Insufficient data for activism potential"}

        revenues = [item.revenue for item in financial_line_items if item.revenue is not None]
        op_margins = [item.operating_margin for item in financial_line_items if item.operating_margin is not None]

        if len(revenues) < 2 or not op_margins:
            return {"score": 0, "details": "Not enough data to assess activism potential (need multi-year revenue + margins)."}

        initial, final = revenues[-1], revenues[0]
        revenue_growth = (final - initial) / abs(initial) if initial else 0
        avg_margin = sum(op_margins) / len(op_margins)

        score = 0
        details = []
        if revenue_growth > 0.15 and avg_margin < 0.10:
            score += 2
            details.append(
                f"Revenue growth is healthy (~{revenue_growth * 100:.1f}%), but margins are low (avg {avg_margin * 100:.1f}%). "
                "Activism could unlock margin improvements."
            )
        else:
            details.append("No clear sign of activism opportunity (either margins are already decent or growth is weak).")

        return {"score": score, "details": "; ".join(details)}

    @staticmethod
    def analyze_valuation(financial_line_items: list, market_cap: float | None) -> dict[str, Any]:
        """Simplified DCF on the latest FCF, plus margin of safety against market cap."""
        if not financial_line_items or market_cap is None:
            return {"score": 0, "details": "Insufficient data to perform valuation"}

        latest = financial_line_items[0]
        fcf = latest.free_cash_flow if latest.free_cash_flow else 0

        if fcf <= 0:
            return {"score": 0, "details": f"No positive FCF for valuation; FCF = {fcf}", "intrinsic_value": None}

        growth_rate = 0.06
        discount_rate = 0.10
        terminal_multiple = 15
        projection_years = 5

        present_value = 0
        for year in range(1, projection_years + 1):
            future_fcf = fcf * (1 + growth_rate) ** year
            pv = future_fcf / ((1 + discount_rate) ** year)
            present_value += pv

        terminal_value = (fcf * (1 + growth_rate) ** projection_years * terminal_multiple) / ((1 + discount_rate) ** projection_years)
        intrinsic_value = present_value + terminal_value
        margin_of_safety = (intrinsic_value - market_cap) / market_cap

        score = 0
        if margin_of_safety > 0.3:
            score += 3
        elif margin_of_safety > 0.1:
            score += 1

        details = [
            f"Calculated intrinsic value: ~{intrinsic_value:,.2f}",
            f"Market cap: ~{market_cap:,.2f}",
            f"Margin of safety: {margin_of_safety:.2%}",
        ]
        return {
            "score": score,
            "details": "; ".join(details),
            "intrinsic_value": intrinsic_value,
            "margin_of_safety": margin_of_safety,
            "assumptions": {
                "growth_rate": growth_rate,
                "discount_rate": discount_rate,
                "terminal_multiple": terminal_multiple,
                "projection_years": projection_years,
            },
        }
