"""Rakesh Jhunjhunwala — the Big Bull of India.

Ported from ``src/agents/rakesh_jhunjhunwala.py`` at virattt/ai-hedge-fund
v2026.5.14 (MIT).  The five scoring functions, ``assess_quality_metrics``,
``calculate_intrinsic_value`` and the ``analyze_rakesh_jhunjhunwala_style``
aggregate are kept rule-for-rule; the LangGraph state, progress bar and LLM
call are gone.  Upstream already decided the signal *in code* — a 30% margin
of safety either way, else a quality-score tie-breaker — and the confidence
from ``abs(margin_of_safety) * 150`` (or the score ratio when no intrinsic
value could be computed); that logic is :meth:`RakeshJhunjhunwala.decide`.
Upstream also fetched five TTM metrics rows that no rule ever read, so the
port reads only the line items (TTM, ten periods — the API defaults it used).

One deliberate correction: upstream annualised every growth rate by
``len(values) - 1``, treating ten TTM rows as nine years. Ten TTM rows span
about two years, so EPS, revenue and income CAGRs came out roughly four
times too low and the DCF's growth input with them — the persona was bearish
on nearly everything. Growth now uses :meth:`Persona.cagr`, which measures
the span from the rows' report dates.  A second correction: the two
"consistency" loops compared neighbours as if rows were oldest-first, so a
company growing every period scored 0% consistent; they now read the
newest-first order the data actually has.
"""

from __future__ import annotations

from typing import Any

from v2.personas.base import Persona, clamp
from v2.personas.models import Evaluation, Signal
from v2.personas.snapshot import PersonaSnapshot


class RakeshJhunjhunwala(Persona):
    key = "rakesh_jhunjhunwala"
    name = "Rakesh Jhunjhunwala"
    name_zh = "拉克什·金君瓦拉"
    style = "the Big Bull of India"
    period = "ttm"
    lookback = 10
    needs = frozenset()
    system_prompt = (
        "You are Rakesh Jhunjhunwala. Explain a bullish, bearish, or neutral verdict using only the provided facts, "
        "in his conversational voice.\n"
        "Principles: circle of competence; margin of safety > 30% to intrinsic value; economic moat; conservative, "
        "shareholder-oriented management; low debt and strong ROE; long-term horizon — invest in businesses, not "
        "stocks; consistent earnings and revenue growth; sell only if fundamentals deteriorate or valuation far "
        "exceeds intrinsic value.\n"
        "Cite the decisive factors, quantify (ROE, margins, debt, CAGR), and say which principles are met or violated.\n"
        "Signal rules: bullish = margin_of_safety >= 30%, or quality >= 0.7 with score >= 60% at a fair price; "
        "bearish = margin_of_safety <= -30%, or quality <= 0.4, or score <= 30%; otherwise neutral. "
        "Confidence = |margin_of_safety| x 150 (20-95), or the score ratio (10-80) when no intrinsic value exists."
    )

    # ------------------------------------------------------------------ rules

    def evaluate(self, snap: PersonaSnapshot) -> Evaluation:
        items = snap.line_items(self.period, self.lookback)
        market_cap = snap.market_cap

        growth = self.analyze_growth(items)
        profitability = self.analyze_profitability(items)
        balance_sheet = self.analyze_balance_sheet(items)
        cash_flow = self.analyze_cash_flow(items)
        management = self.analyze_management_actions(items)
        intrinsic_value = self.calculate_intrinsic_value(items, market_cap)
        quality_score = self.assess_quality_metrics(items)
        style = self.analyze_rakesh_jhunjhunwala_style(items, intrinsic_value=intrinsic_value, current_price=market_cap)

        # 8 (profitability) + 7 (growth) + 4 (balance sheet) + 3 (cash flow) + 2 (management) = 24
        ev = Evaluation(parts=[
            self.part("profitability", profitability["score"], 8, profitability["details"]),
            self.part("growth", growth["score"], 7, growth["details"]),
            self.part("balance_sheet", balance_sheet["score"], 4, balance_sheet["details"]),
            self.part("cash_flow", cash_flow["score"], 3, cash_flow["details"]),
            self.part("management", management["score"], 2, management["details"]),
        ])

        if intrinsic_value and market_cap:
            ev.margin_of_safety = (intrinsic_value - market_cap) / market_cap
        ev.facts = {
            "intrinsic_value": intrinsic_value,
            "market_cap": market_cap,
            "margin_of_safety": ev.margin_of_safety,
            "quality_score": quality_score,
            "style_total_score": style["total_score"],
            "valuation_gap": style["valuation_gap"],
            "style_details": style["details"],
        }
        if not items:
            ev.insufficient = True
            ev.notes.append("no fundamentals")
        return ev

    def decide(self, ev: Evaluation) -> tuple[Signal, int]:
        mos = ev.margin_of_safety
        total_score, max_score = ev.score, ev.max_score
        quality_score = float(ev.facts.get("quality_score", 0.5))

        # Jhunjhunwala's decision rules (30% minimum margin of safety for conviction)
        if mos is not None and mos >= 0.30:
            signal: Signal = "bullish"
        elif mos is not None and mos <= -0.30:
            signal = "bearish"
        else:
            # Use quality score as tie-breaker for neutral cases
            if quality_score >= 0.7 and total_score >= max_score * 0.6:
                signal = "bullish"  # High quality company at fair price
            elif quality_score <= 0.4 or total_score <= max_score * 0.3:
                signal = "bearish"  # Poor quality or fundamentals
            else:
                signal = "neutral"

        # Confidence based on margin of safety and quality
        if mos is not None:
            confidence = min(max(abs(mos) * 150, 20), 95)  # 20-95% range
        else:
            ratio = total_score / max_score if max_score else 0.0
            confidence = min(max(ratio * 100, 10), 80)  # Based on score
        return signal, int(round(clamp(confidence, 0, 100)))

    # ---------------------------------------------------- upstream functions

    @staticmethod
    def analyze_profitability(items: list) -> dict[str, Any]:
        if not items:
            return {"score": 0, "details": "No profitability data available"}
        latest = items[0]
        score = 0
        reasoning = []

        # ROE (Return on Equity) - Jhunjhunwala's key metric
        if latest.net_income and latest.net_income > 0 and latest.total_assets and latest.total_liabilities:
            shareholders_equity = latest.total_assets - latest.total_liabilities
            if shareholders_equity > 0:
                roe = (latest.net_income / shareholders_equity) * 100
                if roe > 20:
                    score += 3
                    reasoning.append(f"Excellent ROE: {roe:.1f}%")
                elif roe > 15:
                    score += 2
                    reasoning.append(f"Good ROE: {roe:.1f}%")
                elif roe > 10:
                    score += 1
                    reasoning.append(f"Decent ROE: {roe:.1f}%")
                else:
                    reasoning.append(f"Low ROE: {roe:.1f}%")
            else:
                reasoning.append("Negative shareholders equity")
        else:
            reasoning.append("Unable to calculate ROE - missing data")

        # Operating margin
        if latest.operating_income and latest.revenue and latest.revenue > 0:
            operating_margin = (latest.operating_income / latest.revenue) * 100
            if operating_margin > 20:
                score += 2
                reasoning.append(f"Excellent operating margin: {operating_margin:.1f}%")
            elif operating_margin > 15:
                score += 1
                reasoning.append(f"Good operating margin: {operating_margin:.1f}%")
            elif operating_margin > 0:
                reasoning.append(f"Positive operating margin: {operating_margin:.1f}%")
            else:
                reasoning.append(f"Negative operating margin: {operating_margin:.1f}%")
        else:
            reasoning.append("Unable to calculate operating margin")

        # EPS growth consistency
        eps_values = [i.earnings_per_share for i in items if i.earnings_per_share is not None and i.earnings_per_share > 0]
        if len(eps_values) >= 3:
            growth = Persona.cagr(items, "earnings_per_share")
            if growth is not None:
                eps_cagr = growth[0] * 100
                if eps_cagr > 20:
                    score += 3
                    reasoning.append(f"High EPS CAGR: {eps_cagr:.1f}%")
                elif eps_cagr > 15:
                    score += 2
                    reasoning.append(f"Good EPS CAGR: {eps_cagr:.1f}%")
                elif eps_cagr > 10:
                    score += 1
                    reasoning.append(f"Moderate EPS CAGR: {eps_cagr:.1f}%")
                else:
                    reasoning.append(f"Low EPS CAGR: {eps_cagr:.1f}%")
            else:
                reasoning.append("Cannot calculate EPS growth (negative base or span under half a year)")
        else:
            reasoning.append("Insufficient EPS data for growth analysis")

        return {"score": score, "details": "; ".join(reasoning)}

    @staticmethod
    def analyze_growth(items: list) -> dict[str, Any]:
        if len(items) < 3:
            return {"score": 0, "details": "Insufficient data for growth analysis"}
        score = 0
        reasoning = []

        revenues = [i.revenue for i in items if i.revenue is not None and i.revenue > 0]
        if len(revenues) >= 3:
            growth = Persona.cagr(items, "revenue")
            if growth is not None:
                revenue_cagr = growth[0] * 100
                if revenue_cagr > 20:
                    score += 3
                    reasoning.append(f"Excellent revenue CAGR: {revenue_cagr:.1f}%")
                elif revenue_cagr > 15:
                    score += 2
                    reasoning.append(f"Good revenue CAGR: {revenue_cagr:.1f}%")
                elif revenue_cagr > 10:
                    score += 1
                    reasoning.append(f"Moderate revenue CAGR: {revenue_cagr:.1f}%")
                else:
                    reasoning.append(f"Low revenue CAGR: {revenue_cagr:.1f}%")
            else:
                reasoning.append("Cannot calculate revenue CAGR (zero base or span under half a year)")
        else:
            reasoning.append("Insufficient revenue data for CAGR calculation")

        net_incomes = [i.net_income for i in items if i.net_income is not None and i.net_income > 0]
        if len(net_incomes) >= 3:
            growth = Persona.cagr(items, "net_income")
            if growth is not None:
                income_cagr = growth[0] * 100
                if income_cagr > 25:
                    score += 3
                    reasoning.append(f"Excellent income CAGR: {income_cagr:.1f}%")
                elif income_cagr > 20:
                    score += 2
                    reasoning.append(f"High income CAGR: {income_cagr:.1f}%")
                elif income_cagr > 15:
                    score += 1
                    reasoning.append(f"Good income CAGR: {income_cagr:.1f}%")
                else:
                    reasoning.append(f"Moderate income CAGR: {income_cagr:.1f}%")
            else:
                reasoning.append("Cannot calculate income CAGR (zero base or span under half a year)")
        else:
            reasoning.append("Insufficient net income data for CAGR calculation")

        # Revenue consistency check (period over period)
        if len(revenues) >= 3:
            # rows are newest-first: a decline is newer < older
            declining_years = sum(1 for i in range(1, len(revenues)) if revenues[i - 1] < revenues[i])
            consistency_ratio = 1 - (declining_years / (len(revenues) - 1))
            if consistency_ratio >= 0.8:
                score += 1
                reasoning.append(f"Consistent growth pattern ({consistency_ratio * 100:.0f}% of periods)")
            else:
                reasoning.append(f"Inconsistent growth pattern ({consistency_ratio * 100:.0f}% of periods)")

        return {"score": score, "details": "; ".join(reasoning)}

    @staticmethod
    def analyze_balance_sheet(items: list) -> dict[str, Any]:
        if not items:
            return {"score": 0, "details": "No balance sheet data"}
        latest = items[0]
        score = 0
        reasoning = []

        if latest.total_assets and latest.total_liabilities and latest.total_assets > 0:
            debt_ratio = latest.total_liabilities / latest.total_assets
            if debt_ratio < 0.5:
                score += 2
                reasoning.append(f"Low debt ratio: {debt_ratio:.2f}")
            elif debt_ratio < 0.7:
                score += 1
                reasoning.append(f"Moderate debt ratio: {debt_ratio:.2f}")
            else:
                reasoning.append(f"High debt ratio: {debt_ratio:.2f}")
        else:
            reasoning.append("Insufficient data to calculate debt ratio")

        if latest.current_assets and latest.current_liabilities and latest.current_liabilities > 0:
            current_ratio = latest.current_assets / latest.current_liabilities
            if current_ratio > 2.0:
                score += 2
                reasoning.append(f"Excellent liquidity with current ratio: {current_ratio:.2f}")
            elif current_ratio > 1.5:
                score += 1
                reasoning.append(f"Good liquidity with current ratio: {current_ratio:.2f}")
            else:
                reasoning.append(f"Weak liquidity with current ratio: {current_ratio:.2f}")
        else:
            reasoning.append("Insufficient data to calculate current ratio")

        return {"score": score, "details": "; ".join(reasoning)}

    @staticmethod
    def analyze_cash_flow(items: list) -> dict[str, Any]:
        if not items:
            return {"score": 0, "details": "No cash flow data"}
        latest = items[0]
        score = 0
        reasoning = []

        if latest.free_cash_flow:
            if latest.free_cash_flow > 0:
                score += 2
                reasoning.append(f"Positive free cash flow: {latest.free_cash_flow}")
            else:
                reasoning.append(f"Negative free cash flow: {latest.free_cash_flow}")
        else:
            reasoning.append("Free cash flow data not available")

        if latest.dividends_and_other_cash_distributions:
            if latest.dividends_and_other_cash_distributions < 0:  # cash outflow for dividends
                score += 1
                reasoning.append("Company pays dividends to shareholders")
            else:
                reasoning.append("No significant dividend payments")
        else:
            reasoning.append("No dividend payment data available")

        return {"score": score, "details": "; ".join(reasoning)}

    @staticmethod
    def analyze_management_actions(items: list) -> dict[str, Any]:
        if not items:
            return {"score": 0, "details": "No management action data"}
        latest = items[0]
        score = 0
        reasoning = []

        issuance = latest.issuance_or_purchase_of_equity_shares
        if issuance is not None:
            if issuance < 0:  # buybacks
                score += 2
                reasoning.append(f"Company buying back shares: {abs(issuance)}")
            elif issuance > 0:
                reasoning.append(f"Share issuance detected (potential dilution): {issuance}")
            else:
                score += 1
                reasoning.append("No recent share issuance or buyback")
        else:
            reasoning.append("No data on share issuance or buybacks")

        return {"score": score, "details": "; ".join(reasoning)}

    @staticmethod
    def assess_quality_metrics(items: list) -> float:
        """Quality score in [0, 1]; 0.5 is the neutral default."""
        if not items:
            return 0.5
        latest = items[0]
        quality_factors: list[float] = []

        # ROE level
        if latest.net_income and latest.total_assets and latest.total_liabilities:
            shareholders_equity = latest.total_assets - latest.total_liabilities
            if shareholders_equity > 0 and latest.net_income:
                roe = latest.net_income / shareholders_equity
                if roe > 0.20:
                    quality_factors.append(1.0)
                elif roe > 0.15:
                    quality_factors.append(0.8)
                elif roe > 0.10:
                    quality_factors.append(0.6)
                else:
                    quality_factors.append(0.3)
            else:
                quality_factors.append(0.0)
        else:
            quality_factors.append(0.5)

        # Debt levels (lower is better)
        if latest.total_assets and latest.total_liabilities:
            debt_ratio = latest.total_liabilities / latest.total_assets
            if debt_ratio < 0.3:
                quality_factors.append(1.0)
            elif debt_ratio < 0.5:
                quality_factors.append(0.7)
            elif debt_ratio < 0.7:
                quality_factors.append(0.4)
            else:
                quality_factors.append(0.1)
        else:
            quality_factors.append(0.5)

        # Growth consistency
        net_incomes = [i.net_income for i in items[:4] if i.net_income is not None and i.net_income > 0]
        if len(net_incomes) >= 3:
            declining_years = sum(1 for i in range(1, len(net_incomes)) if net_incomes[i - 1] < net_incomes[i])  # newest-first
            consistency = 1 - (declining_years / (len(net_incomes) - 1))
            quality_factors.append(consistency)
        else:
            quality_factors.append(0.5)

        return sum(quality_factors) / len(quality_factors) if quality_factors else 0.5

    @classmethod
    def calculate_intrinsic_value(cls, items: list, market_cap: float | None) -> float | None:
        """Earnings-power DCF with a quality-dependent discount rate and terminal multiple."""
        if not items or not market_cap:
            return None
        latest = items[0]
        try:
            if not latest.net_income or latest.net_income <= 0:
                return None

            net_incomes = [i.net_income for i in items[:5] if i.net_income is not None and i.net_income > 0]
            if len(net_incomes) < 2:
                return latest.net_income * 12  # conservative P/E of 12

            growth = cls.cagr(items[:5], "net_income")
            historical_growth = growth[0] if growth is not None else 0.05

            if historical_growth > 0.25:
                sustainable_growth = 0.20
            elif historical_growth > 0.15:
                sustainable_growth = historical_growth * 0.8
            elif historical_growth > 0.05:
                sustainable_growth = historical_growth * 0.9
            else:
                sustainable_growth = 0.05

            quality_score = cls.assess_quality_metrics(items)
            if quality_score >= 0.8:
                discount_rate, terminal_multiple = 0.12, 18
            elif quality_score >= 0.6:
                discount_rate, terminal_multiple = 0.15, 15
            else:
                discount_rate, terminal_multiple = 0.18, 12

            current_earnings = latest.net_income
            dcf_value = 0.0
            for year in range(1, 6):
                projected_earnings = current_earnings * ((1 + sustainable_growth) ** year)
                dcf_value += projected_earnings / ((1 + discount_rate) ** year)
            year_5_earnings = current_earnings * ((1 + sustainable_growth) ** 5)
            terminal_value = (year_5_earnings * terminal_multiple) / ((1 + discount_rate) ** 5)
            return dcf_value + terminal_value
        except Exception:  # noqa: BLE001 — upstream fallback to a simple earnings multiple
            if latest.net_income and latest.net_income > 0:
                return latest.net_income * 15
            return None

    @classmethod
    def analyze_rakesh_jhunjhunwala_style(
        cls,
        items: list,
        owner_earnings: float | None = None,
        intrinsic_value: float | None = None,
        current_price: float | None = None,
    ) -> dict[str, Any]:
        """Upstream's aggregate: the five sub-analyses, their sum and the valuation gap."""
        profitability = cls.analyze_profitability(items)
        growth = cls.analyze_growth(items)
        balance_sheet = cls.analyze_balance_sheet(items)
        cash_flow = cls.analyze_cash_flow(items)
        management = cls.analyze_management_actions(items)

        total_score = profitability["score"] + growth["score"] + balance_sheet["score"] + cash_flow["score"] + management["score"]
        details = (
            f"Profitability: {profitability['details']}\n"
            f"Growth: {growth['details']}\n"
            f"Balance Sheet: {balance_sheet['details']}\n"
            f"Cash Flow: {cash_flow['details']}\n"
            f"Management Actions: {management['details']}"
        )
        if not intrinsic_value:
            intrinsic_value = cls.calculate_intrinsic_value(items, current_price)
        valuation_gap = None
        if intrinsic_value and current_price:
            valuation_gap = intrinsic_value - current_price
        return {
            "total_score": total_score,
            "details": details,
            "owner_earnings": owner_earnings,
            "intrinsic_value": intrinsic_value,
            "current_price": current_price,
            "valuation_gap": valuation_gap,
            "breakdown": {
                "profitability": profitability,
                "growth": growth,
                "balance_sheet": balance_sheet,
                "cash_flow": cash_flow,
                "management": management,
            },
        }
