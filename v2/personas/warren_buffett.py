"""Warren Buffett — wonderful companies at a fair price.

Ported from ``src/agents/warren_buffett.py`` at virattt/ai-hedge-fund
v2026.5.14 (MIT).  The seven scoring functions are kept rule-for-rule; the
LangGraph state, progress bar and LLM call are gone.  The LLM's old job —
turning ``score / max_score`` and the margin of safety into a verdict — is
now :meth:`WarrenBuffett.decide`, which encodes the prompt's signal rules:
bullish needs a strong business *and* a positive margin of safety.
"""

from __future__ import annotations

from typing import Any

from v2.personas.base import Persona, apply_margin_of_safety, clamp, classic_verdict, confidence_from_ratio
from v2.personas.models import Evaluation, Signal
from v2.personas.snapshot import PersonaSnapshot


class WarrenBuffett(Persona):
    key = "warren_buffett"
    name = "Warren Buffett"
    name_zh = "沃伦·巴菲特"
    style = "seeks wonderful companies at a fair price"
    period = "ttm"
    lookback = 10
    needs = frozenset()
    system_prompt = (
        "You are Warren Buffett. Explain a bullish, bearish, or neutral verdict using only the provided facts.\n"
        "Checklist: circle of competence; competitive moat; management quality; financial strength; "
        "valuation vs intrinsic value; long-term prospects.\n"
        "Signal rules: bullish = strong business AND margin_of_safety > 0; bearish = poor business OR clearly "
        "overvalued; neutral = good business but margin_of_safety <= 0, or mixed evidence."
    )

    # ------------------------------------------------------------------ rules

    def evaluate(self, snap: PersonaSnapshot) -> Evaluation:
        metrics = snap.metrics(self.period, self.lookback)
        items = snap.line_items(self.period, self.lookback)
        market_cap = snap.market_cap

        fundamentals = self.analyze_fundamentals(metrics)
        consistency = self.analyze_consistency(items)
        moat = self.analyze_moat(metrics)
        management = self.analyze_management_quality(items)
        pricing = self.analyze_pricing_power(items, metrics)
        book_value = self.analyze_book_value_growth(items)
        valuation = self.calculate_intrinsic_value(items)

        ev = Evaluation(parts=[
            self.part("fundamentals", fundamentals["score"], 10, fundamentals["details"]),
            self.part("consistency", consistency["score"], 3, consistency["details"]),
            self.part("moat", moat["score"], moat["max_score"], moat["details"]),
            self.part("management", management["score"], management["max_score"], management["details"]),
            self.part("pricing_power", pricing["score"], 5, pricing["details"]),
            self.part("book_value_growth", book_value["score"], 5, book_value["details"]),
        ])

        intrinsic = valuation.get("intrinsic_value")
        if intrinsic and market_cap:
            ev.margin_of_safety = (intrinsic - market_cap) / market_cap
        ev.facts = {
            "intrinsic_value": intrinsic,
            "raw_intrinsic_value": valuation.get("raw_intrinsic_value"),
            "owner_earnings": valuation.get("owner_earnings"),
            "market_cap": market_cap,
            "margin_of_safety": ev.margin_of_safety,
            "dcf_assumptions": valuation.get("assumptions"),
            "valuation_details": valuation.get("details"),
        }
        if not metrics and not items:
            ev.insufficient = True
            ev.notes.append("no fundamentals")
        return ev

    def decide(self, ev: Evaluation) -> tuple[Signal, int]:
        signal = classic_verdict(ev.ratio)
        confidence = confidence_from_ratio(ev.ratio, signal)
        # Buffett is explicit: no margin of safety, no purchase — and a clearly
        # overvalued business is a bearish call even when the checklist is fine.
        signal, confidence = apply_margin_of_safety(signal, confidence, ev.margin_of_safety, overvalued_at=-0.3)
        if signal == "bullish" and ev.margin_of_safety is not None and ev.margin_of_safety > 0.3:
            confidence = int(clamp(confidence + 5, 10, 95))
        return signal, confidence

    # ---------------------------------------------------- upstream functions

    @staticmethod
    def analyze_fundamentals(metrics: list) -> dict[str, Any]:
        if not metrics:
            return {"score": 0, "details": "Insufficient fundamental data"}
        m = metrics[0]
        score = 0
        reasoning = []
        if m.return_on_equity and m.return_on_equity > 0.15:
            score += 2
            reasoning.append(f"Strong ROE of {m.return_on_equity:.1%}")
        elif m.return_on_equity:
            reasoning.append(f"Weak ROE of {m.return_on_equity:.1%}")
        else:
            reasoning.append("ROE data not available")
        if m.debt_to_equity and m.debt_to_equity < 0.5:
            score += 2
            reasoning.append("Conservative debt levels")
        elif m.debt_to_equity:
            reasoning.append(f"High debt to equity ratio of {m.debt_to_equity:.1f}")
        else:
            reasoning.append("Debt to equity data not available")
        if m.operating_margin and m.operating_margin > 0.15:
            score += 2
            reasoning.append("Strong operating margins")
        elif m.operating_margin:
            reasoning.append(f"Weak operating margin of {m.operating_margin:.1%}")
        else:
            reasoning.append("Operating margin data not available")
        if m.current_ratio and m.current_ratio > 1.5:
            score += 1
            reasoning.append("Good liquidity position")
        elif m.current_ratio:
            reasoning.append(f"Weak liquidity with current ratio of {m.current_ratio:.1f}")
        else:
            reasoning.append("Current ratio data not available")
        return {"score": score, "details": "; ".join(reasoning)}

    @staticmethod
    def analyze_consistency(items: list) -> dict[str, Any]:
        if len(items) < 4:
            return {"score": 0, "details": "Insufficient historical data"}
        score = 0
        reasoning = []
        earnings = [i.net_income for i in items if i.net_income]
        if len(earnings) >= 4:
            growing = all(earnings[i] > earnings[i + 1] for i in range(len(earnings) - 1))
            if growing:
                score += 3
                reasoning.append("Consistent earnings growth over past periods")
            else:
                reasoning.append("Inconsistent earnings growth pattern")
            if earnings[-1] != 0:
                growth = (earnings[0] - earnings[-1]) / abs(earnings[-1])
                reasoning.append(f"Total earnings growth of {growth:.1%} over past {len(earnings)} periods")
        else:
            reasoning.append("Insufficient earnings data for trend analysis")
        return {"score": score, "details": "; ".join(reasoning)}

    @staticmethod
    def analyze_moat(metrics: list) -> dict[str, Any]:
        max_score = 5
        if not metrics or len(metrics) < 5:
            return {"score": 0, "max_score": max_score, "details": "Insufficient data for comprehensive moat analysis"}
        reasoning = []
        score = 0
        roes = [m.return_on_equity for m in metrics if m.return_on_equity is not None]
        margins = [m.operating_margin for m in metrics if m.operating_margin is not None]

        if len(roes) >= 5:
            high = sum(1 for r in roes if r > 0.15)
            consistency = high / len(roes)
            if consistency >= 0.8:
                score += 2
                reasoning.append(f"Excellent ROE consistency: {high}/{len(roes)} periods >15% (avg: {sum(roes) / len(roes):.1%}) - indicates durable competitive advantage")
            elif consistency >= 0.6:
                score += 1
                reasoning.append(f"Good ROE performance: {high}/{len(roes)} periods >15%")
            else:
                reasoning.append(f"Inconsistent ROE: only {high}/{len(roes)} periods >15%")
        else:
            reasoning.append("Insufficient ROE history for moat analysis")

        if len(margins) >= 5:
            avg = sum(margins) / len(margins)
            recent = sum(margins[:3]) / 3
            older = sum(margins[-3:]) / 3
            if avg > 0.2 and recent >= older:
                score += 1
                reasoning.append(f"Strong and stable operating margins (avg: {avg:.1%}) indicate pricing power moat")
            elif avg > 0.15:
                reasoning.append(f"Decent operating margins (avg: {avg:.1%}) suggest some competitive advantage")
            else:
                reasoning.append(f"Low operating margins (avg: {avg:.1%}) suggest limited pricing power")

        turnovers = [m.asset_turnover for m in metrics if m.asset_turnover is not None]
        if len(turnovers) >= 3 and any(t > 1.0 for t in turnovers):
            score += 1
            reasoning.append("Efficient asset utilization suggests operational moat")

        if len(roes) >= 5 and len(margins) >= 5:
            roe_avg = sum(roes) / len(roes)
            roe_var = sum((r - roe_avg) ** 2 for r in roes) / len(roes)
            roe_stab = 1 - (roe_var ** 0.5) / roe_avg if roe_avg > 0 else 0
            m_avg = sum(margins) / len(margins)
            m_var = sum((x - m_avg) ** 2 for x in margins) / len(margins)
            m_stab = 1 - (m_var ** 0.5) / m_avg if m_avg > 0 else 0
            stability = (roe_stab + m_stab) / 2
            if stability > 0.7:
                score += 1
                reasoning.append(f"High performance stability ({stability:.1%}) suggests strong competitive moat")

        return {"score": min(score, max_score), "max_score": max_score, "details": "; ".join(reasoning) or "Limited moat analysis available"}

    @staticmethod
    def analyze_management_quality(items: list) -> dict[str, Any]:
        if not items:
            return {"score": 0, "max_score": 2, "details": "Insufficient data for management analysis"}
        latest = items[0]
        score = 0
        reasoning = []
        issuance = latest.issuance_or_purchase_of_equity_shares
        if issuance and issuance < 0:
            score += 1
            reasoning.append("Company has been repurchasing shares (shareholder-friendly)")
        if issuance and issuance > 0:
            reasoning.append("Recent common stock issuance (potential dilution)")
        else:
            reasoning.append("No significant new stock issuance detected")
        dividends = latest.dividends_and_other_cash_distributions
        if dividends and dividends < 0:
            score += 1
            reasoning.append("Company has a track record of paying dividends")
        else:
            reasoning.append("No or minimal dividends paid")
        return {"score": score, "max_score": 2, "details": "; ".join(reasoning)}

    @staticmethod
    def analyze_pricing_power(items: list, metrics: list) -> dict[str, Any]:
        if not items or not metrics:
            return {"score": 0, "details": "Insufficient data for pricing power analysis"}
        score = 0
        reasoning = []
        gms = [i.gross_margin for i in items if i.gross_margin is not None]
        if len(gms) >= 3:
            recent = sum(gms[:2]) / 2
            older = sum(gms[-2:]) / 2
            if recent > older + 0.02:
                score += 3
                reasoning.append("Expanding gross margins indicate strong pricing power")
            elif recent > older:
                score += 2
                reasoning.append("Improving gross margins suggest good pricing power")
            elif abs(recent - older) < 0.01:
                score += 1
                reasoning.append("Stable gross margins during economic uncertainty")
            else:
                reasoning.append("Declining gross margins may indicate pricing pressure")
        if gms:
            avg = sum(gms) / len(gms)
            if avg > 0.5:
                score += 2
                reasoning.append(f"Consistently high gross margins ({avg:.1%}) indicate strong pricing power")
            elif avg > 0.3:
                score += 1
                reasoning.append(f"Good gross margins ({avg:.1%}) suggest decent pricing power")
        return {"score": min(score, 5), "details": "; ".join(reasoning) or "Limited pricing power analysis available"}

    @classmethod
    def analyze_book_value_growth(cls, items: list) -> dict[str, Any]:
        if len(items) < 3:
            return {"score": 0, "details": "Insufficient data for book value analysis"}
        book_values = [i.shareholders_equity / i.outstanding_shares for i in items if i.shareholders_equity and i.outstanding_shares]
        if len(book_values) < 3:
            return {"score": 0, "details": "Insufficient book value data for growth analysis"}
        score = 0
        reasoning = []
        periods = sum(1 for i in range(len(book_values) - 1) if book_values[i] > book_values[i + 1])
        rate = periods / (len(book_values) - 1)
        if rate >= 0.8:
            score += 3
            reasoning.append("Consistent book value per share growth (Buffett's favorite metric)")
        elif rate >= 0.6:
            score += 2
            reasoning.append("Good book value per share growth pattern")
        elif rate >= 0.4:
            score += 1
            reasoning.append("Moderate book value per share growth")
        else:
            reasoning.append("Inconsistent book value per share growth")
        cagr_score, cagr_reason = cls._book_value_cagr(book_values)
        score += cagr_score
        reasoning.append(cagr_reason)
        return {"score": min(score, 5), "details": "; ".join(reasoning)}

    @staticmethod
    def _book_value_cagr(book_values: list[float]) -> tuple[int, str]:
        if len(book_values) < 2:
            return 0, "Insufficient data for CAGR calculation"
        oldest, latest = book_values[-1], book_values[0]
        years = len(book_values) - 1
        if oldest > 0 and latest > 0:
            cagr = (latest / oldest) ** (1 / years) - 1
            if cagr > 0.15:
                return 2, f"Excellent book value CAGR: {cagr:.1%}"
            if cagr > 0.1:
                return 1, f"Good book value CAGR: {cagr:.1%}"
            return 0, f"Book value CAGR: {cagr:.1%}"
        if oldest < 0 < latest:
            return 3, "Excellent: Company improved from negative to positive book value"
        if oldest > 0 > latest:
            return 0, "Warning: Company declined from positive to negative book value"
        return 0, "Unable to calculate meaningful book value CAGR due to negative values"

    @staticmethod
    def estimate_maintenance_capex(items: list) -> float:
        if not items:
            return 0.0
        capex_ratios = []
        for item in items[:5]:
            if item.capital_expenditure and item.revenue and item.revenue > 0:
                capex_ratios.append(abs(item.capital_expenditure) / item.revenue)
        latest_dep = items[0].depreciation_and_amortization or 0
        latest_capex = abs(items[0].capital_expenditure) if items[0].capital_expenditure else 0
        method_1 = latest_capex * 0.85
        method_2 = latest_dep
        if len(capex_ratios) >= 3:
            latest_rev = items[0].revenue or 0
            method_3 = (sum(capex_ratios) / len(capex_ratios)) * latest_rev if latest_rev else 0
            return sorted([method_1, method_2, method_3])[1]
        return max(method_1, method_2)

    @classmethod
    def calculate_owner_earnings(cls, items: list) -> dict[str, Any]:
        if not items or len(items) < 2:
            return {"owner_earnings": None, "details": ["Insufficient data for owner earnings calculation"]}
        latest = items[0]
        details = []
        net_income, dep, capex = latest.net_income, latest.depreciation_and_amortization, latest.capital_expenditure
        if net_income is None or dep is None or capex is None:
            missing = [n for n, v in (("net income", net_income), ("depreciation", dep), ("capital expenditure", capex)) if v is None]
            return {"owner_earnings": None, "details": [f"Missing components: {', '.join(missing)}"]}
        maintenance_capex = cls.estimate_maintenance_capex(items)
        wc_change = 0.0
        prev = items[1]
        if all(v for v in (latest.current_assets, latest.current_liabilities, prev.current_assets, prev.current_liabilities)):
            wc_change = (latest.current_assets - latest.current_liabilities) - (prev.current_assets - prev.current_liabilities)
            details.append(f"Working capital change: ${wc_change:,.0f}")
        owner_earnings = net_income + dep - maintenance_capex - wc_change
        if owner_earnings < net_income * 0.3:
            details.append("Warning: Owner earnings significantly below net income - high capex intensity")
        if maintenance_capex > dep * 2:
            details.append("Warning: Estimated maintenance capex seems high relative to depreciation")
        details.extend([
            f"Net income: ${net_income:,.0f}",
            f"Depreciation: ${dep:,.0f}",
            f"Estimated maintenance capex: ${maintenance_capex:,.0f}",
            f"Owner earnings: ${owner_earnings:,.0f}",
        ])
        return {
            "owner_earnings": owner_earnings,
            "components": {"net_income": net_income, "depreciation": dep, "maintenance_capex": maintenance_capex, "working_capital_change": wc_change, "total_capex": abs(capex)},
            "details": details,
        }

    @classmethod
    def calculate_intrinsic_value(cls, items: list) -> dict[str, Any]:
        if not items or len(items) < 3:
            return {"intrinsic_value": None, "details": ["Insufficient data for reliable valuation"]}
        earnings_data = cls.calculate_owner_earnings(items)
        if not earnings_data["owner_earnings"]:
            return {"intrinsic_value": None, "details": earnings_data["details"]}
        owner_earnings = earnings_data["owner_earnings"]
        shares = items[0].outstanding_shares
        if not shares or shares <= 0:
            return {"intrinsic_value": None, "details": ["Missing or invalid shares outstanding data"]}

        # Date-based growth (rows may be quarterly TTM, so row count ≠ years); a loss in the
        # newest period is a decline, not a complex number — clamp it to the floor.
        newest_income = items[0].net_income
        if newest_income is not None and float(newest_income) <= 0:
            conservative_growth = -0.05 * 0.7
        else:
            grown = cls.cagr(items[:5], "net_income", min_years=1.0)
            if grown is not None:
                growth = max(-0.05, min(grown[0], 0.15))
                conservative_growth = growth * 0.7
            else:
                conservative_growth = 0.03

        stage1_growth = min(conservative_growth, 0.08)
        stage2_growth = min(conservative_growth * 0.5, 0.04)
        terminal_growth = 0.025
        discount_rate = 0.10
        stage1_years = stage2_years = 5

        stage1_pv = sum(owner_earnings * (1 + stage1_growth) ** y / (1 + discount_rate) ** y for y in range(1, stage1_years + 1))
        stage1_final = owner_earnings * (1 + stage1_growth) ** stage1_years
        stage2_pv = sum(stage1_final * (1 + stage2_growth) ** y / (1 + discount_rate) ** (stage1_years + y) for y in range(1, stage2_years + 1))
        final = stage1_final * (1 + stage2_growth) ** stage2_years
        terminal_value = final * (1 + terminal_growth) / (discount_rate - terminal_growth)
        terminal_pv = terminal_value / (1 + discount_rate) ** (stage1_years + stage2_years)
        intrinsic = stage1_pv + stage2_pv + terminal_pv
        conservative = intrinsic * 0.85

        details = [
            f"Using three-stage DCF: Stage 1 ({stage1_growth:.1%}, {stage1_years}y), Stage 2 ({stage2_growth:.1%}, {stage2_years}y), Terminal ({terminal_growth:.1%})",
            f"Stage 1 PV: ${stage1_pv:,.0f}",
            f"Stage 2 PV: ${stage2_pv:,.0f}",
            f"Terminal PV: ${terminal_pv:,.0f}",
            f"Total IV: ${intrinsic:,.0f}",
            f"Conservative IV (15% haircut): ${conservative:,.0f}",
            f"Owner earnings: ${owner_earnings:,.0f}",
            f"Discount rate: {discount_rate:.1%}",
        ]
        return {
            "intrinsic_value": conservative,
            "raw_intrinsic_value": intrinsic,
            "owner_earnings": owner_earnings,
            "assumptions": {
                "stage1_growth": stage1_growth,
                "stage2_growth": stage2_growth,
                "terminal_growth": terminal_growth,
                "discount_rate": discount_rate,
                "stage1_years": stage1_years,
                "stage2_years": stage2_years,
                "historical_growth": conservative_growth,
            },
            "details": details,
        }
