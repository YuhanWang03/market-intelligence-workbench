"""Phil Fisher — long-term growth, R&D and management quality, checked by scuttlebutt.

Ported from ``src/agents/phil_fisher.py`` at virattt/ai-hedge-fund
v2026.5.14 (MIT).  The six scoring functions are kept rule-for-rule, each
scoring 0-10 exactly as upstream; the LangGraph state, progress bar and LLM
call are gone.  Upstream then blended the six with fixed weights
(30/25/20/15/5/5 %) into a 0-10 total and cut at 7.5 / 4.5 — here every
part's ``max_score`` is ``10 * weight`` so the weighted blend *is* the sum
of the parts, and :meth:`PhilFisher.decide` applies the same two cuts.
Fisher never computed an intrinsic value, so there is no margin of safety.
"""

from __future__ import annotations

import statistics
from typing import Any

from v2.personas.base import Persona, classic_verdict, confidence_from_ratio
from v2.personas.models import Evaluation, Signal
from v2.personas.snapshot import PersonaSnapshot

#: upstream fetched the 50 most recent insider trades / news items
_INSIDER_LIMIT = 50
_NEWS_LIMIT = 50

#: upstream total-score cuts (on a 0-10 scale) expressed as ratios
_BULLISH_AT = 7.5 / 10
_BEARISH_AT = 4.5 / 10


class PhilFisher(Persona):
    key = "phil_fisher"
    name = "Phil Fisher"
    name_zh = "菲利普·费雪"
    style = "uses deep scuttlebutt research"
    period = "annual"
    lookback = 5
    needs = frozenset({"insiders", "news"})
    system_prompt = (
        "You are Phil Fisher. Judge a company by its long-term growth potential and the quality of its "
        "management, using only the provided facts.\n"
        "Principles: 1) emphasize sustained above-average growth and management quality; 2) favor companies "
        "investing in R&D for future products; 3) look for strong, consistent margins and profitability; "
        "4) pay up for exceptional companies but stay mindful of valuation; 5) rely on thorough scuttlebutt "
        "and fundamental checks (insider activity, sentiment).\n"
        "Discuss growth with specific metrics and trends, management's capital allocation, R&D intensity, "
        "margin consistency, and advantages that can sustain growth for 3-5+ years, in a methodical, "
        "long-term voice.\n"
        "Signal rules: the weighted score (growth 30%, margins 25%, management 20%, valuation 15%, insiders "
        "5%, sentiment 5%) is bullish at >= 7.5/10, bearish at <= 4.5/10, neutral between. Confidence 0-100."
    )

    #: upstream weights, in evaluation order
    WEIGHTS = {
        "growth_quality": 0.30,
        "margins_stability": 0.25,
        "management_efficiency": 0.20,
        "valuation": 0.15,
        "insider_activity": 0.05,
        "sentiment": 0.05,
    }

    # ------------------------------------------------------------------ rules

    def evaluate(self, snap: PersonaSnapshot) -> Evaluation:
        items = snap.line_items(self.period, self.lookback)
        market_cap = snap.market_cap
        insider_trades = list(snap.insider_trades)[:_INSIDER_LIMIT]
        news = list(snap.news)[:_NEWS_LIMIT]

        results = {
            "growth_quality": self.analyze_fisher_growth_quality(items),
            "margins_stability": self.analyze_margins_stability(items),
            "management_efficiency": self.analyze_management_efficiency_leverage(items),
            "valuation": self.analyze_fisher_valuation(items, market_cap),
            "insider_activity": self.analyze_insider_activity(insider_trades),
            "sentiment": self.analyze_sentiment(news),
        }

        parts = []
        for name, weight in self.WEIGHTS.items():
            r = results[name]
            parts.append(self.part(name, r["score"] * weight, 10 * weight, r["details"], raw_score=r["score"], weight=weight))
        ev = Evaluation(parts=parts)

        latest = items[0] if items else None
        pe = pfcf = None
        if latest is not None and market_cap:
            if latest.net_income and latest.net_income > 0:
                pe = market_cap / latest.net_income
            if latest.free_cash_flow and latest.free_cash_flow > 0:
                pfcf = market_cap / latest.free_cash_flow
        ev.facts = {
            "market_cap": market_cap,
            "pe": pe,
            "p_fcf": pfcf,
            "weighted_score": ev.score,
            "max_score": 10,
            "insider_trades_considered": len(insider_trades),
            "news_items_considered": len(news),
        }
        if not items:
            ev.insufficient = True
            ev.notes.append("no annual line items")
        return ev

    def decide(self, ev: Evaluation) -> tuple[Signal, int]:
        # Upstream: total >= 7.5 → bullish, <= 4.5 → bearish, else neutral (out of 10).
        signal = classic_verdict(ev.ratio, bullish_at=_BULLISH_AT, bearish_at=_BEARISH_AT)
        confidence = confidence_from_ratio(ev.ratio, signal, bullish_at=_BULLISH_AT, bearish_at=_BEARISH_AT)
        return signal, confidence

    # ---------------------------------------------------- upstream functions

    @staticmethod
    def analyze_fisher_growth_quality(financial_line_items: list) -> dict[str, Any]:
        """Revenue CAGR, EPS CAGR and R&D intensity; 9 raw points scaled to 0-10."""
        if not financial_line_items or len(financial_line_items) < 2:
            return {"score": 0, "details": "Insufficient financial data for growth/quality analysis"}

        details = []
        raw_score = 0

        revenues = [fi.revenue for fi in financial_line_items if fi.revenue is not None]
        if len(revenues) >= 2:
            latest_rev = revenues[0]
            oldest_rev = revenues[-1]
            num_years = len(revenues) - 1
            if oldest_rev > 0 and latest_rev > 0:
                rev_growth = (latest_rev / oldest_rev) ** (1 / num_years) - 1
                if rev_growth > 0.20:
                    raw_score += 3
                    details.append(f"Very strong annualized revenue growth: {rev_growth:.1%}")
                elif rev_growth > 0.10:
                    raw_score += 2
                    details.append(f"Moderate annualized revenue growth: {rev_growth:.1%}")
                elif rev_growth > 0.03:
                    raw_score += 1
                    details.append(f"Slight annualized revenue growth: {rev_growth:.1%}")
                else:
                    details.append(f"Minimal or negative annualized revenue growth: {rev_growth:.1%}")
            else:
                details.append("Oldest revenue is zero/negative; cannot compute growth.")
        else:
            details.append("Not enough revenue data points for growth calculation.")

        eps_values = [fi.earnings_per_share for fi in financial_line_items if fi.earnings_per_share is not None]
        if len(eps_values) >= 2:
            latest_eps = eps_values[0]
            oldest_eps = eps_values[-1]
            num_years = len(eps_values) - 1
            if oldest_eps > 0 and latest_eps > 0:
                eps_growth = (latest_eps / oldest_eps) ** (1 / num_years) - 1
                if eps_growth > 0.20:
                    raw_score += 3
                    details.append(f"Very strong annualized EPS growth: {eps_growth:.1%}")
                elif eps_growth > 0.10:
                    raw_score += 2
                    details.append(f"Moderate annualized EPS growth: {eps_growth:.1%}")
                elif eps_growth > 0.03:
                    raw_score += 1
                    details.append(f"Slight annualized EPS growth: {eps_growth:.1%}")
                else:
                    details.append(f"Minimal or negative annualized EPS growth: {eps_growth:.1%}")
            else:
                details.append("Oldest EPS near zero; skipping EPS growth calculation.")
        else:
            details.append("Not enough EPS data points for growth calculation.")

        rnd_values = [fi.research_and_development for fi in financial_line_items if fi.research_and_development is not None]
        if rnd_values and revenues and len(rnd_values) == len(revenues):
            recent_rnd = rnd_values[0]
            recent_rev = revenues[0] if revenues[0] else 1e-9
            rnd_ratio = recent_rnd / recent_rev
            if 0.03 <= rnd_ratio <= 0.15:
                raw_score += 3
                details.append(f"R&D ratio {rnd_ratio:.1%} indicates significant investment in future growth")
            elif rnd_ratio > 0.15:
                raw_score += 2
                details.append(f"R&D ratio {rnd_ratio:.1%} is very high (could be good if well-managed)")
            elif rnd_ratio > 0.0:
                raw_score += 1
                details.append(f"R&D ratio {rnd_ratio:.1%} is somewhat low but still positive")
            else:
                details.append("No meaningful R&D expense ratio")
        else:
            details.append("Insufficient R&D data to evaluate")

        final_score = min(10, (raw_score / 9) * 10)
        return {"score": final_score, "details": "; ".join(details)}

    @staticmethod
    def analyze_margins_stability(financial_line_items: list) -> dict[str, Any]:
        """Operating-margin trend, gross-margin level and margin volatility; 6 raw points scaled to 0-10."""
        if not financial_line_items or len(financial_line_items) < 2:
            return {"score": 0, "details": "Insufficient data for margin stability analysis"}

        details = []
        raw_score = 0

        op_margins = [fi.operating_margin for fi in financial_line_items if fi.operating_margin is not None]
        if len(op_margins) >= 2:
            oldest_op_margin = op_margins[-1]
            newest_op_margin = op_margins[0]
            if newest_op_margin >= oldest_op_margin > 0:
                raw_score += 2
                details.append(f"Operating margin stable or improving ({oldest_op_margin:.1%} -> {newest_op_margin:.1%})")
            elif newest_op_margin > 0:
                raw_score += 1
                details.append("Operating margin positive but slightly declined")
            else:
                details.append("Operating margin may be negative or uncertain")
        else:
            details.append("Not enough operating margin data points")

        gm_values = [fi.gross_margin for fi in financial_line_items if fi.gross_margin is not None]
        if gm_values:
            recent_gm = gm_values[0]
            if recent_gm > 0.5:
                raw_score += 2
                details.append(f"Strong gross margin: {recent_gm:.1%}")
            elif recent_gm > 0.3:
                raw_score += 1
                details.append(f"Moderate gross margin: {recent_gm:.1%}")
            else:
                details.append(f"Low gross margin: {recent_gm:.1%}")
        else:
            details.append("No gross margin data available")

        if len(op_margins) >= 3:
            stdev = statistics.pstdev(op_margins)
            if stdev < 0.02:
                raw_score += 2
                details.append("Operating margin extremely stable over multiple years")
            elif stdev < 0.05:
                raw_score += 1
                details.append("Operating margin reasonably stable")
            else:
                details.append("Operating margin volatility is high")
        else:
            details.append("Not enough margin data points for volatility check")

        final_score = min(10, (raw_score / 6) * 10)
        return {"score": final_score, "details": "; ".join(details)}

    @staticmethod
    def analyze_management_efficiency_leverage(financial_line_items: list) -> dict[str, Any]:
        """ROE, debt-to-equity and FCF consistency; 6 raw points scaled to 0-10."""
        if not financial_line_items:
            return {"score": 0, "details": "No financial data for management efficiency analysis"}

        details = []
        raw_score = 0

        ni_values = [fi.net_income for fi in financial_line_items if fi.net_income is not None]
        eq_values = [fi.shareholders_equity for fi in financial_line_items if fi.shareholders_equity is not None]
        if ni_values and eq_values and len(ni_values) == len(eq_values):
            recent_ni = ni_values[0]
            recent_eq = eq_values[0] if eq_values[0] else 1e-9
            if recent_ni > 0:
                roe = recent_ni / recent_eq
                if roe > 0.2:
                    raw_score += 3
                    details.append(f"High ROE: {roe:.1%}")
                elif roe > 0.1:
                    raw_score += 2
                    details.append(f"Moderate ROE: {roe:.1%}")
                elif roe > 0:
                    raw_score += 1
                    details.append(f"Positive but low ROE: {roe:.1%}")
                else:
                    details.append(f"ROE is near zero or negative: {roe:.1%}")
            else:
                details.append("Recent net income is zero or negative, hurting ROE")
        else:
            details.append("Insufficient data for ROE calculation")

        debt_values = [fi.total_debt for fi in financial_line_items if fi.total_debt is not None]
        if debt_values and eq_values and len(debt_values) == len(eq_values):
            recent_debt = debt_values[0]
            recent_equity = eq_values[0] if eq_values[0] else 1e-9
            dte = recent_debt / recent_equity
            if dte < 0.3:
                raw_score += 2
                details.append(f"Low debt-to-equity: {dte:.2f}")
            elif dte < 1.0:
                raw_score += 1
                details.append(f"Manageable debt-to-equity: {dte:.2f}")
            else:
                details.append(f"High debt-to-equity: {dte:.2f}")
        else:
            details.append("Insufficient data for debt/equity analysis")

        fcf_values = [fi.free_cash_flow for fi in financial_line_items if fi.free_cash_flow is not None]
        if fcf_values and len(fcf_values) >= 2:
            positive_fcf_count = sum(1 for x in fcf_values if x and x > 0)
            ratio = positive_fcf_count / len(fcf_values)
            if ratio > 0.8:
                raw_score += 1
                details.append(f"Majority of periods have positive FCF ({positive_fcf_count}/{len(fcf_values)})")
            else:
                details.append("Free cash flow is inconsistent or often negative")
        else:
            details.append("Insufficient or no FCF data to check consistency")

        final_score = min(10, (raw_score / 6) * 10)
        return {"score": final_score, "details": "; ".join(details)}

    @staticmethod
    def analyze_fisher_valuation(financial_line_items: list, market_cap: float | None) -> dict[str, Any]:
        """P/E and P/FCF, 2 points each; 4 raw points scaled to 0-10."""
        if not financial_line_items or market_cap is None:
            return {"score": 0, "details": "Insufficient data to perform valuation"}

        details = []
        raw_score = 0

        net_incomes = [fi.net_income for fi in financial_line_items if fi.net_income is not None]
        fcf_values = [fi.free_cash_flow for fi in financial_line_items if fi.free_cash_flow is not None]

        recent_net_income = net_incomes[0] if net_incomes else None
        if recent_net_income and recent_net_income > 0:
            pe = market_cap / recent_net_income
            pe_points = 0
            if pe < 20:
                pe_points = 2
                details.append(f"Reasonably attractive P/E: {pe:.2f}")
            elif pe < 30:
                pe_points = 1
                details.append(f"Somewhat high but possibly justifiable P/E: {pe:.2f}")
            else:
                details.append(f"Very high P/E: {pe:.2f}")
            raw_score += pe_points
        else:
            details.append("No positive net income for P/E calculation")

        recent_fcf = fcf_values[0] if fcf_values else None
        if recent_fcf and recent_fcf > 0:
            pfcf = market_cap / recent_fcf
            pfcf_points = 0
            if pfcf < 20:
                pfcf_points = 2
                details.append(f"Reasonable P/FCF: {pfcf:.2f}")
            elif pfcf < 30:
                pfcf_points = 1
                details.append(f"Somewhat high P/FCF: {pfcf:.2f}")
            else:
                details.append(f"Excessively high P/FCF: {pfcf:.2f}")
            raw_score += pfcf_points
        else:
            details.append("No positive free cash flow for P/FCF calculation")

        final_score = min(10, (raw_score / 4) * 10)
        return {"score": final_score, "details": "; ".join(details)}

    @staticmethod
    def analyze_insider_activity(insider_trades: list) -> dict[str, Any]:
        """Buy/sell count ratio; neutral 5, heavy buying 8, moderate 6, mostly selling 4."""
        score = 5
        details = []

        if not insider_trades:
            details.append("No insider trades data; defaulting to neutral")
            return {"score": score, "details": "; ".join(details)}

        buys, sells = 0, 0
        for trade in insider_trades:
            if trade.transaction_shares is not None:
                if trade.transaction_shares > 0:
                    buys += 1
                elif trade.transaction_shares < 0:
                    sells += 1

        total = buys + sells
        if total == 0:
            details.append("No buy/sell transactions found; neutral")
            return {"score": score, "details": "; ".join(details)}

        buy_ratio = buys / total
        if buy_ratio > 0.7:
            score = 8
            details.append(f"Heavy insider buying: {buys} buys vs. {sells} sells")
        elif buy_ratio > 0.4:
            score = 6
            details.append(f"Moderate insider buying: {buys} buys vs. {sells} sells")
        else:
            score = 4
            details.append(f"Mostly insider selling: {buys} buys vs. {sells} sells")

        return {"score": score, "details": "; ".join(details)}

    @staticmethod
    def analyze_sentiment(news_items: list) -> dict[str, Any]:
        """Negative-keyword headline check; 3 / 6 / 8 out of 10, neutral 5 with no news."""
        if not news_items:
            return {"score": 5, "details": "No news data; defaulting to neutral sentiment"}

        negative_keywords = ["lawsuit", "fraud", "negative", "downturn", "decline", "investigation", "recall"]
        negative_count = 0
        for news in news_items:
            title_lower = (news.title or "").lower()
            if any(word in title_lower for word in negative_keywords):
                negative_count += 1

        details = []
        if negative_count > len(news_items) * 0.3:
            score = 3
            details.append(f"High proportion of negative headlines: {negative_count}/{len(news_items)}")
        elif negative_count > 0:
            score = 6
            details.append(f"Some negative headlines: {negative_count}/{len(news_items)}")
        else:
            score = 8
            details.append("Mostly positive/neutral headlines")

        return {"score": score, "details": "; ".join(details)}
