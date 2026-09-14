"""Peter Lynch — ten-baggers in everyday businesses, growth at a reasonable price.

Ported from ``src/agents/peter_lynch.py`` at virattt/ai-hedge-fund v2026.5.14
(MIT).  The five scoring functions (growth, fundamentals, PEG valuation, news
sentiment, insider activity) are kept rule-for-rule; the LangGraph state,
progress bar and LLM call are gone.  Upstream scored each part 0-10 and
blended them 30/25/20/15/10 into a 10-point total, then decided the signal in
code (>= 7.5 bullish, <= 4.5 bearish).  Here each part carries its weighted
score (max 3 / 2.5 / 2 / 1.5 / 1, still summing to 10) with the raw 0-10
score kept in ``part.data``, so ``ev.ratio`` equals the upstream total / 10
and :meth:`PeterLynch.decide` applies the same 0.75 / 0.45 cuts.  Upstream
left confidence to the LLM; it is now derived from distance to those cuts.
"""

from __future__ import annotations

from typing import Any

from v2.personas.base import Persona, classic_verdict, confidence_from_ratio
from v2.personas.models import Evaluation, Signal
from v2.personas.snapshot import PersonaSnapshot

BULLISH_AT = 0.75  # upstream: total_score >= 7.5 of 10
BEARISH_AT = 0.45  # upstream: total_score <= 4.5 of 10

#: upstream weights — 30% growth, 25% valuation, 20% fundamentals, 15% sentiment, 10% insiders
WEIGHTS = {"growth": 0.30, "valuation": 0.25, "fundamentals": 0.20, "sentiment": 0.15, "insider_activity": 0.10}


class PeterLynch(Persona):
    key = "peter_lynch"
    name = "Peter Lynch"
    name_zh = "彼得·林奇"
    style = "seeks ten-baggers in everyday businesses"
    period = "annual"
    lookback = 5
    needs = frozenset({"insiders", "news"})
    system_prompt = (
        "You are Peter Lynch. Explain a bullish, bearish, or neutral verdict using only the provided facts.\n"
        "Principles: invest in what you know (understandable, everyday businesses); growth at a reasonable price — "
        "the PEG ratio is the prime metric (< 1 very attractive, 1-2 fair, > 2 expensive); look for ten-baggers; "
        "prefer steady revenue and EPS growth over short-term noise; avoid dangerous leverage; a good story, but "
        "not overhyped or too complex; news sentiment and insider buying are secondary inputs.\n"
        "Voice: cite the PEG, mention ten-bagger potential if it applies, use practical, folksy, anecdotal language, "
        "list key positives and negatives, end with a clear stance.\n"
        "Signal rules: weighted score (30% growth, 25% valuation, 20% fundamentals, 15% sentiment, 10% insiders) "
        ">= 7.5/10 = bullish; <= 4.5/10 = bearish; otherwise neutral."
    )

    # ------------------------------------------------------------------ rules

    def evaluate(self, snap: PersonaSnapshot) -> Evaluation:
        items = snap.line_items(self.period, self.lookback)
        market_cap = snap.market_cap
        insider_trades = snap.insider_trades[:50]
        news = snap.news[:50]

        growth = self.analyze_lynch_growth(items)
        fundamentals = self.analyze_lynch_fundamentals(items)
        valuation = self.analyze_lynch_valuation(items, market_cap)
        sentiment = self.analyze_sentiment(news)
        insiders = self.analyze_insider_activity(insider_trades)

        def weighted(name: str, result: dict[str, Any]):
            w = WEIGHTS[name]
            return self.part(name, result["score"] * w, 10 * w, result["details"], raw_score=result["score"], raw_max=10, weight=w)

        ev = Evaluation(parts=[
            weighted("growth", growth),
            weighted("valuation", valuation),
            weighted("fundamentals", fundamentals),
            weighted("sentiment", sentiment),
            weighted("insider_activity", insiders),
        ])
        ev.facts = {
            "market_cap": market_cap,
            "pe_ratio": valuation.get("pe_ratio"),
            "eps_growth_rate": valuation.get("eps_growth_rate"),
            "peg_ratio": valuation.get("peg_ratio"),
            "weighted_score": ev.score,
            "news_count": len(news),
            "insider_trade_count": len(insider_trades),
        }
        if not items:
            ev.insufficient = True
            ev.notes.append("no fundamentals")
        return ev

    def decide(self, ev: Evaluation) -> tuple[Signal, int]:
        # Upstream: total_score >= 7.5 -> bullish, <= 4.5 -> bearish, else neutral (max 10).
        signal = classic_verdict(ev.ratio, bullish_at=BULLISH_AT, bearish_at=BEARISH_AT)
        confidence = confidence_from_ratio(ev.ratio, signal, bullish_at=BULLISH_AT, bearish_at=BEARISH_AT)
        return signal, confidence

    # ---------------------------------------------------- upstream functions

    @staticmethod
    def analyze_lynch_growth(items: list) -> dict[str, Any]:
        if not items or len(items) < 2:
            return {"score": 0, "details": "Insufficient financial data for growth analysis"}
        details = []
        raw_score = 0

        revenues = [i.revenue for i in items if i.revenue is not None]
        if len(revenues) >= 2:
            latest_rev, older_rev = revenues[0], revenues[-1]
            if older_rev > 0:
                rev_growth = (latest_rev - older_rev) / abs(older_rev)
                if rev_growth > 0.25:
                    raw_score += 3
                    details.append(f"Strong revenue growth: {rev_growth:.1%}")
                elif rev_growth > 0.10:
                    raw_score += 2
                    details.append(f"Moderate revenue growth: {rev_growth:.1%}")
                elif rev_growth > 0.02:
                    raw_score += 1
                    details.append(f"Slight revenue growth: {rev_growth:.1%}")
                else:
                    details.append(f"Flat or negative revenue growth: {rev_growth:.1%}")
            else:
                details.append("Older revenue is zero/negative; can't compute revenue growth.")
        else:
            details.append("Not enough revenue data to assess growth.")

        eps_values = [i.earnings_per_share for i in items if i.earnings_per_share is not None]
        if len(eps_values) >= 2:
            latest_eps, older_eps = eps_values[0], eps_values[-1]
            if abs(older_eps) > 1e-9:
                eps_growth = (latest_eps - older_eps) / abs(older_eps)
                if eps_growth > 0.25:
                    raw_score += 3
                    details.append(f"Strong EPS growth: {eps_growth:.1%}")
                elif eps_growth > 0.10:
                    raw_score += 2
                    details.append(f"Moderate EPS growth: {eps_growth:.1%}")
                elif eps_growth > 0.02:
                    raw_score += 1
                    details.append(f"Slight EPS growth: {eps_growth:.1%}")
                else:
                    details.append(f"Minimal or negative EPS growth: {eps_growth:.1%}")
            else:
                details.append("Older EPS is near zero; skipping EPS growth calculation.")
        else:
            details.append("Not enough EPS data for growth calculation.")

        final_score = min(10, (raw_score / 6) * 10)
        return {"score": final_score, "details": "; ".join(details)}

    @staticmethod
    def analyze_lynch_fundamentals(items: list) -> dict[str, Any]:
        if not items:
            return {"score": 0, "details": "Insufficient fundamentals data"}
        details = []
        raw_score = 0

        debt_values = [i.total_debt for i in items if i.total_debt is not None]
        eq_values = [i.shareholders_equity for i in items if i.shareholders_equity is not None]
        if debt_values and eq_values and len(debt_values) == len(eq_values) and len(debt_values) > 0:
            recent_debt = debt_values[0]
            recent_equity = eq_values[0] if eq_values[0] else 1e-9
            de_ratio = recent_debt / recent_equity
            if de_ratio < 0.5:
                raw_score += 2
                details.append(f"Low debt-to-equity: {de_ratio:.2f}")
            elif de_ratio < 1.0:
                raw_score += 1
                details.append(f"Moderate debt-to-equity: {de_ratio:.2f}")
            else:
                details.append(f"High debt-to-equity: {de_ratio:.2f}")
        else:
            details.append("No consistent debt/equity data available.")

        om_values = [i.operating_margin for i in items if i.operating_margin is not None]
        if om_values:
            om_recent = om_values[0]
            if om_recent > 0.20:
                raw_score += 2
                details.append(f"Strong operating margin: {om_recent:.1%}")
            elif om_recent > 0.10:
                raw_score += 1
                details.append(f"Moderate operating margin: {om_recent:.1%}")
            else:
                details.append(f"Low operating margin: {om_recent:.1%}")
        else:
            details.append("No operating margin data available.")

        fcf_values = [i.free_cash_flow for i in items if i.free_cash_flow is not None]
        if fcf_values and fcf_values[0] is not None:
            if fcf_values[0] > 0:
                raw_score += 2
                details.append(f"Positive free cash flow: {fcf_values[0]:,.0f}")
            else:
                details.append(f"Recent FCF is negative: {fcf_values[0]:,.0f}")
        else:
            details.append("No free cash flow data available.")

        final_score = min(10, (raw_score / 6) * 10)
        return {"score": final_score, "details": "; ".join(details)}

    @staticmethod
    def analyze_lynch_valuation(items: list, market_cap: float | None) -> dict[str, Any]:
        if not items or market_cap is None:
            return {"score": 0, "details": "Insufficient data for valuation"}
        details = []
        raw_score = 0

        net_incomes = [i.net_income for i in items if i.net_income is not None]
        eps_values = [i.earnings_per_share for i in items if i.earnings_per_share is not None]

        pe_ratio = None
        if net_incomes and net_incomes[0] and net_incomes[0] > 0:
            pe_ratio = market_cap / net_incomes[0]
            details.append(f"Estimated P/E: {pe_ratio:.2f}")
        else:
            details.append("No positive net income => can't compute approximate P/E")

        eps_growth_rate = None
        if len(eps_values) >= 2:
            latest_eps, older_eps = eps_values[0], eps_values[-1]
            if older_eps > 0:
                num_years = len(eps_values) - 1
                if latest_eps > 0:
                    eps_growth_rate = (latest_eps / older_eps) ** (1 / num_years) - 1
                else:
                    eps_growth_rate = (latest_eps - older_eps) / (older_eps * num_years)
                details.append(f"Annualized EPS growth rate: {eps_growth_rate:.1%}")
            else:
                details.append("Cannot compute EPS growth rate (older EPS <= 0)")
        else:
            details.append("Not enough EPS data to compute growth rate")

        peg_ratio = None
        if pe_ratio and eps_growth_rate and eps_growth_rate > 0:
            peg_ratio = pe_ratio / (eps_growth_rate * 100)
            details.append(f"PEG ratio: {peg_ratio:.2f}")

        if pe_ratio is not None:
            if pe_ratio < 15:
                raw_score += 2
            elif pe_ratio < 25:
                raw_score += 1
        if peg_ratio is not None:
            if peg_ratio < 1:
                raw_score += 3
            elif peg_ratio < 2:
                raw_score += 2
            elif peg_ratio < 3:
                raw_score += 1

        final_score = min(10, (raw_score / 5) * 10)
        return {"score": final_score, "details": "; ".join(details), "pe_ratio": pe_ratio, "eps_growth_rate": eps_growth_rate, "peg_ratio": peg_ratio}

    @staticmethod
    def analyze_sentiment(news_items: list) -> dict[str, Any]:
        if not news_items:
            return {"score": 5, "details": "No news data; default to neutral sentiment"}
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
            details.append("Mostly positive or neutral headlines")
        return {"score": score, "details": "; ".join(details)}

    @staticmethod
    def analyze_insider_activity(insider_trades: list) -> dict[str, Any]:
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
            details.append("No significant buy/sell transactions found; neutral stance")
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
