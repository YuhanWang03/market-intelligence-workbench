"""Stanley Druckenmiller — asymmetric risk-reward, growth, momentum and sentiment.

Ported from ``src/agents/stanley_druckenmiller.py`` at virattt/ai-hedge-fund
v2026.5.14 (MIT).  The five scoring functions are kept rule-for-rule, each
scoring 0-10 exactly as upstream; the LangGraph state, progress bar and LLM
call are gone.  Upstream blended the five with fixed weights (growth/momentum
35 %, risk-reward 20 %, valuation 20 %, sentiment 15 %, insiders 10 %) into a
0-10 total and cut at 7.5 / 4.5 — here every part's ``max_score`` is
``10 * weight`` so the weighted blend *is* the sum of the parts, and
:meth:`StanleyDruckenmiller.decide` applies the same two cuts.  Upstream also
fetched annual metrics it never read; they are not fetched here.  Prices come
from the snapshot (ascending by ``time``) instead of a fresh ``get_prices``.
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


def _sorted_closes(prices: list) -> list[float]:
    """Closes in ascending time order, nulls skipped (upstream sorted by ``p.time``)."""
    ordered = sorted(prices, key=lambda p: str(p.time or ""))
    return [p.close for p in ordered if p.close is not None]


class StanleyDruckenmiller(Persona):
    key = "stanley_druckenmiller"
    name = "Stanley Druckenmiller"
    name_zh = "斯坦利·德鲁肯米勒"
    style = "hunts for asymmetric opportunities with growth potential"
    period = "annual"
    lookback = 5
    needs = frozenset({"insiders", "news", "prices"})
    requires = frozenset({"metrics", "line_items", "prices"})
    system_prompt = (
        "You are Stanley Druckenmiller. Use only the provided facts.\n"
        "Principles: 1) seek asymmetric risk-reward (large upside, limited downside); 2) emphasize growth, "
        "momentum and market sentiment; 3) preserve capital by avoiding major drawdowns; 4) pay up for true "
        "growth leaders; 5) be aggressive when conviction is high; 6) cut losses fast if the thesis changes.\n"
        "Rules: reward strong revenue/EPS growth and positive price momentum; read sentiment and insider "
        "activity as supporting or contradicting evidence; watch for high leverage or extreme volatility.\n"
        "Explain the growth/momentum numbers behind the call, risk-reward with figures, and valuation vs "
        "growth, in a decisive, conviction-driven voice.\n"
        "Signal rules: weighted score (growth/momentum 35%, risk-reward 20%, valuation 20%, sentiment 15%, "
        "insiders 10%) is bullish at >= 7.5/10, bearish at <= 4.5/10, neutral between. Confidence 0-100."
    )

    #: upstream weights, in evaluation order
    WEIGHTS = {
        "growth_momentum": 0.35,
        "risk_reward": 0.20,
        "valuation": 0.20,
        "sentiment": 0.15,
        "insider_activity": 0.10,
    }

    # ------------------------------------------------------------------ rules

    def evaluate(self, snap: PersonaSnapshot) -> Evaluation:
        items = snap.line_items(self.period, self.lookback)
        market_cap = snap.market_cap
        insider_trades = list(snap.insider_trades)[:_INSIDER_LIMIT]
        news = list(snap.news)[:_NEWS_LIMIT]
        prices = list(snap.prices)

        results = {
            "growth_momentum": self.analyze_growth_and_momentum(items, prices),
            "risk_reward": self.analyze_risk_reward(items, prices),
            "valuation": self.analyze_druckenmiller_valuation(items, market_cap),
            "sentiment": self.analyze_sentiment(news),
            "insider_activity": self.analyze_insider_activity(insider_trades),
        }

        parts = []
        for name, weight in self.WEIGHTS.items():
            r = results[name]
            parts.append(self.part(name, r["score"] * weight, 10 * weight, r["details"], raw_score=r["score"], weight=weight))
        ev = Evaluation(parts=parts)

        closes = _sorted_closes(prices)
        momentum = None
        if len(closes) >= 2 and closes[0] > 0:
            momentum = (closes[-1] - closes[0]) / closes[0]
        ev.facts = {
            "market_cap": market_cap,
            "enterprise_value": results["valuation"].get("enterprise_value"),
            "price_momentum": momentum,
            "price_points": len(closes),
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
    def analyze_growth_and_momentum(financial_line_items: list, prices: list) -> dict[str, Any]:
        """Revenue CAGR, EPS CAGR and price momentum, 3 points each; 9 raw points scaled to 0-10."""
        if not financial_line_items or len(financial_line_items) < 2:
            return {"score": 0, "details": "Insufficient financial data for growth analysis"}

        details = []
        raw_score = 0

        revenues = [fi.revenue for fi in financial_line_items if fi.revenue is not None]
        if len(revenues) >= 2:
            latest_rev = revenues[0]
            older_rev = revenues[-1]
            num_years = len(revenues) - 1
            if older_rev > 0 and latest_rev > 0:
                rev_growth = (latest_rev / older_rev) ** (1 / num_years) - 1
                if rev_growth > 0.08:
                    raw_score += 3
                    details.append(f"Strong annualized revenue growth: {rev_growth:.1%}")
                elif rev_growth > 0.04:
                    raw_score += 2
                    details.append(f"Moderate annualized revenue growth: {rev_growth:.1%}")
                elif rev_growth > 0.01:
                    raw_score += 1
                    details.append(f"Slight annualized revenue growth: {rev_growth:.1%}")
                else:
                    details.append(f"Minimal/negative revenue growth: {rev_growth:.1%}")
            else:
                details.append("Older revenue is zero/negative; can't compute revenue growth.")
        else:
            details.append("Not enough revenue data points for growth calculation.")

        eps_values = [fi.earnings_per_share for fi in financial_line_items if fi.earnings_per_share is not None]
        if len(eps_values) >= 2:
            latest_eps = eps_values[0]
            older_eps = eps_values[-1]
            num_years = len(eps_values) - 1
            if older_eps > 0 and latest_eps > 0:
                eps_growth = (latest_eps / older_eps) ** (1 / num_years) - 1
                if eps_growth > 0.08:
                    raw_score += 3
                    details.append(f"Strong annualized EPS growth: {eps_growth:.1%}")
                elif eps_growth > 0.04:
                    raw_score += 2
                    details.append(f"Moderate annualized EPS growth: {eps_growth:.1%}")
                elif eps_growth > 0.01:
                    raw_score += 1
                    details.append(f"Slight annualized EPS growth: {eps_growth:.1%}")
                else:
                    details.append(f"Minimal/negative annualized EPS growth: {eps_growth:.1%}")
            else:
                details.append("Older EPS is near zero; skipping EPS growth calculation.")
        else:
            details.append("Not enough EPS data points for growth calculation.")

        if prices and len(prices) > 30:
            close_prices = _sorted_closes(prices)
            if len(close_prices) >= 2:
                start_price = close_prices[0]
                end_price = close_prices[-1]
                if start_price > 0:
                    pct_change = (end_price - start_price) / start_price
                    if pct_change > 0.50:
                        raw_score += 3
                        details.append(f"Very strong price momentum: {pct_change:.1%}")
                    elif pct_change > 0.20:
                        raw_score += 2
                        details.append(f"Moderate price momentum: {pct_change:.1%}")
                    elif pct_change > 0:
                        raw_score += 1
                        details.append(f"Slight positive momentum: {pct_change:.1%}")
                    else:
                        details.append(f"Negative price momentum: {pct_change:.1%}")
                else:
                    details.append("Invalid start price (<= 0); can't compute momentum.")
            else:
                details.append("Insufficient price data for momentum calculation.")
        else:
            details.append("Not enough recent price data for momentum analysis.")

        final_score = min(10, (raw_score / 9) * 10)
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

    @staticmethod
    def analyze_risk_reward(financial_line_items: list, prices: list) -> dict[str, Any]:
        """Debt-to-equity and daily-return volatility, 3 points each; 6 raw points scaled to 0-10."""
        if not financial_line_items or not prices:
            return {"score": 0, "details": "Insufficient data for risk-reward analysis"}

        details = []
        raw_score = 0

        debt_values = [fi.total_debt for fi in financial_line_items if fi.total_debt is not None]
        equity_values = [fi.shareholders_equity for fi in financial_line_items if fi.shareholders_equity is not None]

        if debt_values and equity_values and len(debt_values) == len(equity_values) and len(debt_values) > 0:
            recent_debt = debt_values[0]
            recent_equity = equity_values[0] if equity_values[0] else 1e-9
            de_ratio = recent_debt / recent_equity
            if de_ratio < 0.3:
                raw_score += 3
                details.append(f"Low debt-to-equity: {de_ratio:.2f}")
            elif de_ratio < 0.7:
                raw_score += 2
                details.append(f"Moderate debt-to-equity: {de_ratio:.2f}")
            elif de_ratio < 1.5:
                raw_score += 1
                details.append(f"Somewhat high debt-to-equity: {de_ratio:.2f}")
            else:
                details.append(f"High debt-to-equity: {de_ratio:.2f}")
        else:
            details.append("No consistent debt/equity data available.")

        if len(prices) > 10:
            close_prices = _sorted_closes(prices)
            if len(close_prices) > 10:
                daily_returns = []
                for i in range(1, len(close_prices)):
                    prev_close = close_prices[i - 1]
                    if prev_close > 0:
                        daily_returns.append((close_prices[i] - prev_close) / prev_close)
                if daily_returns:
                    stdev = statistics.pstdev(daily_returns)
                    if stdev < 0.01:
                        raw_score += 3
                        details.append(f"Low volatility: daily returns stdev {stdev:.2%}")
                    elif stdev < 0.02:
                        raw_score += 2
                        details.append(f"Moderate volatility: daily returns stdev {stdev:.2%}")
                    elif stdev < 0.04:
                        raw_score += 1
                        details.append(f"High volatility: daily returns stdev {stdev:.2%}")
                    else:
                        details.append(f"Very high volatility: daily returns stdev {stdev:.2%}")
                else:
                    details.append("Insufficient daily returns data for volatility calc.")
            else:
                details.append("Not enough close-price data points for volatility analysis.")
        else:
            details.append("Not enough price data for volatility analysis.")

        final_score = min(10, (raw_score / 6) * 10)
        return {"score": final_score, "details": "; ".join(details)}

    @staticmethod
    def analyze_druckenmiller_valuation(financial_line_items: list, market_cap: float | None) -> dict[str, Any]:
        """P/E, P/FCF, EV/EBIT and EV/EBITDA, 2 points each; 8 raw points scaled to 0-10."""
        if not financial_line_items or market_cap is None:
            return {"score": 0, "details": "Insufficient data to perform valuation"}

        details = []
        raw_score = 0

        net_incomes = [fi.net_income for fi in financial_line_items if fi.net_income is not None]
        fcf_values = [fi.free_cash_flow for fi in financial_line_items if fi.free_cash_flow is not None]
        ebit_values = [fi.ebit for fi in financial_line_items if fi.ebit is not None]
        ebitda_values = [fi.ebitda for fi in financial_line_items if fi.ebitda is not None]

        debt_values = [fi.total_debt for fi in financial_line_items if fi.total_debt is not None]
        cash_values = [fi.cash_and_equivalents for fi in financial_line_items if fi.cash_and_equivalents is not None]
        recent_debt = debt_values[0] if debt_values else 0
        recent_cash = cash_values[0] if cash_values else 0

        enterprise_value = market_cap + recent_debt - recent_cash

        recent_net_income = net_incomes[0] if net_incomes else None
        if recent_net_income and recent_net_income > 0:
            pe = market_cap / recent_net_income
            pe_points = 0
            if pe < 15:
                pe_points = 2
                details.append(f"Attractive P/E: {pe:.2f}")
            elif pe < 25:
                pe_points = 1
                details.append(f"Fair P/E: {pe:.2f}")
            else:
                details.append(f"High or Very high P/E: {pe:.2f}")
            raw_score += pe_points
        else:
            details.append("No positive net income for P/E calculation")

        recent_fcf = fcf_values[0] if fcf_values else None
        if recent_fcf and recent_fcf > 0:
            pfcf = market_cap / recent_fcf
            pfcf_points = 0
            if pfcf < 15:
                pfcf_points = 2
                details.append(f"Attractive P/FCF: {pfcf:.2f}")
            elif pfcf < 25:
                pfcf_points = 1
                details.append(f"Fair P/FCF: {pfcf:.2f}")
            else:
                details.append(f"High/Very high P/FCF: {pfcf:.2f}")
            raw_score += pfcf_points
        else:
            details.append("No positive free cash flow for P/FCF calculation")

        recent_ebit = ebit_values[0] if ebit_values else None
        if enterprise_value > 0 and recent_ebit and recent_ebit > 0:
            ev_ebit = enterprise_value / recent_ebit
            ev_ebit_points = 0
            if ev_ebit < 15:
                ev_ebit_points = 2
                details.append(f"Attractive EV/EBIT: {ev_ebit:.2f}")
            elif ev_ebit < 25:
                ev_ebit_points = 1
                details.append(f"Fair EV/EBIT: {ev_ebit:.2f}")
            else:
                details.append(f"High EV/EBIT: {ev_ebit:.2f}")
            raw_score += ev_ebit_points
        else:
            details.append("No valid EV/EBIT because EV <= 0 or EBIT <= 0")

        recent_ebitda = ebitda_values[0] if ebitda_values else None
        if enterprise_value > 0 and recent_ebitda and recent_ebitda > 0:
            ev_ebitda = enterprise_value / recent_ebitda
            ev_ebitda_points = 0
            if ev_ebitda < 10:
                ev_ebitda_points = 2
                details.append(f"Attractive EV/EBITDA: {ev_ebitda:.2f}")
            elif ev_ebitda < 18:
                ev_ebitda_points = 1
                details.append(f"Fair EV/EBITDA: {ev_ebitda:.2f}")
            else:
                details.append(f"High EV/EBITDA: {ev_ebitda:.2f}")
            raw_score += ev_ebitda_points
        else:
            details.append("No valid EV/EBITDA because EV <= 0 or EBITDA <= 0")

        final_score = min(10, (raw_score / 8) * 10)
        return {"score": final_score, "details": "; ".join(details), "enterprise_value": enterprise_value}
