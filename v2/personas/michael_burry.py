"""Michael Burry — deep value, contrarian, downside first.

Ported from ``src/agents/michael_burry.py`` at virattt/ai-hedge-fund
v2026.5.14 (MIT).  The four scoring functions (value, balance sheet,
insider activity, contrarian sentiment) are kept rule-for-rule; the
LangGraph state, progress bar and LLM call are gone.  Upstream already
derived a preliminary signal in code — bullish at >= 70% of the max score,
bearish at <= 30% — and that rule is :meth:`MichaelBurry.decide` verbatim;
the LLM only supplied a confidence, which is now the shared deterministic
mapping from the same ratio.
"""

from __future__ import annotations

from typing import Any

from v2.personas.base import Persona, classic_verdict, confidence_from_ratio
from v2.personas.models import Evaluation, Signal
from v2.personas.snapshot import PersonaSnapshot


class MichaelBurry(Persona):
    key = "michael_burry"
    name = "Michael Burry"
    name_zh = "迈克尔·伯里"
    style = "hunts for deep value"
    period = "ttm"
    lookback = 5
    #: upstream fetched line items with the API default (``limit=10``)
    line_item_lookback = 10
    needs = frozenset({"insiders", "news"})
    system_prompt = (
        "You are an AI agent emulating Dr. Michael J. Burry. Mandate: hunt for deep value using hard numbers "
        "(free cash flow yield, EV/EBIT, balance sheet); be contrarian — hatred in the press is your friend if "
        "fundamentals are solid; downside first — avoid leveraged balance sheets; look for hard catalysts such "
        "as insider buying, buybacks or asset sales.\n"
        "Reasoning: lead with the key metric(s); cite concrete numbers (\"FCF yield 14.7%\", \"EV/EBIT 5.3\"); "
        "state risk factors and whether they are acceptable; note insider activity or contrarian set-ups; "
        "terse, data-driven, minimal words.\n"
        "Signal rules: bullish = score >= 70% of max; bearish = score <= 30%; neutral otherwise. Confidence 0-100.\n"
        "Bullish example: \"FCF yield 12.8%. EV/EBIT 6.2. D/E 0.4. Net insider buying 25k shares. Strong buy.\" "
        "Bearish example: \"FCF yield only 2.1%. D/E 2.3. Diluting shareholders. Pass.\""
    )

    # ------------------------------------------------------------------ rules

    def evaluate(self, snap: PersonaSnapshot) -> Evaluation:
        metrics = snap.metrics(self.period, self.lookback)
        items = snap.line_items(self.period, self.line_item_lookback)
        market_cap = snap.market_cap
        insiders = list(snap.insider_trades)
        news = list(snap.news)

        value = self.analyze_value(metrics, items, market_cap)
        balance = self.analyze_balance_sheet(metrics, items)
        insider = self.analyze_insider_activity(insiders)
        contrarian = self.analyze_contrarian_sentiment(news)

        ev = Evaluation(parts=[
            self.part("value", value["score"], value["max_score"], value["details"]),
            self.part("balance_sheet", balance["score"], balance["max_score"], balance["details"]),
            self.part("insider_activity", insider["score"], insider["max_score"], insider["details"]),
            self.part("contrarian_sentiment", contrarian["score"], contrarian["max_score"], contrarian["details"]),
        ])

        latest_item = items[0] if items else None
        latest_metrics = metrics[0] if metrics else None
        fcf = latest_item.free_cash_flow if latest_item else None
        ev.facts = {
            "market_cap": market_cap,
            "free_cash_flow": fcf,
            "fcf_yield": (fcf / market_cap) if (fcf is not None and market_cap) else None,
            "ev_to_ebit": latest_metrics.ev_to_ebit if latest_metrics else None,
            "debt_to_equity": latest_metrics.debt_to_equity if latest_metrics else None,
            "cash_and_equivalents": latest_item.cash_and_equivalents if latest_item else None,
            "total_debt": latest_item.total_debt if latest_item else None,
            "net_insider_shares": insider.get("net_shares"),
            "negative_headlines": contrarian.get("negative_count"),
        }
        if not metrics and not items:
            ev.insufficient = True
            ev.notes.append("no fundamentals")
        return ev

    def decide(self, ev: Evaluation) -> tuple[Signal, int]:
        # Upstream: bullish if total >= 0.7 * max, bearish if total <= 0.3 * max.
        signal = classic_verdict(ev.ratio, bullish_at=0.7, bearish_at=0.3)
        confidence = confidence_from_ratio(ev.ratio, signal, bullish_at=0.7, bearish_at=0.3)
        return signal, confidence

    # ---------------------------------------------------- upstream functions

    @staticmethod
    def _latest_line_item(line_items: list):
        """Return the most recent line-item object or *None*."""
        return line_items[0] if line_items else None

    @classmethod
    def analyze_value(cls, metrics: list, line_items: list, market_cap: float | None) -> dict[str, Any]:
        """Free cash-flow yield, EV/EBIT, other classic deep-value metrics."""
        max_score = 6  # 4 pts for FCF-yield, 2 pts for EV/EBIT
        score = 0
        details: list[str] = []

        latest_item = cls._latest_line_item(line_items)
        fcf = latest_item.free_cash_flow if latest_item else None
        if fcf is not None and market_cap:
            fcf_yield = fcf / market_cap
            if fcf_yield >= 0.15:
                score += 4
                details.append(f"Extraordinary FCF yield {fcf_yield:.1%}")
            elif fcf_yield >= 0.12:
                score += 3
                details.append(f"Very high FCF yield {fcf_yield:.1%}")
            elif fcf_yield >= 0.08:
                score += 2
                details.append(f"Respectable FCF yield {fcf_yield:.1%}")
            else:
                details.append(f"Low FCF yield {fcf_yield:.1%}")
        else:
            details.append("FCF data unavailable")

        if metrics:
            ev_ebit = metrics[0].ev_to_ebit
            if ev_ebit is not None:
                if ev_ebit < 6:
                    score += 2
                    details.append(f"EV/EBIT {ev_ebit:.1f} (<6)")
                elif ev_ebit < 10:
                    score += 1
                    details.append(f"EV/EBIT {ev_ebit:.1f} (<10)")
                else:
                    details.append(f"High EV/EBIT {ev_ebit:.1f}")
            else:
                details.append("EV/EBIT data unavailable")
        else:
            details.append("Financial metrics unavailable")

        return {"score": score, "max_score": max_score, "details": "; ".join(details)}

    @classmethod
    def analyze_balance_sheet(cls, metrics: list, line_items: list) -> dict[str, Any]:
        """Leverage and liquidity checks."""
        max_score = 3
        score = 0
        details: list[str] = []

        latest_metrics = metrics[0] if metrics else None
        latest_item = cls._latest_line_item(line_items)

        debt_to_equity = latest_metrics.debt_to_equity if latest_metrics else None
        if debt_to_equity is not None:
            if debt_to_equity < 0.5:
                score += 2
                details.append(f"Low D/E {debt_to_equity:.2f}")
            elif debt_to_equity < 1:
                score += 1
                details.append(f"Moderate D/E {debt_to_equity:.2f}")
            else:
                details.append(f"High leverage D/E {debt_to_equity:.2f}")
        else:
            details.append("Debt-to-equity data unavailable")

        if latest_item is not None:
            cash = latest_item.cash_and_equivalents
            total_debt = latest_item.total_debt
            if cash is not None and total_debt is not None:
                if cash > total_debt:
                    score += 1
                    details.append("Net cash position")
                else:
                    details.append("Net debt position")
            else:
                details.append("Cash/debt data unavailable")

        return {"score": score, "max_score": max_score, "details": "; ".join(details)}

    @staticmethod
    def analyze_insider_activity(insider_trades: list) -> dict[str, Any]:
        """Net insider buying over the last 12 months acts as a hard catalyst."""
        max_score = 2
        score = 0
        details: list[str] = []

        if not insider_trades:
            details.append("No insider trade data")
            return {"score": score, "max_score": max_score, "details": "; ".join(details), "net_shares": None}

        shares_bought = sum(t.transaction_shares or 0 for t in insider_trades if (t.transaction_shares or 0) > 0)
        shares_sold = abs(sum(t.transaction_shares or 0 for t in insider_trades if (t.transaction_shares or 0) < 0))
        net = shares_bought - shares_sold
        if net > 0:
            score += 2 if net / max(shares_sold, 1) > 1 else 1
            details.append(f"Net insider buying of {net:,} shares")
        else:
            details.append("Net insider selling")

        return {"score": score, "max_score": max_score, "details": "; ".join(details), "net_shares": net}

    @staticmethod
    def analyze_contrarian_sentiment(news: list) -> dict[str, Any]:
        """Very rough gauge: a wall of recent negative headlines can be a *positive* for a contrarian."""
        max_score = 1
        score = 0
        details: list[str] = []

        if not news:
            details.append("No recent news")
            return {"score": score, "max_score": max_score, "details": "; ".join(details), "negative_count": 0}

        sentiment_negative_count = sum(
            1 for n in news if n.sentiment and str(n.sentiment).lower() in ["negative", "bearish"]
        )
        if sentiment_negative_count >= 5:
            score += 1  # The more hated, the better (assuming fundamentals hold up)
            details.append(f"{sentiment_negative_count} negative headlines (contrarian opportunity)")
        else:
            details.append("Limited negative press")

        return {"score": score, "max_score": max_score, "details": "; ".join(details), "negative_count": sentiment_negative_count}
