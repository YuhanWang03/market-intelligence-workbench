"""Benjamin Graham — hidden gems bought with a margin of safety.

Ported from ``src/agents/ben_graham.py`` at virattt/ai-hedge-fund
v2026.5.14 (MIT).  The three scoring functions (earnings stability,
financial strength, Graham valuation) are kept rule-for-rule; the
LangGraph state, progress bar and LLM call are gone.  Upstream drew its
pre-signal at 70% / 30% of a declared 15-point maximum, which is exactly
the base :func:`~v2.personas.base.classic_verdict`, so ``decide`` is
inherited; the Graham-Number margin of safety feeds the shared valuation
gate.  Note upstream's valuation rules can award 7 points against a
declared budget of 15 (4 + 5 + 6): the part max is kept at 6 so the total
matches upstream, and a deep net-net can score slightly above it.
"""

from __future__ import annotations

import math
from typing import Any

from v2.personas.base import Persona
from v2.personas.models import Evaluation
from v2.personas.snapshot import PersonaSnapshot


class BenGraham(Persona):
    key = "ben_graham"
    name = "Ben Graham"
    name_zh = "本杰明·格雷厄姆"
    style = "only buys hidden gems with a margin of safety"
    period = "annual"
    lookback = 10
    needs = frozenset()
    system_prompt = (
        "You are Benjamin Graham. Decide bullish, bearish, or neutral using only the provided facts, in a "
        "conservative, analytical voice.\n"
        "Principles: insist on a margin of safety below intrinsic value (Graham Number, net-net); demand "
        "financial strength (current ratio >= 2.0, liabilities < 50% of assets); prefer stable, consistently "
        "positive earnings; count a dividend record as extra safety; avoid speculative growth assumptions.\n"
        "Cite the numbers that moved the verdict (Graham Number vs price, NCAV vs market cap, current ratio, "
        "debt ratio, EPS record) and compare them to Graham's thresholds.\n"
        "Signal rules: bullish >= 70% of the 15-point score AND price below the Graham Number; bearish <= 30% "
        "or no margin of safety with weak strength; neutral otherwise. Confidence 0-100."
    )

    # ------------------------------------------------------------------ rules

    def evaluate(self, snap: PersonaSnapshot) -> Evaluation:
        metrics = snap.metrics(self.period, self.lookback)
        items = snap.line_items(self.period, self.lookback)
        market_cap = snap.market_cap

        earnings = self.analyze_earnings_stability(metrics, items)
        strength = self.analyze_financial_strength(items)
        valuation = self.analyze_valuation_graham(items, market_cap)

        ev = Evaluation(parts=[
            self.part("earnings_stability", earnings["score"], 4, earnings["details"]),
            self.part("financial_strength", strength["score"], 5, strength["details"]),
            self.part("valuation", valuation["score"], 6, valuation["details"]),
        ])
        ev.margin_of_safety = valuation.get("margin_of_safety")
        ev.facts = {
            "market_cap": market_cap,
            "net_current_asset_value": valuation.get("net_current_asset_value"),
            "ncav_per_share": valuation.get("ncav_per_share"),
            "price_per_share": valuation.get("price_per_share"),
            "graham_number": valuation.get("graham_number"),
            "margin_of_safety": ev.margin_of_safety,
        }
        if not metrics and not items:
            ev.insufficient = True
            ev.notes.append("no fundamentals")
        return ev

    # ---------------------------------------------------- upstream functions

    @staticmethod
    def analyze_earnings_stability(metrics: list, items: list) -> dict[str, Any]:
        score = 0
        details = []
        if not metrics or not items:
            return {"score": score, "details": "Insufficient data for earnings stability analysis"}

        eps_vals = [i.earnings_per_share for i in items if i.earnings_per_share is not None]
        if len(eps_vals) < 2:
            details.append("Not enough multi-year EPS data.")
            return {"score": score, "details": "; ".join(details)}

        # 1. Consistently positive EPS
        positive = sum(1 for e in eps_vals if e > 0)
        total = len(eps_vals)
        if positive == total:
            score += 3
            details.append("EPS was positive in all available periods.")
        elif positive >= total * 0.8:
            score += 2
            details.append("EPS was positive in most periods.")
        else:
            details.append("EPS was negative in multiple periods.")

        # 2. EPS growth from earliest to latest
        if eps_vals[0] > eps_vals[-1]:
            score += 1
            details.append("EPS grew from earliest to latest period.")
        else:
            details.append("EPS did not grow from earliest to latest period.")

        return {"score": score, "details": "; ".join(details)}

    @staticmethod
    def analyze_financial_strength(items: list) -> dict[str, Any]:
        score = 0
        details = []
        if not items:
            return {"score": score, "details": "No data for financial strength analysis"}

        latest = items[0]
        total_assets = latest.total_assets or 0
        total_liabilities = latest.total_liabilities or 0
        current_assets = latest.current_assets or 0
        current_liabilities = latest.current_liabilities or 0

        # 1. Current ratio
        if current_liabilities > 0:
            current_ratio = current_assets / current_liabilities
            if current_ratio >= 2.0:
                score += 2
                details.append(f"Current ratio = {current_ratio:.2f} (>=2.0: solid).")
            elif current_ratio >= 1.5:
                score += 1
                details.append(f"Current ratio = {current_ratio:.2f} (moderately strong).")
            else:
                details.append(f"Current ratio = {current_ratio:.2f} (<1.5: weaker liquidity).")
        else:
            details.append("Cannot compute current ratio (missing or zero current_liabilities).")

        # 2. Debt vs. assets
        if total_assets > 0:
            debt_ratio = total_liabilities / total_assets
            if debt_ratio < 0.5:
                score += 2
                details.append(f"Debt ratio = {debt_ratio:.2f}, under 0.50 (conservative).")
            elif debt_ratio < 0.8:
                score += 1
                details.append(f"Debt ratio = {debt_ratio:.2f}, somewhat high but could be acceptable.")
            else:
                details.append(f"Debt ratio = {debt_ratio:.2f}, quite high by Graham standards.")
        else:
            details.append("Cannot compute debt ratio (missing total_assets).")

        # 3. Dividend track record (outflows are negative in most feeds)
        div_periods = [i.dividends_and_other_cash_distributions for i in items if i.dividends_and_other_cash_distributions is not None]
        if div_periods:
            paid = sum(1 for d in div_periods if d < 0)
            if paid > 0:
                if paid >= (len(div_periods) // 2 + 1):
                    score += 1
                    details.append("Company paid dividends in the majority of the reported years.")
                else:
                    details.append("Company has some dividend payments, but not most years.")
            else:
                details.append("Company did not pay dividends in these periods.")
        else:
            details.append("No dividend data available to assess payout consistency.")

        return {"score": score, "details": "; ".join(details)}

    @staticmethod
    def analyze_valuation_graham(items: list, market_cap: float | None) -> dict[str, Any]:
        if not items or not market_cap or market_cap <= 0:
            return {"score": 0, "details": "Insufficient data to perform valuation"}

        latest = items[0]
        current_assets = latest.current_assets or 0
        total_liabilities = latest.total_liabilities or 0
        book_value_ps = latest.book_value_per_share or 0
        eps = latest.earnings_per_share or 0
        shares = latest.outstanding_shares or 0

        details = []
        score = 0
        out: dict[str, Any] = {
            "net_current_asset_value": None, "ncav_per_share": None, "price_per_share": None,
            "graham_number": None, "margin_of_safety": None,
        }
        if shares > 0:
            out["price_per_share"] = market_cap / shares

        # 1. Net-net check: NCAV = current assets - total liabilities
        ncav = current_assets - total_liabilities
        out["net_current_asset_value"] = ncav
        if ncav > 0 and shares > 0:
            ncav_ps = ncav / shares
            price_ps = market_cap / shares if shares else 0
            out["ncav_per_share"] = ncav_ps
            details.append(f"Net Current Asset Value = {ncav:,.2f}")
            details.append(f"NCAV Per Share = {ncav_ps:,.2f}")
            details.append(f"Price Per Share = {price_ps:,.2f}")
            if ncav > market_cap:
                score += 4  # classic Graham deep value
                details.append("Net-Net: NCAV > Market Cap (classic Graham deep value).")
            elif ncav_ps >= (price_ps * 0.67):
                score += 2
                details.append("NCAV Per Share >= 2/3 of Price Per Share (moderate net-net discount).")
        else:
            details.append("NCAV not exceeding market cap or insufficient data for net-net approach.")

        # 2. Graham Number = sqrt(22.5 * EPS * BVPS)
        graham_number = None
        if eps > 0 and book_value_ps > 0:
            graham_number = math.sqrt(22.5 * eps * book_value_ps)
            out["graham_number"] = graham_number
            details.append(f"Graham Number = {graham_number:.2f}")
        else:
            details.append("Unable to compute Graham Number (EPS or Book Value missing/<=0).")

        # 3. Margin of safety relative to the Graham Number
        if graham_number and shares > 0:
            current_price = market_cap / shares
            if current_price > 0:
                mos = (graham_number - current_price) / current_price
                out["margin_of_safety"] = mos
                details.append(f"Margin of Safety (Graham Number) = {mos:.2%}")
                if mos > 0.5:
                    score += 3
                    details.append("Price is well below Graham Number (>=50% margin).")
                elif mos > 0.2:
                    score += 1
                    details.append("Some margin of safety relative to Graham Number.")
                else:
                    details.append("Price close to or above Graham Number, low margin of safety.")
            else:
                details.append("Current price is zero or invalid; can't compute margin of safety.")

        out.update({"score": score, "details": "; ".join(details)})
        return out
