"""Aswath Damodaran — story, numbers and a disciplined FCFF valuation.

Ported from ``src/agents/aswath_damodaran.py`` at virattt/ai-hedge-fund
v2026.5.14 (MIT).  The growth, risk, relative-valuation, CAPM cost-of-equity
and FCFF DCF functions are kept rule-for-rule (including the DCF's
non-compounding fade and the ``fcff0``-based terminal value); the LangGraph
state, progress bar and LLM call are gone.  Upstream's signal was already
drawn by code from the margin of safety (>= +25 % bullish, <= -25 % bearish),
which is now :meth:`AswathDamodaran.decide`; only the confidence, which the
LLM used to invent, is a fixed function of that margin and the score.

One deliberate change: upstream read ``revenue``, ``free_cash_flow``,
``beta``, ``ebit`` and ``interest_expense`` off its ``FinancialMetrics``
rows, which never carried those fields, so its revenue CAGR, interest
coverage and the entire DCF silently never ran.  Here each of those values
is read from the metrics row first and, when absent, from the matching line
item, so the rules can actually fire.

Deliberate correction shared with the Jhunjhunwala port: revenue CAGR is
annualised over the rows' actual report-date span (:meth:`Persona.cagr`)
instead of ``len(rows) - 1``, which on TTM rows treated quarters as years.
"""

from __future__ import annotations

from typing import Any

from v2.personas.base import Persona, clamp
from v2.personas.models import Evaluation, Signal
from v2.personas.snapshot import PersonaSnapshot

#: upstream: metrics limit=5 (ttm); search_line_items left at its default limit=10
_METRICS_LIMIT = 5
_LINE_ITEMS_LIMIT = 10

#: Damodaran "tends to act with ~20-25 % MOS"
_BULLISH_MOS = 0.25
_BEARISH_MOS = -0.25


def _field(rec: Any, name: str, fallback: Any = None) -> Any:
    """``rec.name`` if present and not null, else the same field on ``fallback``."""
    value = getattr(rec, name, None) if rec is not None else None
    if value is None and fallback is not None:
        value = getattr(fallback, name, None)
    return value


def _merged(metrics: list, line_items: list) -> list:
    """Metrics rows with line-item fields filled in where the metrics lack them.

    Both lists are latest-first; rows are paired by position.  Only the fields
    upstream expected on the metrics row are back-filled.
    """
    from v2.personas.models import Record

    out = []
    for i, m in enumerate(metrics):
        li = line_items[i] if i < len(line_items) else None
        data = m.to_dict() if hasattr(m, "to_dict") else dict(vars(m))
        for name in ("revenue", "free_cash_flow", "ebit", "interest_expense", "beta"):
            if data.get(name) is None and li is not None:
                data[name] = getattr(li, name, None)
        out.append(Record(data))
    return out


class AswathDamodaran(Persona):
    key = "aswath_damodaran"
    name = "Aswath Damodaran"
    name_zh = "阿斯瓦斯·达摩达兰"
    style = "focuses on story, numbers, and disciplined valuation"
    period = "ttm"
    lookback = 5
    needs = frozenset()
    system_prompt = (
        "You are Aswath Damodaran, Professor of Finance at NYU Stern. Issue a signal from the provided "
        "facts only, in your clear, data-driven tone.\n"
        "Structure: start with the company's story (qualitatively); connect it to the numerical drivers: "
        "revenue growth, margins, reinvestment (ROIC vs hurdle), risk (beta, leverage, interest coverage); "
        "conclude with value: the FCFF DCF estimate at the CAPM cost of equity (rf 4% + beta x 5% ERP), "
        "the margin of safety vs market cap, and the relative P/E sanity check; highlight the major "
        "uncertainties and how they move value.\n"
        "Signal rules: margin of safety >= +25% = bullish; <= -25% = bearish; otherwise neutral (including "
        "when no intrinsic value could be computed). Confidence 0-100, higher the further the margin sits "
        "from those cuts."
    )

    # ------------------------------------------------------------------ rules

    def evaluate(self, snap: PersonaSnapshot) -> Evaluation:
        raw_metrics = snap.metrics(self.period, _METRICS_LIMIT)
        line_items = snap.line_items(self.period, _LINE_ITEMS_LIMIT)
        metrics = _merged(raw_metrics, line_items)
        market_cap = snap.market_cap

        growth = self.analyze_growth_and_reinvestment(metrics, line_items)
        risk = self.analyze_risk_profile(metrics, line_items)
        dcf = self.calculate_intrinsic_value_dcf(metrics, line_items, risk)
        relative = self.analyze_relative_valuation(metrics)

        ev = Evaluation(parts=[
            self.part("growth_reinvestment", growth["score"], growth["max_score"], growth["details"]),
            self.part("risk_profile", risk["score"], risk["max_score"], risk["details"], beta=risk.get("beta"), cost_of_equity=risk.get("cost_of_equity")),
            self.part("relative_valuation", relative["score"], relative["max_score"], relative["details"]),
        ])

        intrinsic_value = dcf.get("intrinsic_value")
        if intrinsic_value and market_cap:
            ev.margin_of_safety = (intrinsic_value - market_cap) / market_cap
        ev.facts = {
            "intrinsic_value": intrinsic_value,
            "intrinsic_per_share": dcf.get("intrinsic_per_share"),
            "market_cap": market_cap,
            "margin_of_safety": ev.margin_of_safety,
            "beta": risk.get("beta"),
            "cost_of_equity": risk.get("cost_of_equity"),
            "dcf_assumptions": dcf.get("assumptions"),
            "dcf_details": dcf.get("details"),
        }
        if not raw_metrics and not line_items:
            ev.insufficient = True
            ev.notes.append("no fundamentals")
        return ev

    def decide(self, ev: Evaluation) -> tuple[Signal, int]:
        """Upstream's margin-of-safety rule; confidence grows with distance from the cuts."""
        mos = ev.margin_of_safety
        if mos is not None and mos >= _BULLISH_MOS:
            signal: Signal = "bullish"
            # 55 at +25 %, 95 at +100 % or more.
            confidence = 55 + 40 * min(1.0, (mos - _BULLISH_MOS) / 0.75)
        elif mos is not None and mos <= _BEARISH_MOS:
            signal = "bearish"
            # 55 at -25 %, 95 at -75 % or worse.
            confidence = 55 + 40 * min(1.0, (_BEARISH_MOS - mos) / 0.5)
        else:
            signal = "neutral"
            # 50 at zero margin, tapering to 35 next to either cut; 30 with no DCF at all.
            confidence = 50 - 15 * min(1.0, abs(mos) / _BULLISH_MOS) if mos is not None else 30
        # The checklist (growth, risk, relative value) nudges conviction by up to ±10.
        confidence += 20 * (ev.ratio - 0.5)
        return signal, int(round(clamp(confidence, 10, 95)))

    # ---------------------------------------------------- upstream functions

    @staticmethod
    def analyze_growth_and_reinvestment(metrics: list, line_items: list) -> dict[str, Any]:
        """Growth 0-4: +2 revenue CAGR > 8 % (+1 if > 3 %); +1 rising FCFF; +1 ROIC > 10 %."""
        max_score = 4
        if len(metrics) < 2:
            return {"score": 0, "max_score": max_score, "details": "Insufficient history"}

        growth = Persona.cagr(metrics, "revenue")
        cagr = growth[0] if growth is not None else None

        score, details = 0, []

        if cagr is not None:
            if cagr > 0.08:
                score += 2
                details.append(f"Revenue CAGR {cagr:.1%} (> 8 %)")
            elif cagr > 0.03:
                score += 1
                details.append(f"Revenue CAGR {cagr:.1%} (> 3 %)")
            else:
                details.append(f"Sluggish revenue CAGR {cagr:.1%}")
        else:
            details.append("Revenue data incomplete")

        fcfs = [li.free_cash_flow for li in reversed(line_items) if li.free_cash_flow]
        if len(fcfs) >= 2 and fcfs[-1] > fcfs[0]:
            score += 1
            details.append("Positive FCFF growth")
        else:
            details.append("Flat or declining FCFF")

        latest = metrics[0]
        if latest.return_on_invested_capital and latest.return_on_invested_capital > 0.10:
            score += 1
            details.append(f"ROIC {latest.return_on_invested_capital:.1%} (> 10 %)")

        return {"score": score, "max_score": max_score, "details": "; ".join(details), "metrics": latest.model_dump()}

    @classmethod
    def analyze_risk_profile(cls, metrics: list, line_items: list) -> dict[str, Any]:
        """Risk 0-3: +1 beta < 1.3; +1 D/E < 1; +1 interest coverage > 3x. Also sets cost of equity."""
        max_score = 3
        if not metrics:
            return {"score": 0, "max_score": max_score, "details": "No metrics"}

        latest = metrics[0]
        score, details = 0, []

        beta = getattr(latest, "beta", None)
        if beta is not None:
            if beta < 1.3:
                score += 1
                details.append(f"Beta {beta:.2f}")
            else:
                details.append(f"High beta {beta:.2f}")
        else:
            details.append("Beta NA")

        dte = getattr(latest, "debt_to_equity", None)
        if dte is not None:
            if dte < 1:
                score += 1
                details.append(f"D/E {dte:.1f}")
            else:
                details.append(f"High D/E {dte:.1f}")
        else:
            details.append("D/E NA")

        ebit = getattr(latest, "ebit", None)
        interest = getattr(latest, "interest_expense", None)
        if ebit and interest and interest != 0:
            coverage = ebit / abs(interest)
            if coverage > 3:
                score += 1
                details.append(f"Interest coverage × {coverage:.1f}")
            else:
                details.append(f"Weak coverage × {coverage:.1f}")
        else:
            details.append("Interest coverage NA")

        cost_of_equity = cls.estimate_cost_of_equity(beta)

        return {
            "score": score,
            "max_score": max_score,
            "details": "; ".join(details),
            "beta": beta,
            "cost_of_equity": cost_of_equity,
        }

    @staticmethod
    def analyze_relative_valuation(metrics: list) -> dict[str, Any]:
        """P/E vs its own 5-period median: +1 below 70 %, -1 above 130 %, else 0 (max 1)."""
        max_score = 1
        if not metrics or len(metrics) < 5:
            return {"score": 0, "max_score": max_score, "details": "Insufficient P/E history"}

        pes = [m.price_to_earnings_ratio for m in metrics if m.price_to_earnings_ratio]
        if len(pes) < 5:
            return {"score": 0, "max_score": max_score, "details": "P/E data sparse"}

        ttm_pe = pes[0]
        median_pe = sorted(pes)[len(pes) // 2]

        if ttm_pe < 0.7 * median_pe:
            score, desc = 1, f"P/E {ttm_pe:.1f} vs. median {median_pe:.1f} (cheap)"
        elif ttm_pe > 1.3 * median_pe:
            score, desc = -1, f"P/E {ttm_pe:.1f} vs. median {median_pe:.1f} (expensive)"
        else:
            score, desc = 0, "P/E inline with history"

        return {"score": score, "max_score": max_score, "details": desc}

    @staticmethod
    def calculate_intrinsic_value_dcf(metrics: list, line_items: list, risk_analysis: dict) -> dict[str, Any]:
        """FCFF DCF: base = latest FCF, growth = revenue CAGR capped 12 %, linear fade to 2.5 % over 10 years, discounted at cost of equity."""
        if not metrics or len(metrics) < 2 or not line_items:
            return {"intrinsic_value": None, "details": ["Insufficient data"]}

        latest_m = metrics[0]
        fcff0 = getattr(latest_m, "free_cash_flow", None)
        shares = getattr(line_items[0], "outstanding_shares", None)
        if not fcff0 or not shares:
            return {"intrinsic_value": None, "details": ["Missing FCFF or share count"]}

        growth = Persona.cagr(metrics, "revenue")
        base_growth = min(growth[0], 0.12) if growth is not None else 0.04

        terminal_growth = 0.025
        years = 10

        discount = risk_analysis.get("cost_of_equity") or 0.09

        pv_sum = 0.0
        g = base_growth
        g_step = (terminal_growth - base_growth) / (years - 1)
        for yr in range(1, years + 1):
            fcff_t = fcff0 * (1 + g)
            pv = fcff_t / (1 + discount) ** yr
            pv_sum += pv
            g += g_step

        tv = (
            fcff0
            * (1 + terminal_growth)
            / (discount - terminal_growth)
            / (1 + discount) ** years
        )

        equity_value = pv_sum + tv
        intrinsic_per_share = equity_value / shares

        return {
            "intrinsic_value": equity_value,
            "intrinsic_per_share": intrinsic_per_share,
            "assumptions": {
                "base_fcff": fcff0,
                "base_growth": base_growth,
                "terminal_growth": terminal_growth,
                "discount_rate": discount,
                "projection_years": years,
            },
            "details": ["FCFF DCF completed"],
        }

    @staticmethod
    def estimate_cost_of_equity(beta: float | None) -> float:
        """CAPM: r_e = r_f + β × ERP (Damodaran's long-term averages)."""
        risk_free = 0.04
        erp = 0.05
        beta = beta if beta is not None else 1.0
        return risk_free + beta * erp
