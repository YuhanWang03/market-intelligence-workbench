"""Print actual metric values for the entire TECH_30 universe.

Use this to tune filter thresholds against real-world distribution.

Usage:
    poetry run python scripts/dump_universe_metrics.py
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
from dotenv import load_dotenv

from v2.data import FDClient
from v2.data.price_source import default_price_source
from v2.screening import TECH_30

load_dotenv()


def fmt_money(v: float | None) -> str:
    if v is None:
        return "     —"
    if v >= 1e12:
        return f"${v / 1e12:5.2f}T"
    if v >= 1e9:
        return f"${v / 1e9:5.1f}B"
    if v >= 1e6:
        return f"${v / 1e6:5.1f}M"
    return f"${v:,.0f}"


def fmt_pct(v: float | None, *, signed: bool = False) -> str:
    if v is None:
        return "    —"
    return f"{v:+6.1%}" if signed else f"{v:5.1%}"


def main() -> None:
    today = date.today()
    today_str = today.isoformat()
    history_start = (today - timedelta(days=400)).isoformat()

    print(
        f"\n{'Tkr':<6} {'MktCap':>7}  {'RevGr':>7}  {'GrMrg':>6}  {'AnVol':>6}  Notes"
    )
    print("─" * 70)

    counts = {"have_rev": 0, "have_gm": 0, "have_vol": 0, "no_data": 0}

    # Phase 4.5-mini: prices via yfinance (no FD 3-day lag); FD still
    # serves the financial-metrics column.
    price_source = default_price_source()
    with FDClient() as fd:
        for ticker in TECH_30:
            metrics = fd.get_financial_metrics(ticker, today_str, limit=1)
            prices = price_source.get_prices(ticker, history_start, today_str)

            note_parts: list[str] = []
            mc = rev = gm = vol = None

            if not metrics:
                note_parts.append("no metrics")
                counts["no_data"] += 1
            else:
                m = metrics[0]
                mc = m.market_cap
                rev = m.revenue_growth
                gm = m.gross_margin
                if rev is not None:
                    counts["have_rev"] += 1
                if gm is not None:
                    counts["have_gm"] += 1

            if len(prices) >= 60:
                closes = np.array([p.close for p in prices], dtype=float)
                log_rets = np.diff(np.log(closes))
                vol = float(log_rets.std(ddof=1) * np.sqrt(252))
                counts["have_vol"] += 1
            else:
                note_parts.append(f"{len(prices)} bars")

            print(
                f"{ticker:<6} {fmt_money(mc):>7}  "
                f"{fmt_pct(rev, signed=True):>7}  "
                f"{fmt_pct(gm):>6}  "
                f"{fmt_pct(vol):>6}  "
                f"{' / '.join(note_parts)}"
            )

    print("─" * 70)
    print(
        f"Data availability: rev={counts['have_rev']}/30  "
        f"gm={counts['have_gm']}/30  vol={counts['have_vol']}/30  "
        f"no_data={counts['no_data']}/30"
    )


if __name__ == "__main__":
    main()
