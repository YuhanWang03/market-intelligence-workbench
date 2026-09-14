"""Research estimates are independent of the earnings calendar date."""
import math


def number(value):
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def collect_consensus(ticker, client=None):
    """Keep source failures distinct from a successful empty response."""
    if client is None:
        import yfinance as yf
        client = yf.Ticker(ticker)
    values, attempts = {}, []
    # Both tables use the same explicit target period. Never combine different
    # forecast horizons, nor interpret actual earnings as a forward estimate.
    for endpoint, field in [('earnings_estimate', 'eps_estimate'), ('revenue_estimate', 'revenue_estimate')]:
        try:
            table = getattr(client, endpoint)
            value = number(table.loc['0q', 'avg']) if table is not None and '0q' in table.index and 'avg' in table.columns else None
            values[field] = value
            attempts.append({'source': endpoint, 'status': 'AVAILABLE' if value is not None else 'NO_DATA'})
        except Exception:
            attempts.append({'source': endpoint, 'status': 'FETCH_FAILED'})
    if any(value is not None for value in values.values()):
        return {**values, 'period': '0q', 'period_label': '数据源当前预测季度', 'source': 'Yahoo Finance', 'attempts': attempts}
    # Calendar averages are a fallback, not gated on a future release date.
    try:
        calendar = client.calendar
        if not isinstance(calendar, dict):
            calendar = {}
        for field, keys in [('eps_estimate', ('Earnings Average', 'EPS Estimate', 'epsEstimate')),
                            ('revenue_estimate', ('Revenue Average', 'Revenue Estimate', 'revenueEstimate'))]:
            values[field] = next((number(calendar[key]) for key in keys if number(calendar.get(key)) is not None), None)
        attempts.append({'source': 'calendar', 'status': 'AVAILABLE' if any(v is not None for v in values.values()) else 'NO_DATA'})
    except Exception:
        attempts.append({'source': 'calendar', 'status': 'FETCH_FAILED'})
    return {**values, 'period_label': '财报日历对应期间（需核对报告期）', 'source': 'Yahoo Finance', 'attempts': attempts}
