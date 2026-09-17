"""YoY is matched by date, so a month FRED never published does not move the base."""
import pandas as pd

from v2.macro.transforms import yoy_pct


def _cpi():
    index = pd.date_range("2025-07-01", "2026-08-01", freq="MS")
    values = [322.169, 323.291, 324.245, float("nan"), 325.063, 326.031, 326.588, 327.460, 330.293, 332.407, 333.979, 332.568, 332.813, 334.131]
    return pd.Series(values, index=index)


def test_missing_month_does_not_shift_the_year_ago_base():
    assert round(yoy_pct(_cpi()) * 100, 2) == 3.35  # by position this was 3.71


def test_missing_year_ago_observation_is_none_not_a_neighbour():
    series = _cpi()
    series.loc["2025-08-01"] = float("nan")
    assert yoy_pct(series) is None


def test_plain_list_still_uses_twelve_periods():
    assert yoy_pct(list(range(1, 13))) is None
    assert round(yoy_pct([100.0] * 12 + [103.0]), 4) == 0.03
