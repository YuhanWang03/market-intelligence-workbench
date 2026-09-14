from types import SimpleNamespace
import pandas as pd
import pytest
from v2.agent_v3.tools import Registry
from v2.agent_v3.data_extensions import register_data_extensions


def test_macro_uses_calendar_periods_and_keeps_observation_date():
    series = pd.Series([100.0]*12+[110.0], index=pd.date_range("2025-01-01",periods=13,freq="MS"))
    registry=Registry()
    register_data_extensions(registry,series_reader=lambda *a,**k:series)
    result=registry.handlers["macro.release"]({"release_type":"cpi"},None)
    yoy=next(e for e in result.evidence if e.id.startswith("fred-CPIAUCSL-yoy"))
    assert yoy.value==10.0 and yoy.period=="2026-01-01"
    assert "publication" in yoy.metadata["date_basis"]
    assert yoy.source_url.endswith("CPIAUCSL")


def test_missing_month_not_replaced_by_neighbor():
    series=pd.Series([100.,110.],index=pd.to_datetime(["2024-12-01","2026-01-01"]))
    registry=Registry();register_data_extensions(registry,series_reader=lambda *a,**k:series)
    result=registry.handlers["macro.release"]({"release_type":"cpi"},None)
    assert result.status.value=="partial_data"
    assert not any(e.metric=="yoy_pct" for e in result.evidence)


def test_holdings_preserve_provider_weights_not_normalized_top_subset():
    table=pd.DataFrame({"Name":["A","B"],"Holding Percent":[.07,.05]},index=["AAA","BBB"])
    registry=Registry();register_data_extensions(registry,fund_reader=lambda t:table)
    result=registry.handlers["etf.holdings"]({"ticker":"SPY","top":1},None)
    assert result.evidence[0].value==7 and result.evidence[0].as_of==""
    assert result.status.value=="partial_data" and len(result.evidence)==1


def test_fund_invalid_weight_rejected():
    registry=Registry();register_data_extensions(registry,fund_reader=lambda t:pd.DataFrame({"Holding Percent":[8.]},index=["A"]))
    with pytest.raises(ValueError):registry.handlers["etf.holdings"]({"ticker":"SPY"},None)


def test_lab_rejects_implicit_large_universe():
    from v2.agent_v3.lab import V3LabPort
    lab=V3LabPort();lab._bindings={"lab.sweep":SimpleNamespace()}
    result=lab.run("lab.sweep",{},None)
    assert result.status.value=="partial_data"


def test_telegram_flags_are_explicit_syntax():
    from v2.bot.agent_v3_bridge import parse_flags
    assert parse_flags("--web 查CPI")==("查CPI",True)
    assert parse_flags("查一下网页")==("查一下网页",False)


def test_lab_units_are_translated_without_changing_strategy_parameters():
    from v2.agent_v3.lab import percentage_payload
    raw={"rows":[{"total_return_pct":.69,"win_rate":.52,"near_high_pct":.1}],"fd_cost_usd":0,"cost_bps":10}
    fixed=percentage_payload(raw)
    assert fixed["rows"][0]=={"total_return_pct":69.,"win_rate":52.,"near_high_pct":.1}
    assert fixed["cost_bps"]==10 and raw["rows"][0]["total_return_pct"]==.69


def test_event_groups_retain_security_identity_and_statistics():
    from pydantic import BaseModel
    from v2.agent_v3.lab import V3LabPort
    class Input(BaseModel):
        tickers: list[str]
    lab = V3LabPort()
    payload = {"tickers":["NVDA"],"events":[{}],"aggregates":[{"group":"BEAT","windows":[{"window":"[0,+1]","n_events":1,"mean_car":.02,"p_value":.3,"ci":{"lower":-.01,"upper":.05}}]}]}
    lab._bindings = {"lab.event_study":SimpleNamespace(input_model=Input,supports_progress=False,runner=lambda body:payload)}
    result = lab.run("lab.event_study",{"tickers":["NVDA"]},SimpleNamespace(run_id="fixture",v3_run=SimpleNamespace(check=lambda:None)))
    row = next(e for e in result.evidence if isinstance(e.value,dict))
    assert row.entity == "NVDA" and row.value["group"] == "BEAT"
    assert row.value["mean_car"] == 2 and row.value["p_value"] == .3
