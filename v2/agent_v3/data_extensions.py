"""Structured macro and fund data; never parse display cards into numbers."""
from datetime import date, timedelta, datetime, timezone
import math

from v2.agent_v2.catalog import CapabilityCatalog, CapabilitySpec
from v2.agent_v2.models import EvidenceItem, ToolEnvelope, ResultStatus


def register_data_extensions(registry, *, series_reader=None, fund_reader=None, statement_reader=None):
    default_fund_reader = fund_reader is None
    if series_reader is None:
        from v2.macro.fred_client import get_series
        series_reader = get_series
    if fund_reader is None:
        def fund_reader(ticker):
            import yfinance as yf
            return yf.Ticker(ticker).funds_data.top_holdings
    schema = {"type":"object", "properties":{"ticker":{"type":"string","pattern":"^[A-Z][A-Z0-9.-]{0,7}$"}, "top":{"type":"integer","minimum":1,"maximum":10}}, "required":["ticker"], "additionalProperties":False}
    registry.catalog = CapabilityCatalog([*registry.catalog.specs(), CapabilitySpec("etf.holdings", "research", "Fund top holdings and reported weights; publication date may be unavailable.", schema)])

    def holdings(args, context):
        ticker = args["ticker"]
        issuer_limits = []
        if default_fund_reader:
            from v2.agent_v3.fund_holdings import ISSUER_FILES, issuer_holdings
            if ticker in ISSUER_FILES:
                try:
                    return issuer_holdings(ticker,args.get("top",5))
                except Exception as exc:
                    issuer_limits.append(f"发行商完整持仓读取失败（{type(exc).__name__}）；以下降级为第三方前列持仓。")
        table = fund_reader(ticker)
        if table is None or table.empty:
            return ToolEnvelope("etf.holdings", ResultStatus.PARTIAL_DATA, limitations=["Provider returned no fund holdings"])
        evidence = []
        for symbol, row in table.sort_values("Holding Percent", ascending=False).head(args.get("top",5)).iterrows():
            fraction = float(row["Holding Percent"])
            if not math.isfinite(fraction) or not 0 <= fraction <= 1:
                raise ValueError("Invalid provider holding weight")
            weight = round(fraction*100, 4)
            evidence.append(EvidenceItem(f"holding-{ticker}-{symbol}", ticker, f"{symbol}（{row.get('Name',symbol)}）持仓权重 {weight}%", metric="holding_weight", value=weight, unit="%", source_id="yahoo_fund_holdings", source_title="Yahoo Finance fund holdings", source_url=f"https://finance.yahoo.com/quote/{ticker}/holdings/", metadata={"holding_symbol":str(symbol), "retrieved_at":datetime.now(timezone.utc).isoformat(), "weight_basis":"reported portfolio weight; not renormalized"}))
        return ToolEnvelope("etf.holdings", ResultStatus.PARTIAL_DATA, subject=ticker, evidence=evidence, limitations=[*issuer_limits,"供应商未提供持仓生效日期；仅为所返回的前列持仓，不是完整组合，抓取时间不能当持仓日期。"])

    series_by_release = {"cpi":["CPIAUCSL","CPILFESL"],"pce":["PCEPI","PCEPILFE"],"ppi":["PPIFIS"],"nfp":["PAYEMS","UNRATE"],"gdp":["GDPC1"],"claims":["ICSA"],"fomc":["DFEDTARL","DFEDTARU"]}
    def macro(args, context, overview=False):
        from v2.macro.series import FRED_SERIES
        from pandas import DateOffset
        ids = ["CPIAUCSL","UNRATE","DFEDTARU","DGS10","VIXCLS"] if overview else series_by_release[args["release_type"]]
        evidence, missing = [], []
        for sid in ids:
            try:
                series = series_reader(sid, start=(date.today()-timedelta(days=800)).isoformat(), end=date.today().isoformat()).sort_index().dropna()
                if series.empty:
                    raise ValueError("Empty series")
                stamp, value = series.index[-1], float(series.iloc[-1])
                if not math.isfinite(value):
                    raise ValueError("Nonfinite observation")
                period = stamp.strftime("%Y-%m-%d")
                observations = [("level",value,"原始序列单位")]
                transform = FRED_SERIES[sid]["transform"]
                if transform == "mom_yoy":
                    for label, months in [("mom_pct",1),("yoy_pct",12)]:
                        previous = series.get(stamp-DateOffset(months=months))
                        if previous is not None and float(previous) != 0:
                            observations.append((label,round((value/float(previous)-1)*100,4),"%"))
                        else:
                            missing.append(f"{sid}: {label} comparison period missing")
                elif sid == "PAYEMS":
                    previous = series.get(stamp-DateOffset(months=1))
                    if previous is not None:
                        observations.append(("monthly_change",value-float(previous),"千人"))
                elif sid == "GDPC1":
                    previous = series.get(stamp-DateOffset(months=3))
                    if previous is not None and previous > 0:
                        observations.append(("annualized_qoq",round(((value/float(previous))**4-1)*100,4),"%"))
                for metric, val, unit in observations:
                    evidence.append(EvidenceItem(f"fred-{sid}-{metric}-{period}",sid,f"{FRED_SERIES[sid]['name']} {period} {metric} = {val} {unit}",metric=metric,value=val,unit=unit,period=period,as_of=period,source_id="FRED",source_title=f"FRED {sid}",source_url=f"https://fred.stlouisfed.org/series/{sid}",metadata={"date_basis":"observation period; not publication date", "seasonal_adjustment":"as specified by source series", "derived":metric != "level"}))
            except Exception as exc:
                missing.append(f"{sid}: {type(exc).__name__}")
        if not overview and args.get("release_type") == "fomc":
            try:
                from v2.agent_v3.fomc import read_statements
                statements = (statement_reader or read_statements)()
                evidence.extend(statements.evidence)
                missing.extend(statements.limitations)
            except Exception as exc:
                missing.append(f"FOMC声明正文读取失败：{type(exc).__name__}")
        return ToolEnvelope("macro.overview" if overview else "macro.release", ResultStatus.PARTIAL_DATA if missing else ResultStatus.COMPLETED, evidence=evidence, limitations=missing, metadata={"observation_dates_not_release_dates":True})
    registry.register("etf.holdings",holdings)
    registry.register("macro.release",macro)
    registry.register("macro.overview",lambda args,ctx:macro(args,ctx,True))
