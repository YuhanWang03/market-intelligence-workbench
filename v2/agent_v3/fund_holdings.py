"""Issuer holdings files with effective dates and complete row retention."""
import csv
import io
import math
from datetime import datetime
from v2.agent_v2.models import EvidenceItem, ToolEnvelope, ResultStatus

ISSUER_FILES = {"IVV":"https://www.ishares.com/us/products/239726/ishares-core-s-p-500-etf/latest-holdings.csv"}


def issuer_holdings(ticker, top=5, *, get=None):
    url = ISSUER_FILES[ticker]
    if get is None:
        import requests
        def get(url):
            response = requests.get(url,timeout=15)
            response.raise_for_status()
            return response.content.decode("utf-8-sig")
    rows = list(csv.reader(io.StringIO(get(url))))
    stamp = next(row[1] for row in rows if row and row[0] == "Fund Holdings as of")
    effective = datetime.strptime(stamp,"%b %d, %Y").date().isoformat()
    start = next(i for i,row in enumerate(rows) if row and row[0] == "Ticker" and "Weight (%)" in row)
    columns = rows[start]
    holdings = []
    for values in rows[start+1:]:
        if len(values) < 2:
            continue
        if len(values) != len(columns):
            raise ValueError("Issuer holdings row does not match CSV schema")
        row = dict(zip(columns,values))
        weight = float(row["Weight (%)"].replace(",",""))
        if not math.isfinite(weight) or abs(weight) > 100:
            raise ValueError("Invalid issuer weight")
        holdings.append({"ticker":row["Ticker"],"name":row["Name"],"weight_pct":weight,"asset_class":row["Asset Class"]})
    if not holdings:
        raise ValueError("Issuer file contains no holdings")
    ranked = sorted(holdings,key=lambda row:row["weight_pct"],reverse=True)
    evidence = [EvidenceItem(f"issuer-{ticker}-{effective}-{i}",ticker,
        f"{row['ticker']}（{row['name']}），{row['asset_class']}，持仓权重 {row['weight_pct']}%。",
        metric="holding_weight",value=row["weight_pct"],unit="%",as_of=effective,
        source_id=url,source_url=url,source_title="iShares published holdings CSV") for i,row in enumerate(ranked[:top])]
    evidence.append(EvidenceItem(f"issuer-{ticker}-inventory-{effective}",ticker,
        f"发行商文件持仓日期为 {effective}，共有 {len(holdings)} 条持仓记录；已保留完整记录，正文显示前 {min(top,len(holdings))} 项，其余见源文件。",
        value={"holding_count":len(holdings),"display_count":min(top,len(holdings))},as_of=effective,source_id=url,source_url=url,source_title="iShares published holdings CSV"))
    return ToolEnvelope("etf.holdings",ResultStatus.COMPLETED,subject=ticker,as_of=effective,evidence=evidence,
        metadata={"holdings":holdings,"complete_provider_file":True,"effective_date":effective,"weight_sum_pct":round(sum(row["weight_pct"] for row in holdings),4)})
