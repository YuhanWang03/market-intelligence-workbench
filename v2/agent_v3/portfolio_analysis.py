"""Position metrics and holding periods are data contracts, not model guesses."""
from dataclasses import replace
from datetime import date, datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo
import math

from v2.agent_v2.catalog import CapabilityCatalog, CapabilitySpec
from v2.agent_v2.models import EvidenceItem, ToolEnvelope, ResultStatus


def finite(value):
    try:
        number=float(value)
        return number if math.isfinite(number) else None
    except (TypeError,ValueError):return None


def rank_positions(positions,metric,direction="low",daily=None):
    field={"unrealized_percent":"unrealized_pl_pct","unrealized_amount":"unrealized_pl","daily_return":"daily_return"}[metric]
    ranked=[]; missing=[]
    for position in positions:
        ticker=position["ticker"]
        value=finite((daily or {}).get(ticker,{}).get("value") if metric=="daily_return" else position.get(field))
        if value is None or (metric=="daily_return" and not (daily or {}).get(ticker,{}).get("as_of")):missing.append(ticker);continue
        ranked.append({"ticker":ticker,"value":value,"as_of":(daily or {}).get(ticker,{}).get("as_of",""),"position":position})
    if metric=="daily_return" and len({row["as_of"] for row in ranked})>1:
        return [],[row["ticker"] for row in ranked]+missing
    ranked.sort(key=lambda row:((-row["value"] if direction=="high" else row["value"]),row["ticker"]))
    for index,row in enumerate(ranked):
        row["rank"]=index+1 if index==0 or row["value"]!=ranked[index-1]["value"] else ranked[index-1]["rank"]
    return ranked,missing


def read_fills(ticker):
    """Read-only, bounded broker activity pagination; never submit an order."""
    from v2.broker.alpaca_client import _client
    client=_client(); rows=[]; token=None; seen=set()
    for _ in range(20):
        args={"direction":"asc","page_size":100}
        if token:args["page_token"]=token
        page=client.get("/account/activities/FILL",data=args)
        if not isinstance(page,list):raise ValueError("Invalid broker activity response")
        if not page:return {"complete":True,"fills":[row for row in rows if row.get("symbol")==ticker]}
        for row in page:
            if row["id"] in seen:return {"complete":False,"fills":[]}
            seen.add(row["id"]);rows.append(row)
        token=page[-1]["id"]
        if len(page)<100:return {"complete":True,"fills":[row for row in rows if row.get("symbol")==ticker]}
    return {"complete":False,"fills":[]}


def holding_period(position,history):
    """Reconcile the continuous long-position cycle with the broker snapshot."""
    if not history.get("complete"):return {"verified":False,"reason":"成交记录不完整，无法确定连续持有期。"}
    qty=Decimal(0); cost=Decimal(0); start=None; buys=0; sells=0
    try:
        def fill_time(row):
            stamp=datetime.fromisoformat(row["transaction_time"].replace("Z","+00:00"))
            if stamp.tzinfo is None:raise ValueError("Fill timezone unavailable")
            return stamp.astimezone(ZoneInfo("America/New_York"))
        for fill in sorted(history.get("fills",[]),key=fill_time):
            if fill.get("symbol")!=position["ticker"]:continue
            units=Decimal(str(fill["qty"])); price=Decimal(str(fill["price"]))
            if not units.is_finite() or not price.is_finite() or units<=0 or price<=0:raise ValueError()
            if fill["side"]=="buy":
                if qty==0:start=fill_time(fill).date().isoformat();buys=0;sells=0
                qty+=units;cost+=units*price;buys+=1
            elif fill["side"]=="sell" and units<=qty:
                cost-=cost*units/qty;qty-=units;sells+=1
                if qty==0:start=None
            else:raise ValueError()
        expected=Decimal(str(position["qty"])); average=Decimal(str(position["avg_entry_price"]))
        if not start or qty<=0 or abs(qty-expected)>Decimal("0.000001") or abs(cost/qty-average)>max(Decimal("0.02"),abs(average)*Decimal("0.001")):
            raise ValueError()
        if date.fromisoformat(start)>date.today():raise ValueError("Future holding date")
    except (ValueError,KeyError,ArithmeticError):
        return {"verified":False,"reason":"成交记录与当前数量/成本未对账，可能缺少转入、拆股或成本调整；不推测买入日期。"}
    return {"verified":True,"start":start,"buy_fills":buys,"sell_fills":sells,"basis":"continuous_holding_cycle","multiple_cashflows":buys>1 or sells>0}


def adjusted_history(ticker,start,end):
    """Explicit adjusted daily closes and exchange sessions, independent of V2 windows."""
    import yfinance as yf
    from alpaca.trading.requests import GetCalendarRequest
    from v2.broker.alpaca_client import _client
    before=date.fromisoformat(start)-timedelta(days=14)
    sessions=_client().get_calendar(GetCalendarRequest(start=before,end=date.fromisoformat(end)))
    today=datetime.now(ZoneInfo("America/New_York"))
    dates=[]
    for row in sessions:
        day=str(row.date)[:10]
        close=row.close
        if close.tzinfo is None:close=close.replace(tzinfo=ZoneInfo("America/New_York"))
        if close<=today:dates.append(day)
    frame=yf.Ticker(ticker).history(start=before.isoformat(),end=(date.fromisoformat(end)+timedelta(days=1)).isoformat(),auto_adjust=True,actions=True)
    bars=[{"date":stamp.date().isoformat(),"close":float(row["Close"])} for stamp,row in frame.iterrows()]
    return {"bars":bars,"sessions":dates,"calendar_from":before.isoformat(),"calendar_through":end,"adjustment":"split_and_dividend_adjusted","source":"Yahoo Finance adjusted daily prices; Alpaca exchange calendar"}


def extreme_days(history,start,end,top=3):
    if not history.get("calendar_from") or history["calendar_from"]>start or history.get("calendar_through","")<end:
        return {"complete":False,"reason":"交易日历未覆盖整个查询区间，不能确认最大下跌日。","events":[]}
    sessions=sorted(set(history.get("sessions",[])))
    # A daily bar on the purchase date includes trading before the fill.
    # Rank only subsequent complete sessions; never attribute that whole day
    # to a position that may have been opened intraday.
    within=[day for day in sessions if start<day<=end]
    previous=[day for day in sessions if day<=start]
    if not within or not previous or history.get("adjustment")!="split_and_dividend_adjusted":
        return {"complete":False,"reason":"缺少交易日历、前一交易日或统一复权口径；不能确认最大下跌日。","events":[]}
    needed=[previous[-1],*within];bars={}
    for row in history.get("bars",[]):
        if row["date"] in bars:return {"complete":False,"reason":"行情存在重复交易日。","events":[]}
        bars[row["date"]]=finite(row["close"])
    if any(not bars.get(day) or bars[day]<=0 for day in needed):
        return {"complete":False,"reason":"持有期行情缺少交易日，不能声称已找到区间最大下跌日。","events":[]}
    returns=[{"date":day,"return":bars[day]/bars[prior]-1} for prior,day in zip(needed,needed[1:])]
    falling=sorted((row for row in returns if row["return"]<0),key=lambda row:(row["return"],row["date"]))[:top]
    return {"complete":True,"start":start,"end":within[-1],"first_complete_session":within[0],"purchase_day_excluded":True,"adjustment":history["adjustment"],"events":[{**row,"id":f"holding-day-{row['date']}"} for row in falling],"session_count":len(within)}


def register_portfolio_workflows(registry,*,fill_source=None,history_source=None):
    schema=lambda props,required:{"type":"object","properties":props,"required":required,"additionalProperties":False}
    registry.catalog=CapabilityCatalog([*registry.catalog.specs(),
        CapabilitySpec("account.ranking","account","Deterministically rank all current positions by unrealized percent, amount or daily return.",schema({"metric":{"enum":["unrealized_percent","unrealized_amount","daily_return"]},"direction":{"enum":["low","high"]}},["metric","direction"])),
        CapabilitySpec("account.position_analysis","account","Explain a position using reconciled holding dates; never infer a holding period from its loss percentage.",schema({"ticker":{"type":"string"}},["ticker"]),long_running=True)])
    portfolio=registry.handlers["account.portfolio"]
    def ranking(args,ctx):
        base=portfolio({},ctx);positions=base.metadata["positions"];daily={}
        metric=args["metric"]
        if metric=="daily_return":
            for position in positions:
                ctx.v3_run.check()
                try:
                    result=registry.handlers["market.performance"]({"ticker":position["ticker"]},ctx)
                    item=next((item for item in result.evidence if item.metadata.get("metric")=="close"),None)
                    daily[position["ticker"]]={"value":result.metrics.get("returns",{}).get("1d"),"as_of":result.as_of or (item.as_of if item else "")}
                except Exception:
                    ctx.v3_run.check()
                    daily[position["ticker"]]={}
        rows,missing=rank_positions(positions,metric,args["direction"],daily)
        label={"unrealized_percent":"剩余持仓相对成本的未实现盈亏比例","unrealized_amount":"剩余持仓未实现盈亏金额","daily_return":"股票单日涨跌幅（按下列行情日期，非累计持仓盈亏）"}[metric]
        evidence=[];lines=[f"{'仅在数据有效的持仓中，' if missing else ''}按{label}排序："]
        for row in rows:
            key=f"position-rank-{row['ticker']}"; value=f"{row['value']:+,.2f} 美元" if metric=="unrealized_amount" else f"{row['value']:+.2%}"
            claim=f"第{row['rank']}名 {row['ticker']}：{value}"+(f"（行情日期{row['as_of']}）" if row["as_of"] else "")
            evidence.append(EvidenceItem(key,row["ticker"],claim,value=row["value"],as_of=row["as_of"] or base.as_of,source_id="account.ranking",metadata={"ranking":row,"metric":metric}))
            lines.append(f"- {claim} [{key}]")
        leaders=[row["ticker"] for row in rows if row["rank"]==1]
        limits=["部分持仓缺少有效数据，不能确定全账户排名："+", ".join(missing)] if missing else []
        if limits:lines.extend(limits)
        if not rows:lines.append("没有足够的同口径数据可排名。")
        focus={"ticker":leaders[0] if len(leaders)==1 and not missing else None,"metric":metric,"leaders":leaders,"position":rows[0]["position"] if len(leaders)==1 and not missing else None}
        return ToolEnvelope("account.ranking",ResultStatus.PARTIAL_DATA if missing or not rows else ResultStatus.COMPLETED,evidence=evidence,limitations=limits,metadata={"ranking":rows,"portfolio_context":focus,"deterministic_answer":"\n".join(lines),"review_stage":"source_cross_review"})
    def position_analysis(args,ctx):
        ticker=args["ticker"];base=portfolio({},ctx)
        position=next((row for row in base.metadata["positions"] if row["ticker"]==ticker),None)
        if not position:return ToolEnvelope("account.position_analysis",ResultStatus.PARTIAL_DATA,limitations=["当前持仓中没有该股票。"],metadata={"deterministic_answer":f"当前持仓中未找到 {ticker}，不能用上一轮快照解释现在的持仓。","portfolio_context":{}})
        key=f"holding-position-{ticker}"
        claim=f"{ticker} 剩余持仓相对成本的未实现盈亏为 {float(position['unrealized_pl_pct']):+.2%}（{float(position['unrealized_pl']):+,.2f} 美元），平均成本 {float(position['avg_entry_price']):.2f} 美元。"
        evidence=[EvidenceItem(key,ticker,claim,as_of=base.as_of,source_id="account.portfolio",metadata={"position":position})]
        try:period=holding_period(position,(fill_source or read_fills)(ticker))
        except Exception:period={"verified":False,"reason":"成交记录读取失败，买入日期尚未核实。"}
        focus={"ticker":ticker,"metric":"unrealized_percent","position":position,"period":period}
        metadata={"portfolio_context":focus,"holding_period":period,"review_stage":"source_cross_review"}
        if not period["verified"]:
            metadata["deterministic_answer"]=f"{claim} [{key}]\n\n{period['reason']} 因此暂时无法解释你实际持有期间的亏损。请补充买入日期或完整成交记录；不会用近三个月收益或高低点回撤替代持仓亏损。"
            return ToolEnvelope("account.position_analysis",ResultStatus.PARTIAL_DATA,subject=ticker,evidence=evidence,limitations=[period["reason"]],metadata=metadata)
        end=date.today().isoformat()
        period_claim=f"{ticker} 当前连续持有期始于 {period['start']}，成交记录与当前数量和平均成本已对账；多次买入或部分卖出时，市场区间回报不等于该持仓盈亏。"
        evidence.append(EvidenceItem(f"holding-period-{ticker}",ticker,period_claim,source_id="broker.fill_history",metadata=period))
        try:extremes=extreme_days((history_source or adjusted_history)(ticker,period["start"],end),period["start"],end)
        except Exception:extremes={"complete":False,"events":[],"reason":"历史行情或交易日历读取失败，无法确认最大下跌日。"}
        metadata["holding_events"]=extremes
        for event in extremes["events"]:
            evidence.append(EvidenceItem(event["id"],ticker,f"{event['date']} 复权日回报 {event['return']:+.2%}。",value=event["return"],as_of=event["date"],source_id="adjusted_holding_history",metadata={"event":event}))
        limits=[] if extremes["complete"] else [extremes["reason"]]
        event_dates=[row['date'] for row in extremes['events']]
        request=replace(ctx.request,text=f"解释{ticker}在连续持有期{period['start']}至{end}的市场背景，重点核查这些交易日：{event_dates}。不能把市场回报或单日原因等同于实际持仓盈亏。",entities=(ticker,),metadata={**ctx.request.metadata,"date_window":{"start":period["start"],"end":end,"basis":"event"},"holding_event_dates":event_dates})
        if ctx.allow_web:
            investigate=getattr(registry,"holding_news",None)
            if investigate:
                try:
                    result=investigate(ticker,request,ctx.v3_run)
                    evidence.extend(result.evidence);limits.extend(result.limitations)
                    metadata['news_search_queries']=result.metadata.get('queries',[])
                    metadata['news_retrieval_metrics']=result.metrics
                except Exception:limits.append("持有期新闻调查未完成，保留已核实持仓与交易日数据。")
            else:limits.append("持有期新闻调查服务未配置。")
        else:limits.append("本轮未开启网页检索，不能核查持有期新闻。")
        metadata['event_coverage']=[{'date':day,'candidate_evidence_ids':[item.id for item in evidence if item.metadata.get('event_date')==day and item.metadata.get('quote_located')],'causality_confirmed':False} for day in event_dates]
        missing_events=[row['date'] for row in metadata['event_coverage'] if not row['candidate_evidence_ids']]
        if missing_events:limits.append('以下交易日未找到通过原文与事件日期核验的新闻，原因仍未确认：'+', '.join(missing_events))
        limits.append("单日媒体归因和条件性推断不能解释全部持仓盈亏。")
        metadata["answer_constraints"]=[{"forbid_claim":"将市场区间收益或高低点回撤等同于这笔持仓的实际盈亏，或把列出的几个交易日解释为整个持有期亏损的全部原因。","warning":"保留持仓成本、连续持有期和市场回报的不同口径。"},{"forbid_claim":"把尚未证实因果的事件称为已解释了部分跌幅或亏损，或把找到下跌交易日称为确认了驱动因素。","warning":"尚未证实的事件只是候选线索；找到下跌日期不等于确认下跌原因。"}]
        return ToolEnvelope("account.position_analysis",ResultStatus.PARTIAL_DATA,subject=ticker,evidence=evidence,limitations=limits,metadata=metadata)
    registry.register("account.ranking",ranking)
    registry.register("account.position_analysis",position_analysis)
