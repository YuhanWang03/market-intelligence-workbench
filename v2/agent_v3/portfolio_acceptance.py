"""Opt-in real-model multi-turn acceptance with synthetic account data only.

No broker, market, search or messaging service is contacted. Only the configured
language model is live. Full answers are retained for review, not just statuses.
"""
import argparse
import json
import os
from pathlib import Path
import time

from v2.agent_v2.models import ToolEnvelope, ResultStatus, EvidenceItem
from v2.agent_v3.adapters import register_account
from v2.agent_v3.brain import ModelBrain
from v2.agent_v3.context import RunContext
from v2.agent_v3.graph import AgentV3, AgentV3Config
from v2.agent_v3.portfolio_analysis import register_portfolio_workflows
from v2.agent_v3.tools import Registry


QUESTIONS = [
    ["我的持仓哪只亏损比例最大？", "为什么亏这么多？", "那今天跌得最多的是哪只？", "按亏损金额重新排名。", "只解释我买入之后的变化。"],
    ["账户里的股票，按浮亏百分比从严重到轻排一下。", "这笔持仓为什么亏成这样？", "改看这些持仓最新交易日的跌幅，谁最差？", "再按浮亏美元数从多到少列出来。", "刚才排首位的，只谈我持有它期间发生了什么。"],
]


def snapshot():
    return {"positions": [{"symbol":ticker,"qty":qty,"avg_entry_price":100,"current_price":100*(1+pct),"unrealized_pl_pct":pct,"unrealized_pl":round(qty*100*pct,2)} for ticker,qty,pct in [("ARM",2,-.3202),("MRVL",10,-.2404),("QCOM",2,-.2458)]]}


def fixture_history(ticker,start,end):
    days = ['2026-08-31','2026-09-01','2026-09-02','2026-09-03','2026-09-04','2026-09-08','2026-09-09','2026-09-10','2026-09-11']
    latest=next(row['current_price'] for row in snapshot()['positions'] if row['symbol']==ticker)
    daily={'ARM':.01,'QCOM':-.02,'MRVL':-.04}[ticker]
    return {"sessions":days,"calendar_from":days[0],"calendar_through":"2026-09-13","adjustment":"split_and_dividend_adjusted","bars":[{"date":day,"close":value} for day,value in zip(days,[100,100,98,90,92,80,81,latest/(1+daily),latest])]}


def fixture_news(ticker,request,run):
    return ToolEnvelope('web.research',ResultStatus.PARTIAL_DATA,evidence=[
        EvidenceItem('fixture-event-'+ticker,ticker,'合成测试材料：2026-09-08 公司发布需求展望调整公告。公告本身没有确认它是当日下跌的原因。',as_of='2026-09-08',source_id='synthetic-company-announcement',metadata={'quote_located':True,'event_date':'2026-09-08'}),
        EvidenceItem('fixture-media-'+ticker,ticker,'合成测试材料：一家媒体将2026-09-08的下跌与需求展望调整联系起来；同文转载不算独立交叉核实，未证实因果。',as_of='2026-09-08',source_id='synthetic-media',metadata={'quote_located':True,'event_date':'2026-09-08'}),
    ],limitations=['合成测试资料仅用于流程验收，不是真实市场新闻；其他交易日没有事件证据。'])


def build_fixture_agent(model,known):
    registry=Registry()
    register_account(registry,portfolio_source=snapshot)
    # Remove all default private-account readers except the injected snapshot.
    for name in ['account.performance','account.risk']:registry.handlers.pop(name,None)
    def fills(ticker):
        qty=next(row['qty'] for row in snapshot()['positions'] if row['symbol']==ticker)
        return {'complete':True,'fills':[{'symbol':ticker,'side':'buy','qty':qty,'price':100,'transaction_time':'2026-09-01T15:00:00Z'}] if known else []}
    register_portfolio_workflows(registry,fill_source=fills,history_source=fixture_history)
    registry.holding_news=fixture_news
    registry.register('market.performance',lambda args,ctx:ToolEnvelope('market.performance',ResultStatus.COMPLETED,as_of='2026-09-11',metrics={'returns':{'1d':{'ARM':.01,'QCOM':-.02,'MRVL':-.04}[args['ticker']]}}))
    return AgentV3(registry=registry,brain=ModelBrain(model),config=AgentV3Config(max_seconds=100,enable_web=True,debate=False))


def configured_model():
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2]/'.env',override=False)
    from v2.agent_common.llm import OpenAICompatLLM
    previous=OpenAICompatLLM()
    for key,value in [('MODEL',previous.model),('BASE_URL',previous.base_url),('API_KEY',previous.api_key)]:os.environ.setdefault('AGENT_V3_'+key,value)
    if os.environ['AGENT_V3_BASE_URL'].rstrip('/')=='https://api.deepseek.com/v1':os.environ.setdefault('AGENT_V3_THINKING','disabled')
    from v2.agent_v3.runtime import build_model
    return build_model()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--live',action='store_true')
    args=parser.parse_args()
    if not args.live:parser.error('--live required for model calls')
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    if (output/'summary.json').exists():parser.error('Use a new output directory')
    def write(name,value):(output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    model=configured_model();summary=[]
    write('conditions.json',{'scope':'real model; synthetic account, fills, market history and news; no real private data or financial source quality acceptance','model':os.environ['AGENT_V3_MODEL'],'questions':QUESTIONS})
    for known in [False,True]:
        for repeat,questions in enumerate(QUESTIONS):
            agent=build_fixture_agent(model,known)
            try:
                for index,question in enumerate(questions):
                    cid=f'{"known" if known else "missing"}-{repeat+1}-{index+1}'
                    result=agent.run(question,session_id='sequence',allow_web=True)
                    data=result.to_dict();write(cid+'.json',{'question':question,'result':data})
                    envelope=result.results[0] if result.results else None
                    metadata=envelope.metadata if envelope else {}
                    context=metadata.get('portfolio_context',{})
                    expected_ticker=['ARM','ARM','MRVL','MRVL','MRVL'][index]
                    expected_metric=['unrealized_percent','unrealized_percent','daily_return','unrealized_amount','unrealized_percent'][index]
                    checks={'subject':context.get('ticker')==expected_ticker,'metric':context.get('metric')==expected_metric,'verified_answer':result.verification.ok,'no_fallback':result.synthesis.get('outcome')!='fallback','capability':bool(envelope) and envelope.capability==('account.position_analysis' if index in [1,4] else 'account.ranking')}
                    if index in [1,4]:checks['holding_period']=metadata.get('holding_period',{}).get('verified')==known
                    summary.append({'id':cid,'passed':all(checks.values()),'checks':checks,'seconds':result.elapsed_ms/1000})
                    write('summary.json',summary);print(json.dumps(summary[-1]),flush=True)
            finally:agent.store.close()
    records=[{'holding_period':{'verified':True,'start':'2026-06-18'},'holding_events':{'complete':True,'events':[{'date':'2026-06-24','return':-.0814}]}}]
    for name,answer,should_reject in [('wrong-month','2026-06-24 下跌8.14%。限制：上述同一个下跌日（7月24日）没有事件证据。',True),('correct-month','2026-06-24 下跌8.14%。限制：上述同一个下跌日（6月24日）没有事件证据。',False)]:
        objections=ModelBrain(model).audit_portfolio(answer,records,RunContext(name,time.monotonic()+60))
        item={'id':name,'passed':bool(objections)==should_reject,'objections':objections}
        summary.append(item);write('summary.json',summary);print(json.dumps(item,ensure_ascii=False),flush=True)
    raise SystemExit(0 if all(row['passed'] for row in summary) else 1)


if __name__=='__main__':main()
