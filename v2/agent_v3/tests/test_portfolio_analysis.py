from dataclasses import replace
from types import SimpleNamespace
import time
import pytest

from v2.agent_v3.portfolio_analysis import rank_positions,holding_period,extreme_days,register_portfolio_workflows
from v2.agent_v3.adapters import register_account
from v2.agent_v3.context import RunContext
from v2.agent_v3.contracts import SemanticIntent
from v2.agent_v3.demo import build_demo_agent
from v2.agent_v3.brain import ModelBrain
from v2.agent_v3.tools import Registry
from v2.agent_v2.models import ToolEnvelope,ResultStatus,EvidenceItem


def positions():
    return [{"symbol":ticker,"ticker":ticker,"qty":2,"avg_entry_price":100,"current_price":100*(1+pct),"unrealized_pl_pct":pct,"unrealized_pl":amount} for ticker,pct,amount in [("ARM",-.3202,-64.04),("MRVL",-.2404,-100),("QCOM",-.2458,-49.16)]]


def test_percentage_and_amount_ranking_are_different_and_deterministic():
    assert [r['ticker'] for r in rank_positions(positions(),'unrealized_percent')[0]]==['ARM','QCOM','MRVL']
    assert [r['ticker'] for r in rank_positions(positions(),'unrealized_amount')[0]]==['MRVL','ARM','QCOM']


def test_daily_ranking_never_uses_lifetime_losses_or_mixed_dates():
    daily={ticker:{'value':value,'as_of':'2026-09-11'} for ticker,value in [('ARM',.02),('QCOM',-.01),('MRVL',-.03)]}
    assert rank_positions(positions(),'daily_return',daily=daily)[0][0]['ticker']=='MRVL'
    daily['ARM']['as_of']='2026-09-10'
    assert not rank_positions(positions(),'daily_return',daily=daily)[0]


def test_missing_values_and_ties_are_explicit():
    rows=positions();rows[1]['unrealized_pl_pct']=None;rows[2]['unrealized_pl_pct']=rows[0]['unrealized_pl_pct']
    ranked,missing=rank_positions(rows,'unrealized_percent')
    assert missing==['MRVL'] and [r['rank'] for r in ranked]==[1,1]


def fill(side,qty,price,day):return {'symbol':'ARM','side':side,'qty':qty,'price':price,'transaction_time':day+'T15:00:00Z'}


def test_multiple_buys_and_partial_sale_reconcile_continuous_period():
    position={'ticker':'ARM','qty':3,'avg_entry_price':150}
    history={'complete':True,'fills':[fill('buy',2,100,'2026-06-18'),fill('buy',2,200,'2026-06-23'),fill('sell',1,160,'2026-06-24')]}
    result=holding_period(position,history)
    assert result['verified'] and result['start']=='2026-06-18' and result['multiple_cashflows']
    assert not holding_period({**position,'qty':4},history)['verified']
    assert not holding_period(position,{**history,'complete':False})['verified']


def test_closed_then_reopened_position_uses_new_cycle():
    history={'complete':True,'fills':[fill('buy',1,100,'2025-01-01'),fill('sell',1,110,'2025-02-01'),fill('buy',2,100,'2026-06-18')]}
    assert holding_period({'ticker':'ARM','qty':2,'avg_entry_price':100},history)['start']=='2026-06-18'
    assert not holding_period({'ticker':'ARM','qty':2,'avg_entry_price':100},{'complete':True,'fills':[]})['verified']


def test_fill_date_uses_exchange_timezone():
    row=fill('buy',2,100,'2026-06-19');row['transaction_time']='2026-06-19T00:30:00Z'
    assert holding_period({'ticker':'ARM','qty':2,'avg_entry_price':100},{'complete':True,'fills':[row]})['start']=='2026-06-18'


def price_history():
    days=['2026-06-17','2026-06-18','2026-06-22','2026-06-23','2026-06-24','2026-06-25']
    return {'sessions':days,'calendar_from':days[0],'calendar_through':days[-1],'adjustment':'split_and_dividend_adjusted','bars':[{'date':day,'close':value} for day,value in zip(days,[100,95,100,80,72,71])]}


def test_largest_days_cover_entire_holding_period_and_keep_dates():
    result=extreme_days(price_history(),'2026-06-18','2026-06-25')
    assert result['complete']
    assert [r['date'] for r in result['events']]==['2026-06-23','2026-06-24','2026-06-25']
    assert result['purchase_day_excluded']
    assert result['events'][0]['return']==pytest.approx(-.2)


@pytest.mark.parametrize('defect',['missing','duplicate','unadjusted','no_calendar'])
def test_incomplete_market_history_cannot_claim_extreme_days(defect):
    history=price_history()
    if defect=='missing':history['bars'].pop(2)
    if defect=='duplicate':history['bars'].append(history['bars'][0])
    if defect=='unadjusted':history['adjustment']='raw'
    if defect=='no_calendar':history['sessions']=[]
    result=extreme_days(history,'2026-06-18','2026-06-25')
    assert not result['complete'] and not result['events']


def test_unknown_purchase_date_does_not_call_market_research():
    registry=Registry();register_account(registry,portfolio_source=lambda:{'positions':positions()})
    register_portfolio_workflows(registry,fill_source=lambda ticker:{'complete':True,'fills':[]})
    registry.holding_news=lambda *args:pytest.fail('No invented holding window')
    result=registry.handlers['account.position_analysis']({'ticker':'ARM'},SimpleNamespace(run_id='test'))
    assert not result.metadata['holding_period']['verified']
    assert '近三个月' in result.metadata['deterministic_answer'] and len(result.evidence)==1


def test_five_turn_workflow_keeps_subject_metric_and_missing_date():
    a=build_demo_agent()
    register_account(a.registry,portfolio_source=lambda:{'positions':positions()})
    register_portfolio_workflows(a.registry,fill_source=lambda ticker:{'complete':True,'fills':[]})
    a.registry.register('market.performance',lambda args,ctx:ToolEnvelope('market.performance',ResultStatus.COMPLETED,as_of='2026-09-11',metrics={'returns':{'1d':{'ARM':.01,'QCOM':-.02,'MRVL':-.04}[args['ticker']]}}))
    meanings=[SemanticIntent(kind='research',portfolio_scope=True,wants=['ranking'],rank='low'),SemanticIntent(kind='research',portfolio_followup='explain_position',wants=['attribution'],refers_back=True),SemanticIntent(kind='research',portfolio_followup='rerank',portfolio_metric='daily_return',scope='today'),SemanticIntent(kind='research',portfolio_followup='rerank',portfolio_metric='unrealized_amount'),SemanticIntent(kind='research',portfolio_followup='explain_position',wants=['attribution'])]
    a.brain.classify=lambda *args:meanings.pop(0)
    planner=ModelBrain(None)
    a.brain.plan=planner.plan
    try:
        results=[a.run(question,session_id='five-turn') for question in ['我的持仓哪只亏损比例最大？','为什么亏这么多？','那今天跌得最多的是哪只？','按亏损金额重新排名。','只解释我买入之后的变化。']]
        assert results[0].results[0].metadata['portfolio_context']['ticker']=='ARM'
        assert results[1].request.entities==('ARM',) and results[1].plan.tasks[0].capability=='account.position_analysis'
        assert results[2].results[0].metadata['portfolio_context']['metric']=='daily_return'
        assert results[3].results[0].metadata['portfolio_context']['ticker']=='MRVL'
        assert results[4].request.entities==('MRVL',)
        assert all(r.verification.ok for r in results),[r.to_dict() for r in results]
    finally:a.store.close()


def test_ranged_citation_is_rejected_even_when_its_numbers_exist():
    a=build_demo_agent()
    state={'plan':{'answer_mode':'research_grounded'},'evidence':[{'id':'inference-ARM-0','entity':'ARM','claim':'Numbers 0 and 2'}],'results':[]}
    try:
        _,report=a._report('Numbers 0 and 2 [inference-ARM-0~2]',state,RunContext('test',time.monotonic()+5))
        assert not report['ok'] and any('Invalid citation' in row for row in report['warnings'])
    finally:a.store.close()


def test_classifier_receives_canonical_portfolio_context_and_wants_enum():
    from v2.agent_v2.intent import WANTS
    brain=ModelBrain(None)
    received={}
    def capture(schema,system,payload,*args):
        received.update(payload)
        return SemanticIntent(kind='research')
    brain.structured=capture
    context={'ticker':'ARM','metric':'unrealized_amount'}
    brain.classify('为什么？',{'portfolio_context':context},None)
    assert received['history']['portfolio_context']==context
    assert SemanticIntent.model_json_schema()['properties']['wants']['items']['enum']==list(WANTS)


def test_missing_daily_date_is_not_comparable():
    rows,missing=rank_positions(positions(),'daily_return',daily={'ARM':{'value':-.2}})
    assert not rows and 'ARM' in missing


def test_news_failure_keeps_reconciled_holding_events(monkeypatch):
    from datetime import date
    from v2.agent_v3 import portfolio_analysis
    class FixtureDate(date):
        @classmethod
        def today(cls):
            return cls(2026, 9, 13)
    # This test's frozen price/calendar data ends on 2026-09-13.
    monkeypatch.setattr(portfolio_analysis, "date", FixtureDate)
    from v2.agent_v2.models import NormalizedRequest
    from v2.agent_v3.portfolio_acceptance import fixture_history
    registry=Registry();register_account(registry,portfolio_source=lambda:{'positions':positions()})
    register_portfolio_workflows(registry,fill_source=lambda ticker:{'complete':True,'fills':[fill('buy',2,100,'2026-09-01')]},history_source=fixture_history)
    def unavailable(*args):raise RuntimeError('injected unavailable')
    registry.holding_news=unavailable
    context=SimpleNamespace(run_id='test',allow_web=True,request=NormalizedRequest('为什么','为什么'),v3_run=RunContext('test',time.monotonic()+10))
    result=registry.handlers['account.position_analysis']({'ticker':'ARM'},context)
    assert result.metadata['holding_period']['verified']
    assert result.metadata['holding_events']['complete']
    assert any(item.id=='holding-day-2026-09-08' for item in result.evidence)
    assert any('新闻调查未完成' in note for note in result.limitations)
    assert all(not row['candidate_evidence_ids'] for row in result.metadata['event_coverage'])


def test_stale_calendar_cannot_claim_complete_holding_history():
    assert not extreme_days(price_history(),'2026-06-18','2026-09-13')['complete']


def test_scope_audit_failure_is_not_reported_as_verified():
    agent=build_demo_agent()
    def broken(*args):raise ValueError('injected invalid model output')
    agent.brain.audit_portfolio=broken
    state={'plan':{'answer_mode':'research_grounded'},'evidence':[],'results':[{'capability':'account.position_analysis','status':'partial_data','metadata':{'holding_period':{'verified':True}}}]}
    try:
        _,report=agent._report('证据不足。',state,RunContext('test',time.monotonic()+5))
        assert not report['ok'] and any('scope audit unavailable' in row for row in report['warnings'])
    finally:agent.store.close()


def test_each_market_event_gets_an_exact_date_search():
    from v2.agent_v3.news_research import event_search_queries
    dates=['2026-06-05','2026-06-23','2026-07-24']
    queries=event_search_queries('ARM',dates)
    assert len(queries)==3 and all(day in query for day,query in zip(dates,queries))
    assert len(event_search_queries('ARM',[*dates,dates[0]]))==3
