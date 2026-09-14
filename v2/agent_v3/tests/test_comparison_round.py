from dataclasses import replace
import time
from v2.agent_v2.models import EvidenceItem,ToolEnvelope,ResultStatus
from v2.agent_v3.research import attach_comparison_sources,audit_research_sources
from v2.agent_v3.news_research import Extraction,SourceFact,located_facts
from v2.agent_v3.contracts import SemanticIntent
from v2.agent_v3.demo import build_demo_agent
from v2.agent_v3.brain import ModelBrain
from v2.agent_v3.context import RunContext
from v2.agent_v3.contracts import plain


def comparison():
    a=EvidenceItem('a','MU','MU margin',metric='margin',unit='percent',period='2026-Q2',source_url='https://example.com/a',metadata={'accounting_basis':'GAAP','measurement_window':'quarter','period_start':'2026-04-01','period_end':'2026-06-30'})
    b=replace(a,id='b',entity='SNDK',source_url='https://example.com/b')
    c=EvidenceItem('c','MU,SNDK','Comparison',metadata={'comparison':True,'from':['a','b']})
    return ToolEnvelope('research.compare',ResultStatus.COMPLETED,evidence=[a,b,c])


def test_same_metric_name_does_not_make_periods_or_accounting_comparable():
    good=attach_comparison_sources(comparison())
    assert good.evidence[-1].metadata['financial_comparison_allowed']
    for changes in [{'period':'2026-Q1'},{'unit':''},{'metadata':{**comparison().evidence[0].metadata,'accounting_basis':'non-GAAP'}},{'metadata':{**comparison().evidence[0].metadata,'period_end':'2026-07-31'}}]:
        bad=comparison();bad.evidence[1]=replace(bad.evidence[1],**changes)
        result=audit_research_sources(attach_comparison_sources(bad))
        assert not result.evidence[-1].metadata['financial_comparison_allowed']
        assert result.status==ResultStatus.PARTIAL_DATA and result.metadata['answer_constraints']
        assert result.evidence[-1].metadata['exclude_from_comparison']


def test_unaligned_comparison_citation_is_rejected_even_with_matching_numbers():
    result=comparison();result.evidence[0]=replace(result.evidence[0],period='')
    result=audit_research_sources(attach_comparison_sources(result))
    a=build_demo_agent()
    try:
        _,report=a._report('同口径比较 [c]',{'plan':{'answer_mode':'research_grounded'},'evidence':[plain(row) for row in result.evidence],'results':[plain(result)]},RunContext('test',time.monotonic()+10))
        assert not report['ok'] and any('Unaligned financial' in row for row in report['warnings'])
    finally:a.store.close()


def test_reported_session_requires_an_exact_body_anchor():
    source={'id':'s0','blocks':['Shares fell 5% before market open.'],'published':'2026-09-11','url':'https://example.com','title':'report','domain':'example.com'}
    fact=SourceFact(source='s0',blocks=[0],summary='Shares fell 5%',trading_session='premarket',time_anchor='before market open')
    assert located_facts(Extraction(facts=[fact]),[source])[0]['trading_session']=='premarket'
    bad=fact.model_copy(update={'time_anchor':'after market close','trading_session':'afterhours'})
    assert located_facts(Extraction(facts=[bad]),[source])[0]['trading_session']=='unknown'


def test_broad_us_market_does_not_ask_for_a_ticker_or_reuse_prior_stock():
    a=build_demo_agent();a.brain.classify=lambda *args:SemanticIntent(kind='research',market_scope='us_broad',clarification='Which ticker?',tickers=['ORCL'],portfolio_followup='rerank')
    a.brain.plan=ModelBrain(None).plan
    a.registry.register('market.performance',lambda args,ctx:ToolEnvelope('market.performance',ResultStatus.COMPLETED))
    try:
        result=a.run('美股大盘最近一周表现如何？',session_id='broad')
        assert result.request.entities==('SPY','QQQ','DIA')
        assert len(result.plan.tasks)==3 and result.status.value!='waiting_clarification'
    finally:a.store.close()
