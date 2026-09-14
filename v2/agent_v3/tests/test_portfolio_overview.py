import time
from types import SimpleNamespace
import pytest

from v2.agent_v3.adapters import register_account
from v2.agent_v3.brain import ModelBrain
from v2.agent_v3.context import RunContext, RunStopped
from v2.agent_v3.contracts import SemanticIntent
from v2.agent_v3.graph import AgentV3, AgentV3Config
from v2.agent_v3.tools import Registry
from v2.agent_v2.models import ResultStatus, ToolEnvelope


def snapshot():
    return {'account': {'equity': 20000, 'cash': 5000, 'buying_power': 60000}, 'positions': [
        {'symbol': f'T{i}', 'qty': 10, 'avg_entry_price': 100, 'current_price': 100+i,
         'market_value': (100+i)*10, 'unrealized_pl': i*10, 'unrealized_pl_pct': i/100}
        for i in range(12)]}


def registry(data=None):
    r = Registry()
    register_account(r, portfolio_source=lambda: snapshot() if data is None else data)
    return r


def calculate(data=None):
    return registry(data).handlers['account.overview']({}, SimpleNamespace(run_id='fixture'))


def test_all_twelve_holdings_are_summed_without_cash_or_buying_power():
    result = calculate()
    assert result.metrics == {'position_count':12,'net_market_value':12660.0,'gross_exposure':12660.0,'unrealized_pnl':660.0}
    assert len(result.metadata['priority_tickers']) == 3
    answer = result.metadata['portfolio_overview_answer']
    assert all(f'T{i}：' in answer for i in range(12))
    assert '购买力（不可加作账户资产）' in answer
    assert '12,660.00' in answer and '5,000.00' in answer
    assert all(e.metadata['source_evidence_ids'] for e in result.evidence)
    from v2.agent_v2.verification import verify_answer
    from v2.agent_v2.models import AnswerMode
    report=verify_answer(answer,result.evidence,answer_mode=AnswerMode.TOOL_GROUNDED,results=[result])
    assert report.ok, report


def test_missing_price_does_not_become_zero_or_partial_total():
    data=snapshot()
    data['positions'][0].update(market_value=None,current_price=None,unrealized_pl=None)
    result=calculate(data)
    assert result.status == ResultStatus.PARTIAL_DATA
    assert result.metrics['net_market_value'] is None
    assert result.metrics['unrealized_pnl'] is None
    assert '数据缺失' in result.metadata['portfolio_overview_answer']


def test_empty_and_missing_positions_are_distinct():
    assert calculate({'positions':[]}).metrics['net_market_value']==0
    missing=calculate({'account':{}})
    assert missing.status == ResultStatus.PARTIAL_DATA
    assert missing.metrics['net_market_value'] is None
    assert '不能认定为空仓' in ' '.join(missing.limitations)


def test_short_exposure_and_loss_rank_use_separate_denominators():
    data={'positions':[
        {'symbol':'LONG','qty':10,'avg_entry_price':100,'current_price':90,'market_value':900,'unrealized_pl':-100,'unrealized_pl_pct':-.1},
        {'symbol':'SHORT','qty':-2,'side':'short','avg_entry_price':100,'current_price':120,'market_value':-240,'unrealized_pl':-40,'unrealized_pl_pct':-.2}]}
    result=calculate(data)
    assert result.metrics['gross_exposure']==1140
    assert result.metrics['net_market_value']==660
    assert result.metadata['priority_tickers']==['LONG','SHORT']
    assert '空头' in result.metadata['portfolio_overview_answer']


class PortfolioBrain(ModelBrain):
    def __init__(self, failure):
        super().__init__(None)
        self.failure=failure
    def classify(self, text, history, run):
        return SemanticIntent(kind='research',portfolio_scope=True,wants=['overview'])
    def structured(self,*args,**kwargs):
        raise AssertionError('Portfolio plan must not depend on model task planning')
    def draft(self, state, registry, run, **kwargs):
        if self.failure=='timeout':
            raise RunStopped('deadline')
        if self.failure=='error':
            raise ValueError('injected model failure')
        return '无依据金额 999999999 美元。[missing]'
    def judge(self, rows, run):
        return {}


@pytest.mark.parametrize('failure',['timeout','error','invalid'])
@pytest.mark.parametrize('question',['分析一下我的持仓','看看账户配置和盈亏情况'])
def test_model_failures_preserve_full_readable_portfolio(failure,question):
    r=registry()
    def unavailable(*args):
        raise RuntimeError('injected market failure')
    r.register('market.performance',unavailable)
    result=AgentV3(registry=r,brain=PortfolioBrain(failure),config=AgentV3Config(max_seconds=90,debate=False)).run(question)
    assert result.answer and '12,660.00' in result.answer
    assert all(f'T{i}：' in result.answer for i in range(12))
    assert '"symbol"' not in result.answer and '999999999' not in result.answer
    assert result.plan.tasks[0].capability=='account.overview'
    assert result.plan.tasks[1].fan_out['max']==3
    assert result.plan.tasks[1].required is False
    assert '成功 0 项' in result.answer


def test_each_request_is_not_silently_reduced_to_three():
    from v2.agent_v2.models import RouteDecision, RouteKind, NormalizedRequest
    intent=SemanticIntent(kind='research',portfolio_scope=True,wants=['overview'],each=True).domain()
    plan=ModelBrain(None).plan(NormalizedRequest('each','each'),RouteDecision(RouteKind.RESEARCH,('account','research'),'fixture',intent=intent),registry(),RunContext('test',time.monotonic()+5))
    assert all(t.capability!='account.overview' for t in plan.tasks)


def test_empty_portfolio_does_not_call_optional_market():
    r=registry({'positions':[]})
    def forbidden(*args):
        raise AssertionError('No market lookup for an empty portfolio')
    r.register('market.performance',forbidden)
    result=AgentV3(registry=r,brain=PortfolioBrain('timeout'),config=AgentV3Config(debate=False)).run('分析我的持仓')
    assert '当前没有持仓' in result.answer
    assert all(not row.errors for row in result.results)
