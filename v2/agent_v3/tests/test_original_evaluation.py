from copy import deepcopy
from types import SimpleNamespace

from v2.agent_eval_capture import build_inventory
from v2.agent_original_comparison import FrozenTools, run_episode


def episode():
    return {'id':'test-original','test':'original_test','turns':[{
        'text':'查询 NVDA 行情','session_id':'original-session','allow_web':False,
        'intent':{'kind':'lookup','wants':['performance'],'tickers':['NVDA']},
        'result':{'status':'completed','request':{'entities':['NVDA']},'plan':{
            'objective':'查询 NVDA 行情','route':'fast_lookup','budget':'standard','answer_mode':'tool_grounded',
            'tasks':[{'id':'quote','capability':'market.performance','arguments':{'ticker':'NVDA'}}]}},
        'tool_records':[{'task':{'capability':'market.performance','arguments':{'ticker':'NVDA'}},
            'result':{'capability':'market.performance','status':'completed','subject':'NVDA',
                      'evidence':[{'id':'price','entity':'NVDA','claim':'NVDA close 120 USD',
                                   'metric':'close','value':120,'unit':'USD','source_id':'fixture'}]}}]}]}


def test_versions_get_identical_independent_fixtures():
    case=episode()
    left,right=FrozenTools(case),FrozenTools(case)
    context=SimpleNamespace(allow_mutations=False)
    a=left.handler('market.performance')({'ticker':'NVDA'},context)
    a.evidence[0].metadata['tampered']=True
    b=right.handler('market.performance')({'ticker':'NVDA'},context)
    assert 'tampered' not in b.evidence[0].metadata
    assert b.evidence[0].value==120


def test_missing_arguments_are_an_infrastructure_gap_not_live_fallback():
    bank=FrozenTools(episode())
    result=bank.handler('market.performance')({'ticker':'AMD'},SimpleNamespace(allow_mutations=False))
    assert result.metadata['fixture_missing'] and not bank.calls[0]['fixture_match']


def test_both_versions_use_original_text_and_fixture_numbers():
    for version in ('v2','v3'):
        result=run_episode(version,episode(),seconds=20)
        turn=result['turns'][0]
        assert turn['input']['text']=='查询 NVDA 行情'
        assert turn['input']['session_id']=='original-session'
        assert turn['fixture_complete'] and turn['common_checks']['no_unapproved_mutation']
        assert turn['result']['evidence'][0]['value']==120
        assert turn['semantic_quality']=='not_evaluated_contract_mode'


def test_capture_keeps_exact_text_and_separates_blank_sessions(tmp_path):
    source=tmp_path/'test_agent_v2.py'
    source.write_text('def test_example():\n    assert True\n',encoding='utf-8')
    rows=[]
    for i,text in enumerate(['  原题?  ','确认']):
        rows.append({'entry':'agent','test':'test_example','object':1,'session_id':'','sequence':i,
                     'text':text,'source_stack':[{'function':'test_example','line':2}],
                     'result':{'run_id':str(i)}})
    capture=SimpleNamespace(rows=rows,outcomes=[])
    inventory=build_inventory(source,capture,0)
    assert len(inventory['episodes'])==2
    assert inventory['episodes'][0]['turns'][0]['text']=='  原题?  '
    assert inventory['coverage'][0]['original_assertions']==['assert True']


def test_original_confirmation_uses_v3_resume_and_only_simulates_one_write():
    base=episode()['turns'][0]
    plan={'objective':'把 NVDA 加入关注列表','route':'command','budget':'direct','answer_mode':'tool_grounded',
          'requires_confirmation':True,'tasks':[{'id':'write','capability':'state.mutate',
          'arguments':{'operation':'watchlist.add','payload':{'ticker':'NVDA'}}}]}
    first={**deepcopy(base),'text':'把 NVDA 加入关注列表','intent':{'kind':'command','command':{'operation':'watchlist.add','payload':{'ticker':'NVDA'}}},
           'tool_records':[],'result':{'status':'waiting_confirmation','request':{'entities':['NVDA']},'plan':plan}}
    second={**deepcopy(first),'text':'确认','result':{'status':'completed','request':{'entities':['NVDA']},'plan':plan},
            'tool_records':[{'task':plan['tasks'][0],'result':{'capability':'state.mutate','status':'completed',
                'evidence':[{'id':'write-ok','entity':'NVDA','claim':'已加入关注列表'}]}}]}
    result=run_episode('v3',{'id':'confirmation','test':'confirmation','turns':[first,second]},seconds=20)
    assert result['turns'][0]['result']['status']=='waiting_confirmation'
    assert result['turns'][1]['common_checks']['actual_mutation_calls']==1
    assert result['turns'][1]['common_checks']['no_unapproved_mutation']


def test_replay_uses_initial_plan_not_the_post_fallback_plan():
    from v2.agent_original_comparison import original_plan
    turn=episode()['turns'][0]
    turn['initial_plan']=deepcopy(turn['result']['plan'])
    turn['result']['plan']['tasks'].append({'id':'web-fallback','capability':'web.research','arguments':{'query':'q'}})
    assert [task.id for task in original_plan(turn).tasks]==['quote']
