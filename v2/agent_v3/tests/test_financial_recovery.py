from copy import deepcopy
from v2.agent_v3.financial_facts import financial_evidence
from v2.agent_v3.news_research import uncovered_event_dates


def payload():
    def concept(value,start='2026-01-01',end='2026-03-31'):
        return {'units':{'USD':[{'start':start,'end':end,'filed':'2026-05-01','accn':'0000000001-26-000001','form':'10-Q','val':value}]}}
    return {'cik':1,'facts':{'us-gaap':{
        'Revenues':concept(100),'GrossProfit':concept(40),'OperatingIncomeLoss':concept(20),
        'NetCashProvidedByUsedInOperatingActivities':concept(50),'PaymentsToAcquirePropertyPlantAndEquipment':concept(15)}}}


def test_original_components_recompute_with_provenance():
    items,gaps=financial_evidence(payload(),'TEST','2026-09-13')
    assert {row.metric:row.value for row in items}=={'gross_margin':.4,'operating_margin':.2,'free_cash_flow':35}
    assert not gaps
    assert all(len(row.metadata['components'])==2 and row.source_url.endswith('-index.html') for row in items)


def test_no_future_filings_ytd_join_or_cross_accession_join():
    assert not financial_evidence(payload(),'TEST','2026-04-01')[0]
    raw=payload()
    raw['facts']['us-gaap']['PaymentsToAcquirePropertyPlantAndEquipment']['units']['USD'][0]['end']='2026-06-30'
    items,gaps=financial_evidence(raw,'TEST','2026-09-13')
    assert 'free_cash_flow' not in {row.metric for row in items}
    raw=payload()
    raw['facts']['us-gaap']['GrossProfit']['units']['USD'][0]['accn']='0000000001-26-000002'
    assert 'gross_margin' not in {row.metric for row in financial_evidence(raw,'TEST','2026-09-13')[0]}


def test_conflicting_revenue_tags_are_not_silently_chosen():
    raw=payload()
    alternative=deepcopy(raw['facts']['us-gaap']['Revenues'])
    alternative['units']['USD'][0]['val']=110
    raw['facts']['us-gaap']['SalesRevenueNet']=alternative
    items,gaps=financial_evidence(raw,'TEST','2026-09-13')
    assert {row.metric for row in items}=={'free_cash_flow'}
    assert any(row['reason']=='conflicting_taxonomy_values' for row in gaps)


def test_news_background_does_not_close_missing_driver_date():
    facts=[{'event_date':'2026-06-05'},{'event_date':'2026-06-23'}]
    checks=[{'index':i,'supported':True,'relevant':True,'event_date_supported':True,'role':role} for i,role in enumerate(['reported_driver','context'])]
    assert uncovered_event_dates(facts,{'facts':checks},['2026-06-05','2026-06-23'])==['2026-06-23']


def test_shared_adapter_preserves_explicit_financial_contract():
    from v2.agent_v2.adapters.research import _evidence
    item=_evidence({'ticker':'TEST','evidence_index':[{'id':'test','metrics':{'gross_margin':.4},'unit':'ratio','data_period':'Q1','period_start':'2026-01-01','period_end':'2026-03-31','accounting_basis':'US-GAAP','measurement_window':'quarter'}]})[0]
    assert item.unit=='ratio' and item.metadata['period_start']=='2026-01-01'
