from v2.research.relationship_evidence import assess, present_relationships
from v2.research.store import ResearchStore


def hit(text, url='https://example.com/report'):
    return {'content': text, 'title': 'Company report', 'url': url}


def test_co_mention_and_ticker_boundaries():
    assert assess('NVDA', 'AMD', 'supplier', hit('NVDA and AMD stock prices'))['status'] == 'CO_MENTION'
    assert assess('NVDA', 'ARM', 'supplier', hit('NVDA and FARM prices')) is None
    assert assess('NVDA', 'AMD', 'supplier', hit('AMD supplies NVDA.', 'javascript:alert(1)')) is None


def test_direction_negation_and_uncertainty():
    assert assess('NVDA', 'TSM', 'supplier', hit('TSM supplies NVDA.'))['status'] == 'EVIDENCE_FOUND'
    for text in ['NVDA supplies TSM.', 'TSM does not supply NVDA.', 'TSM may supply NVDA.', 'TSM previously supplies NVDA.']:
        assert assess('NVDA', 'TSM', 'supplier', hit(text))['status'] == 'CO_MENTION'
    assert assess('NVDA', 'TSM', 'customer', hit('TSM supplies NVDA.'))['status'] == 'CO_MENTION'


def test_verification_is_per_edge(monkeypatch):
    import sys
    from types import SimpleNamespace
    from v2.lateral.models import Neighbor, Label
    from v2.lateral.verify import verify_relation
    monkeypatch.setenv('TAVILY_API_KEY', 'test')
    client = SimpleNamespace(search=lambda **kwargs: {'results': [hit('TSM supplies NVDA.')]})
    monkeypatch.setitem(sys.modules, 'tavily', SimpleNamespace(TavilyClient=lambda **kwargs: client))
    neighbor = Neighbor(ticker='TSM', labels=[Label(seed='NVDA', category='supplier', reason='candidate'), Label(seed='NVDA', category='customer', reason='candidate')])
    assert verify_relation(neighbor) == 2
    assert neighbor.labels[0].evidence_status == 'EVIDENCE_FOUND'
    assert neighbor.labels[1].evidence_status == 'CO_MENTION'
    assert not neighbor.relation_verified


def test_restore_links_without_certifying_legacy(tmp_path):
    store = ResearchStore(tmp_path / 'research.db')
    relation = {'source_ticker': 'NVDA', 'target_ticker': 'TSM', 'relationship_type': 'SUPPLIER', 'status': 'VERIFIED', 'confidence': .9}
    ident = store.upsert_relationship(relation, [{'url': 'https://example.com/report', 'provider': 'Tavily'}])
    source = {'ticker': 'NVDA', 'modules': {'supply_chain': {'details': {'relationships': [{'target_company': 'TSM', 'relationship_type': 'supplier', 'verified': True}]}}}}
    result = present_relationships(source, store)
    row = result['modules']['supply_chain']['details']['relationships'][0]
    assert row['id'] == ident and row['source'] == 'https://example.com/report'
    assert row['evidence_status'] == 'LEGACY' and not row['verified']
    assert source['modules']['supply_chain']['details']['relationships'][0]['verified']
    assert store.relationships('NVDA')[0]['sources']
    # Rechecking the same URL updates its evidence rather than retaining stale metadata.
    store.upsert_relationship({**relation, 'status': 'EVIDENCE_FOUND'}, [{'url': 'https://example.com/report', 'provider': 'Tavily-v2:EVIDENCE_FOUND', 'evidence_text': 'TSM supplies NVDA.'}])
    row = present_relationships(source, store)['modules']['supply_chain']['details']['relationships'][0]
    assert row['evidence_status'] == 'EVIDENCE_FOUND'
    assert row['evidence_text'] == 'TSM supplies NVDA.'
    assert not row['verified']
