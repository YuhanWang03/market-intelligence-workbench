from types import SimpleNamespace

from v2.lateral.models import Label
from v2.research.store import ResearchStore
from v2.research.relationship_evidence import present_relationships


def test_financial_failure_preserves_edges_counts_and_later_candidates(monkeypatch, tmp_path):
    from v2.lateral import orchestrator as o
    from v2.lateral import LATERAL_FILTERS
    from v2.research.services import SupplyChainDataService
    from v2.data.client import ProviderRequestError
    visited = []
    class FD:
        def get_company_facts(self, ticker):
            return SimpleNamespace(sector='Technology')
        def get_financial_metrics(self, ticker, *args, **kwargs):
            if ticker == 'BAD':
                raise ProviderRequestError('EMPTY_DATA', '/financial-metrics/', 404)
            return []
    monkeypatch.setattr(o, 'discover', lambda seeds: ([(t, Label(seed='MU', category='supplier', reason='candidate')) for t in ('BAD', 'GOOD')], 123))
    def relation(n):
        n.labels[0].evidence_status = 'CO_MENTION'
        n.labels[0].evidence_url = 'https://example.com/' + n.ticker
        return 1
    monkeypatch.setattr(o, 'verify_relation', relation)
    def candidate(ticker, fd, *args, **kwargs):
        visited.append(ticker)
        fd.get_financial_metrics(ticker)
        return None
    monkeypatch.setattr(o, 'build_candidate', candidate)
    result = o.run_lateral_expansion(['MU'], set(), FD(), LATERAL_FILTERS, price_source=SimpleNamespace())
    assert visited == ['BAD', 'GOOD']
    assert len(result.neighbors) == 2 and all(n.exists for n in result.neighbors)
    assert result.llm_tokens == 123 and result.tavily_calls == 2
    assert result.api_calls == 4  # Two identity calls and two metrics attempts.
    assert result.candidate_errors == [{'ticker': 'BAD', 'stage': '财务筛选', 'type': 'EMPTY_DATA'}]
    assert result.neighbors[0].labels[0].evidence_url
    import v2.lateral
    monkeypatch.setattr(v2.lateral, 'run_lateral_expansion', lambda **kwargs: result)
    store = ResearchStore(tmp_path / 'research.db')
    service = SupplyChainDataService()
    service.store = store
    payload = service.collect('MU', FD())
    assert len(store.relationships('MU')) == 2
    assert payload['candidate_errors'] and payload['warnings']


def test_identity_failure_isolated_and_no_secret_in_diagnostics(monkeypatch):
    from v2.lateral import orchestrator as o, LATERAL_FILTERS
    monkeypatch.setattr(o, 'discover', lambda seeds: ([(t, Label(seed='MU', category='customer', reason='candidate')) for t in ('BAD', 'GOOD')], 99))
    class FD:
        def get_company_facts(self, ticker):
            if ticker == 'BAD':
                raise RuntimeError('secret metadata')
            return SimpleNamespace(sector='Technology')
    monkeypatch.setattr(o, 'verify_relation', lambda n: 0)
    monkeypatch.setattr(o, 'build_candidate', lambda *args, **kwargs: None)
    result = o.run_lateral_expansion(['MU'], set(), FD(), LATERAL_FILTERS, price_source=SimpleNamespace())
    assert result.neighbors[1].exists
    assert result.api_calls == 2
    assert 'secret' not in str(result.model_dump())


def test_empty_legacy_result_never_shows_full_coverage(tmp_path):
    source = {'ticker': 'MU', 'modules': {'supply_chain': {'status': 'PARTIAL_DATA', 'completeness': 1,
              'metrics': {'relationships': 0}, 'details': {'relationships': []}}}}
    result = present_relationships(source, ResearchStore(tmp_path / 'research.db'))
    module = result['modules']['supply_chain']
    assert module['completeness'] == 0
    assert '不代表公司没有产业关系' in module['summary']
    assert source['modules']['supply_chain']['completeness'] == 1
