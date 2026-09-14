from copy import deepcopy

import pytest

from v2.research.expectations import classify_statement, extract_statements, prepare_expectations


@pytest.mark.parametrize('text', [
    'Actual future results may differ materially from expected revenue.',
    'We assume no obligation to update forward-looking statements about revenue.',
    'We believe our facilities are in good condition and suitable for our business.',
    'Statements reflect our beliefs and opinions about future demand.',
])
def test_disclaimers_are_not_guidance(text):
    assert classify_statement(text) is None


def test_growth_is_not_a_guidance_upgrade():
    row = classify_statement('We expect revenue to increase next year.')
    assert row['group'] == 'outlook'
    assert row['status'] == 'NOT_COMPARABLE'
    assert row['period_label'] == '下一年度'


def test_explicit_guidance_action_and_negation():
    row = classify_statement('We are raising our revenue guidance to $40 billion for next quarter.')
    assert row['group'] == 'guidance'
    assert row['status'] == 'RAISED'
    assert row['value'] is None  # No invented units or parsed financial values.
    assert classify_statement('We are not raising our revenue guidance for next quarter.')['status'] == 'NOT_COMPARABLE'
    assert classify_statement('We may raise our revenue guidance for next quarter.')['status'] == 'NOT_COMPARABLE'
    assert classify_statement('We reaffirm our revenue guidance for next quarter.')['status'] == 'REITERATED'


def test_customer_risk_is_separate():
    row = classify_statement('Customers may postpone purchases, affecting our revenue timing and supply expenses.')
    assert row['group'] == 'risk'
    assert row['status'] == 'RISK_CONTEXT'


def test_old_cache_reclassified_deduplicated_without_mutation():
    text = 'We expect revenue to increase next year.'
    source = {'modules': {'expectations': {'metrics': {'forward_eps': 0}, 'details': {'guidance': [
        {'evidence_text': text, 'status': 'RAISED', 'filing_date': '2026-06-01'},
        {'evidence_text': text, 'status': 'RAISED', 'filing_date': '2026-03-01'},
        {'evidence_text': 'We assume no obligation to update forward-looking statements about revenue.'},
    ]}}, 'earnings': {'details': {'history': [{'eps_surprise': 0}, {'eps_surprise': None}]}}}}
    original = deepcopy(source)
    details = prepare_expectations(source)['modules']['expectations']['details']
    assert source == original
    assert len(details['guidance']) == 1
    assert details['guidance'][0]['status'] == 'NOT_COMPARABLE'
    assert details['guidance'][0]['filing_date'] == '2026-06-01'
    assert details['guidance_quality']['filtered'] == 1
    assert details['guidance_quality']['duplicates'] == 1
    assert details['capability_matrix']['current_consensus']['status'] == 'PARTIAL_DATA'
    assert '1 个' in details['capability_matrix']['earnings_surprise_history']['reason']


def test_empty_results_do_not_advertise_data():
    details = prepare_expectations({'modules': {'expectations': {'details': {}}}})['modules']['expectations']['details']
    assert details['capability_matrix']['current_consensus']['status'] == 'UNKNOWN'
    assert details['capability_matrix']['revision_30d']['status'] == 'NOT_CONNECTED'
    assert details['guidance'] == []


def test_scan_past_introduction_and_keep_actual_guidance():
    boilerplate = 'These forward-looking statements concern revenue and are subject to risks. '
    text = boilerplate * 180 + 'We forecast revenue of $40 billion for next quarter.'
    rows = extract_statements(text, '2026-08-26', 'https://www.sec.gov/example')
    assert len(rows) == 1
    assert rows[0]['group'] == 'guidance'


def test_consensus_calendar_average_without_release_date():
    from types import SimpleNamespace
    from v2.research.consensus import collect_consensus
    client = SimpleNamespace(earnings_estimate=None, revenue_estimate=None,
                             calendar={'Earnings Average': 0, 'Revenue Average': 123})
    result = collect_consensus('TEST', client)
    assert result['eps_estimate'] == 0
    assert result['revenue_estimate'] == 123


def test_consensus_tables_same_period_no_calendar_mixing():
    import pandas as pd
    from types import SimpleNamespace
    from v2.research.consensus import collect_consensus
    client = SimpleNamespace(earnings_estimate=pd.DataFrame({'avg': [2, 10]}, index=['0q', '+1y']),
                             revenue_estimate=None, calendar={'Revenue Average': 123})
    result = collect_consensus('TEST', client)
    assert result['eps_estimate'] == 2
    assert result['revenue_estimate'] is None
    assert result['period'] == '0q'


def test_consensus_failures_do_not_expose_exception():
    from v2.research.consensus import collect_consensus
    class Failed:
        def __getattr__(self, name):
            raise RuntimeError('secret request metadata')
    result = collect_consensus('TEST', Failed())
    assert 'secret' not in str(result)
    source = {'modules': {'expectations': {'details': {'upcoming_earnings': result}}}}
    matrix = prepare_expectations(source)['modules']['expectations']['details']['capability_matrix']
    assert matrix['current_consensus']['status'] == 'FETCH_FAILED'


def test_empty_consensus_distinguished_from_failure():
    from types import SimpleNamespace
    from v2.research.consensus import collect_consensus
    result = collect_consensus('TEST', SimpleNamespace(earnings_estimate=None, revenue_estimate=None, calendar={}))
    source = {'modules': {'expectations': {'details': {'upcoming_earnings': result}}}}
    assert prepare_expectations(source)['modules']['expectations']['details']['capability_matrix']['current_consensus']['status'] == 'NO_DATA'
