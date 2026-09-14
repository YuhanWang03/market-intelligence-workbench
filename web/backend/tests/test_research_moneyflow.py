from datetime import date, timedelta
from types import SimpleNamespace
import base64

from v2.research.moneyflow import build_flow_analysis
from v2.moneyflow.detector import detect_divergence, read_axes
from v2.moneyflow.models import DivergenceConfig


def bars(kind='flat', size=60):
    return [SimpleNamespace(time=str(date(2026, 1, 1) + timedelta(days=i)),
            close=100., high=100.1 if kind == 'inflow' else 100.5,
            low=99. if kind == 'inflow' else 99.5, volume=1000) for i in range(size)]


def test_short_and_invalid_are_not_no_signal():
    assert build_flow_analysis('TEST', bars(size=20))['status'] == 'INSUFFICIENT_DATA'
    data = bars()
    data[-1].volume = float('nan')
    assert build_flow_analysis('TEST', data)['status'] == 'INVALID_DATA'
    for row in data:
        row.volume = 0
    assert build_flow_analysis('TEST', data)['status'] == 'INSUFFICIENT_DATA'


def test_no_signal_skips_narrator_and_keeps_chart():
    def forbidden(*args):
        raise AssertionError('No LLM call allowed')
    result = build_flow_analysis('TEST', bars(), narrate_fn=forbidden, chart_fn=lambda *args, **kwargs: b'png')
    assert result['status'] == 'NO_SIGNAL'
    assert result['reading'] == read_axes('TEST', bars(), DivergenceConfig()).model_dump()
    assert result['narration_status'] == 'NOT_TRIGGERED'
    assert base64.b64decode(result['chart_png']) == b'png'


def test_signal_matches_flow_and_narration_failure_is_nonfatal():
    data = bars('inflow')
    result = build_flow_analysis('TEST', data, narrate_fn=lambda signals: ({'TEST': {'bull': '疑似吸筹', 'bear': '或是假信号'}}, 20), chart_fn=lambda *args, **kwargs: b'png')
    expected = detect_divergence('TEST', data, DivergenceConfig())
    assert result['status'] == 'SIGNAL'
    assert result['signal']['kind'] == expected.kind
    assert result['signal']['strength'] == expected.strength
    assert result['signal']['bull'] == '疑似吸筹'
    assert result['llm_tokens'] == 20
    def failed(*args, **kwargs):
        raise RuntimeError('secret')
    result = build_flow_analysis('TEST', data, narrate_fn=failed, chart_fn=failed)
    assert result['status'] == 'SIGNAL'
    assert result['narration_status'] == result['chart_status'] == 'UNAVAILABLE'
    assert 'secret' not in str(result)


def test_real_chart_returns_png():
    result = build_flow_analysis('TEST', bars())
    assert base64.b64decode(result['chart_png']).startswith(b'\x89PNG')
