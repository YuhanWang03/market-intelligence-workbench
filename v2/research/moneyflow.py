"""Adapt the existing /flow detector, chart and narrator for research results."""
import base64
import math
import os
from threading import Lock

from v2.moneyflow.models import DivergenceConfig
from v2.moneyflow.detector import read_axes, detect_divergence

_CHART_LOCK = Lock()


def build_flow_analysis(ticker, prices, *, narrate_fn=None, chart_fn=None):
    cfg = DivergenceConfig()
    base = {'version': 1, 'config': cfg.model_dump(), 'reading': None, 'signal': None,
            'chart_png': None, 'chart_status': 'NOT_AVAILABLE', 'narration_status': 'NOT_TRIGGERED',
            'llm_tokens': 0, 'bar_count': len(prices), 'as_of': str(prices[-1].time) if prices else None,
            'data_note': '复用本次研究的日线 OHLCV；/flow 命令使用 Financial Datasets，数据源或截止日期不同可能导致结果差异。'}
    # Do not fill missing prices/volume with zeros or bridge invalid bars.
    valid = all(all(isinstance(getattr(p, key, None), (int, float)) and math.isfinite(getattr(p, key))
                    for key in ('high', 'low', 'close', 'volume')) and p.close > 0 and p.volume >= 0
                and p.high >= p.close >= p.low for p in prices)
    if not valid:
        return {**base, 'status': 'INVALID_DATA', 'reason': '日线含缺失或无效价量数据，未生成背离结论。'}
    if len(prices) < cfg.min_history:
        return {**base, 'status': 'INSUFFICIENT_DATA', 'reason': f'至少需要 {cfg.min_history} 个交易日日线，本次仅 {len(prices)} 个。'}
    reading = read_axes(ticker, prices, cfg)
    if reading is None:
        return {**base, 'status': 'INSUFFICIENT_DATA', 'reason': '有效成交量或指标历史不足，无法判断背离。'}
    signal = detect_divergence(ticker, prices, cfg)
    base.update(status='SIGNAL' if signal else 'NO_SIGNAL', reading=reading.model_dump(),
                reason='发现潜在量价背离，非真实账户资金流证据。' if signal else '本次未触发 /flow 的吸筹或派发背离规则。')
    if signal:
        base['narration_status'] = 'NOT_CONFIGURED'
        if narrate_fn is not None or os.environ.get('DEEPSEEK_API_KEY'):
            try:
                if narrate_fn is None:
                    from v2.moneyflow.narrator import narrate
                    narrate_fn = narrate
                notes, tokens = narrate_fn([signal])
                note = notes.get(ticker) or {}
                signal.bull = note.get('bull') if isinstance(note.get('bull'), str) else ''
                signal.bear = note.get('bear') if isinstance(note.get('bear'), str) else ''
                base['llm_tokens'] = tokens
                base['narration_status'] = 'AVAILABLE' if signal.bull and signal.bear else 'UNAVAILABLE'
            except Exception:
                base['narration_status'] = 'UNAVAILABLE'
        base['signal'] = signal.model_dump()
    try:
        if chart_fn is None:
            from v2.moneyflow.chart import render_moneyflow_chart
            chart_fn = render_moneyflow_chart
        with _CHART_LOCK:
            png = chart_fn(ticker, prices, cfg, signal=signal)
        if png:
            base['chart_png'] = base64.b64encode(png).decode('ascii')
            base['chart_status'] = 'AVAILABLE'
        else:
            base['chart_status'] = 'UNAVAILABLE'
    except Exception:
        base['chart_status'] = 'UNAVAILABLE'
    return base
