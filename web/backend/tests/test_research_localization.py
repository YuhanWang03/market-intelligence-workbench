from copy import deepcopy

from v2.research import localization as loc


def test_old_snapshot_translation_preserves_evidence_and_reuses_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(loc, 'DB_PATH', tmp_path / 'translations.db')
    calls = []
    def translate(texts):
        calls.append(texts)
        return {text: '客户可能推迟采购，影响营收确认时间。' for text in texts}
    monkeypatch.setattr(loc, '_translate_batch', translate)
    source = {'modules': {
        'fundamental': {'details': {'company': {'description': 'Customers may postpone purchases.'}}},
        'sec': {'details': {'risk_factor_changes': [{'title': 'Risk factor 1', 'change_type': 'EXPANDED', 'text': 'Customers may postpone purchases.'}],
                           'sec_findings': [{'title': 'Business overview', 'summary': 'Customers may postpone purchases.'}]}},
        'expectations': {'details': {'guidance': [{'metric': 'revenue', 'status': 'REITERATED', 'guidance_type': 'QUALITATIVE_OUTLOOK', 'evidence_text': 'Customers may postpone purchases.'}]}}
    }}
    original = deepcopy(source)
    result = loc.localize_result(source)
    assert source == original
    risk = result['modules']['sec']['details']['risk_factor_changes'][0]
    assert risk['title'] == '风险因素 1'
    assert risk['change_type'] == '披露内容增加'
    assert risk['text_original'] == 'Customers may postpone purchases.'
    assert risk['text'] == '客户可能推迟采购，影响营收确认时间。'
    assert len(calls) == 1
    assert loc.localize_result(source) == result
    assert len(calls) == 1


def test_failed_translation_is_not_cached_or_invented(tmp_path, monkeypatch):
    monkeypatch.setattr(loc, 'DB_PATH', tmp_path / 'translations.db')
    monkeypatch.setattr(loc, '_translate_batch', lambda _: {})
    source = {'modules': {'fundamental': {'details': {'company': {'description': 'Original evidence.'}}}}}
    company = loc.localize_result(source)['modules']['fundamental']['details']['company']
    assert company['description'] == '中文翻译暂不可用，请稍后重试。'
    assert company['description_original'] == 'Original evidence.'
    monkeypatch.setattr(loc, '_translate_batch', lambda texts: {t: '原始证据。' for t in texts})
    assert loc.localize_result(source)['modules']['fundamental']['details']['company']['description'] == '原始证据。'
