"""Conservative, auditable relation evidence; never certify LLM descriptions."""
import re
from copy import deepcopy
from urllib.parse import urlparse


def safe_url(value):
    value = str(value or '')
    parsed = urlparse(value)
    return value if parsed.scheme in ('https', 'http') and parsed.hostname and not parsed.username else None


def assess(seed, target, category, result):
    url = safe_url(result.get('url'))
    text = str(result.get('content') or '')
    combined = str(result.get('title') or '') + ' ' + text
    a, b = (r'(?<![\w])' + re.escape(ticker) + r'(?![\w])' for ticker in (seed, target))
    if not url or not all(re.search(p, combined, re.I) for p in (a, b)):
        return None
    # Only explicit directional phrases count as supporting evidence. A search
    # hit remains a candidate: no implied exclusivity, contract or certification.
    patterns = {
        'supplier': [b + r'\s+(?:is\s+(?:a|the)\s+supplier\s+to|supplies)\s+' + a],
        'customer': [b + r'\s+is\s+(?:a|the)\s+customer\s+of\s+' + a,
                     a + r'\s+supplies\s+' + b],
        'smaller_peer': [b + r'\s+competes\s+with\s+' + a, a + r'\s+competes\s+with\s+' + b],
    }
    for sentence in re.split(r'(?<=[.!?])\s+|\n+', text):
        if re.search(r'\b(?:not|no|never|may|might|could|if|rumou?r|formerly|previously)\b', sentence, re.I):
            continue
        if any(re.search(p, sentence, re.I) for p in patterns.get(category, [])):
            return {'status': 'EVIDENCE_FOUND', 'url': url, 'text': sentence, 'title': result.get('title', '')}
    return {'status': 'CO_MENTION', 'url': url, 'text': combined[:1600], 'title': result.get('title', '')}


def label_sources(label):
    return [{'provider': 'Tavily-v2:' + label.evidence_status, 'url': label.evidence_url,
             'title': label.evidence_title, 'evidence_text': label.evidence_text,
             'evidence_summary': '搜索证据候选；未独立核验整条关系描述'}] if safe_url(label.evidence_url) else []


def present_relationships(source, store):
    result = deepcopy(source)
    module = result.get('modules', {}).get('supply_chain')
    if not module:
        return result
    mapping = {'supplier': 'SUPPLIER', 'customer': 'CUSTOMER', 'smaller_peer': 'COMPETITOR', 'beneficiary': 'BENEFICIARY', 'partner': 'PARTNER', 'platform': 'PLATFORM', 'substitute': 'SUBSTITUTE'}
    saved = {(r['target_ticker'], r['relationship_type']): r for r in store.relationships(result.get('ticker', ''))}
    rows = module.get('details', {}).get('relationships', [])
    for row in rows:
        record = saved.get((row.get('target_company'), mapping.get(row.get('relationship_type'), row.get('relationship_type'))))
        evidence = []
        if record:
            row['id'] = record['id']
            evidence = (store.relationship(record['id']) or {}).get('sources', [])
            row['status'] = record['status']
        evidence = [e for e in evidence if safe_url(e.get('url'))]
        evidence.sort(key=lambda e: (e.get('fetched_at') or '', e.get('id', 0)), reverse=True)
        row['source'] = evidence[0]['url'] if evidence else safe_url(row.get('source'))
        row['evidence_text'] = evidence[0].get('evidence_text') or '' if evidence else ''
        status = row.get('status', '')
        provider = evidence[0].get('provider', '') if evidence else ''
        row['evidence_status'] = status if status in ('CO_MENTION', 'EVIDENCE_FOUND') and provider == 'Tavily-v2:' + status else 'LEGACY' if status == 'VERIFIED' or row.get('verified') else status or 'UNCHECKED'
        row['verified'] = False
        row['confidence'] = None
    module.setdefault('metrics', {})['verified_relationships'] = 0
    module['confidence'] = None
    module['summary'] = '公司身份与产业关系分开核查；搜索证据不等于整条关系描述已被证实。' if rows else '本次未获得可展示关系；请展开数据质量查看候选生成、身份核查或筛选错误，不代表公司没有产业关系。'
    module['verified_source_count'] = 0
    module['completeness'] = 0
    if rows and module.get('status') not in ('PARTIAL_ERROR', 'FAILED'):
        module['status'] = 'PARTIAL'
    return result
