"""Evidence-based presentation of expectations and management outlook."""
from copy import deepcopy
import math
import re

METRICS = {
    'revenue': '营收', 'sales': '销售额', 'gross margin': '毛利率',
    'operating margin': '营业利润率', 'earnings': '盈利', 'eps': '每股收益',
    'capital expenditure': '资本支出', 'capex': '资本支出', 'free cash flow': '自由现金流',
    'demand': '需求', 'deliveries': '交付', 'capacity': '产能', 'supply': '供应',
    'financing': '融资', 'funding': '资金', 'infrastructure': '基础设施',
}
BOILERPLATE = re.compile(
    r'forward[- ]looking statements?|safe harbor|no obligation to update|'
    r'actual (?:future )?(?:results|outcomes).{0,100}(?:differ|different)|'
    r'read.{0,50}(?:report|10-q|10-k).{0,30}(?:completely|entirety)|'
    r'beliefs and opinions|information available to us as of|'
    r'identify.{0,30}forward[- ]looking|securities (?:act|exchange act)|'
    r'facilities.{0,80}good condition|statements.{0,100}(?:limited or incomplete|exhaustive)', re.I)
OUTLOOK = re.compile(r'\b(?:expect\w*|anticipat\w*|forecast\w*|outlook|guidance|plan\w*|intend\w*|project\w*|will)\b', re.I)
RISK = re.compile(r'\b(?:may|might|could)\b.{0,180}\b(?:delay\w*|postpone\w*|constrain\w*|limit\w*|affect\w*|reduc\w*|unable|not|risk\w*|shortage\w*)\b|\b(?:shortages?|constraints?|delays?|lack of)\b', re.I)


def classify_statement(text, filing_date='', source_url=None):
    text = re.sub(r'\s+', ' ', str(text or '')).strip()
    if len(text) < 25 or BOILERPLATE.search(text):
        return None
    metric = next((key for key in sorted(METRICS, key=len, reverse=True)
                   if re.search(r'\b' + re.escape(key) + r'\b', text, re.I)), None)
    if not metric:
        return None
    risk = bool(RISK.search(text))
    if not risk and not OUTLOOK.search(text):
        return None
    numeric = bool(re.search(r'\$\s*\d|\d[\d,.]*\s*(?:%|billion|million|basis points)', text, re.I))
    explicit = bool(re.search(r'\b(?:guidance|forecast|outlook)\b', text, re.I))
    group = 'risk' if risk else 'guidance' if numeric or explicit else 'outlook'
    status, status_label = 'NOT_COMPARABLE', '未确认指引变动'
    # A revenue increase is not a guidance upgrade. Require an explicit action
    # on guidance itself, with a company subject and no negation/conditional.
    action = re.search(r'\b(?:we|the company|management)\s+(?:has\s+|have\s+|is\s+|are\s+)?'
                       r'(rais\w*|increas\w*|lower\w*|reduc\w*|withdraw\w*|suspend\w*|reaffirm\w*|reiterat\w*|maintain\w*)'
                       r'\s+(?:(?:our|its|the|previous|revenue|earnings|full.year|quarterly|eps)\s+){0,5}'
                       r'(?:guidance|forecast|outlook)\b', text, re.I)
    if action and group != 'risk' and not re.search(r'\b(?:if|may|might|could|not)\b', text, re.I):
        verb = action.group(1).lower()
        status, status_label = (('RAISED', '公司明确上调') if verb.startswith(('rais', 'increas')) else
                                ('LOWERED', '公司明确下调') if verb.startswith(('lower', 'reduc')) else
                                ('WITHDRAWN', '公司明确撤回') if verb.startswith(('withdraw', 'suspend')) else
                                ('REITERATED', '公司明确维持'))
    if group == 'risk': status, status_label = 'RISK_CONTEXT', '风险情景'
    period = re.search(r'\b(?:next\s+(?:quarter|year)|(?:fiscal\s+)?Q[1-4](?:\s+(?:20\d{2}|FY\s*\d{2,4}))?|fiscal\s+(?:year\s+)?20\d{2}|FY\s*\d{2,4})\b', text, re.I)
    period_text = period.group(0) if period else ''
    period_label = re.sub(r'(?i)next quarter', '下一季度', period_text)
    period_label = re.sub(r'(?i)next year', '下一年度', period_label)
    period_label = re.sub(r'(?i)fiscal\s*(?:year)?', '财年 ', period_label)
    return {'metric': metric.replace(' ', '_'), 'metric_label': METRICS[metric],
            'group': group, 'guidance_type': 'FORMAL_GUIDANCE' if explicit and numeric else 'QUANTITATIVE_OUTLOOK' if numeric else 'QUALITATIVE_OUTLOOK',
            'status': status, 'status_label': status_label, 'direction': status,
            'period': period_text or None, 'period_label': period_label or '原文未明确期间',
            'value': None, 'range': None, 'unit': None,
            'evidence_text': text, 'filing_date': filing_date, 'source_url': source_url,
            'source': 'SEC filing', 'confidence': .7 if group == 'guidance' else .55}


def extract_statements(text, filing_date, source_url):
    # Scan beyond section introductions while keeping sentence boundaries.
    sentences = re.split(r'(?<=[.!?])\s+', re.sub(r'\s+', ' ', str(text or '')[:150000]))
    rows, seen = [], set()
    for sentence in sentences:
        if len(sentence) > 2400: continue
        row = classify_statement(sentence, filing_date, source_url)
        if row and sentence.lower() not in seen:
            seen.add(sentence.lower())
            rows.append(row)
    return sorted(rows, key=lambda row: {'guidance': 0, 'outlook': 1, 'risk': 2}[row['group']])[:24]


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def prepare_expectations(source):
    """Reclassify old evidence on read; never rewrite stored research snapshots."""
    result = deepcopy(source)
    modules = result.get('modules', {})
    module = modules.get('expectations')
    if not module: return result
    details = module.setdefault('details', {})
    metrics = module.get('metrics', {})
    upcoming = details.get('upcoming_earnings') or {}
    eps = metrics.get('forward_eps') if metrics.get('forward_eps') is not None else upcoming.get('eps_estimate')
    revenue = metrics.get('revenue_consensus') if metrics.get('revenue_consensus') is not None else upcoming.get('revenue_estimate')
    available = [label for label, value in [('每股收益', eps), ('营收', revenue)] if _number(value)]
    attempts = upcoming.get('attempts', [])
    missing_status = 'FETCH_FAILED' if any(a.get('status') == 'FETCH_FAILED' for a in attempts) else 'NO_DATA' if attempts else 'UNKNOWN'
    missing_reason = {'FETCH_FAILED': '数据源请求或解析失败，请重新研究；不是确认没有预期数据',
                      'NO_DATA': '数据源未返回有效预期值；也可能是上游静默失败，可稍后重试',
                      'UNKNOWN': '旧研究未记录获取详情，请重新研究以核查原因'}[missing_status]
    history = modules.get('earnings', {}).get('details', {}).get('history', [])
    comparable = [row for row in history if any(_number(row.get(key)) for key in ('eps_surprise', 'revenue_surprise'))]
    matrix = {
        'current_consensus': {'status': 'AVAILABLE' if len(available) == 2 else 'PARTIAL_DATA' if available else missing_status,
            'reason': ('本次已取得：' + '、'.join(available) + '；' + upcoming.get('period_label', '期间以数据源为准')) if available else missing_reason},
        'earnings_surprise_history': {'status': 'AVAILABLE' if comparable else 'NO_DATA',
            'reason': f'本次有 {len(comparable)} 个报告期可比较实际值与预期值' if comparable else '本次缺少实际值与预期值的有效对照'},
    }
    for days in (30, 60, 90):
        matrix[f'revision_{days}d'] = {'status': 'NOT_CONNECTED', 'reason': '尚未接入同一目标报告期的历史预期快照；刷新无法补齐，需历史数据源或持续积累快照'}
    details['capability_matrix'] = matrix
    raw = details.get('guidance', [])
    rows, seen, filtered, duplicates = [], set(), 0, 0
    for old in sorted(raw, key=lambda r: str(r.get('filing_date') or ''), reverse=True):
        text = old.get('evidence_text_original') or old.get('evidence_text')
        row = classify_statement(text, old.get('filing_date', ''), old.get('source_url'))
        if not row:
            filtered += 1
            continue
        key = re.sub(r'\s+', ' ', str(text)).strip().lower()
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        row['filing_type'] = old.get('filing_type', '')
        row['source_section'] = old.get('source_section', 'MD&A')
        rows.append(row)
    details['guidance'] = rows[:30]
    details['guidance_quality'] = {'filtered': filtered, 'duplicates': duplicates,
        'displayed': len(details['guidance']), 'version': 2}
    return result
