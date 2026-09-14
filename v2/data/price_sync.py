"""Conservative official-price synchronizer; never invent historical rates."""
import hashlib
import json
import re
import threading
from datetime import datetime, timedelta, timezone

import requests
from bs4 import BeautifulSoup
from v2.data.usage_ledger import _conn, init

URL = 'https://api-docs.deepseek.com/zh-cn/quick_start/pricing/'
REVIEW_TTL = timedelta(days=7)
REQUEST_HEADERS = {
    'Accept': 'text/html,application/xhtml+xml',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'User-Agent': 'AI-Hedge-Fund-Billing/1.0 (+official-price-sync)',
}
_lock = threading.Lock()
SCHEMA = '''CREATE TABLE IF NOT EXISTS price_sync_state (
 id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL
);'''


def parse_prices(html):
    soup = BeautifulSoup(html, 'html.parser')
    # Expand rowspan/colspan before associating a rate with a model column.
    tables = []
    for table in soup.find_all('table'):
        occupied = {}
        rows = []
        for y, tr in enumerate(table.find_all('tr')):
            x = 0
            for cell in tr.find_all(['td', 'th'], recursive=False):
                while (y, x) in occupied:
                    x += 1
                value = re.sub(r'\s+', '', cell.get_text())
                rs, cs = int(cell.get('rowspan', 1)), int(cell.get('colspan', 1))
                if not 1 <= rs <= 50 or not 1 <= cs <= 30:
                    raise ValueError('unrecognized table spans')
                for yy in range(y, y+rs):
                    for xx in range(x, x+cs):
                        occupied[yy, xx] = value
                x += cs
            rows.append([occupied.get((y, xx), '') for xx in range(max((xx for yy, xx in occupied if yy == y), default=-1)+1)])
        if rows and any(v.startswith('deepseek-') for v in rows[0]):
            tables.append(rows)
    if len(tables) != 1:
        raise ValueError('missing or ambiguous model table')
    rows = tables[0]
    # Docusaurus currently renders footnote markers into header text, for
    # example ``deepseek-flash (1)``.  They are annotations, not model names.
    models = []
    for index, raw_name in enumerate(rows[0]):
        name = re.sub(r'(?:\(\d+\)|[¹²³⁴⁵⁶⁷⁸⁹⁰]+)$', '', raw_name)
        if re.fullmatch(r'deepseek-[a-z0-9-]+', name):
            models.append((index, name))
    if not models or len(set(n for _, n in models)) != len(models):
        raise ValueError('invalid model columns')
    text = re.sub(r'\s+', '', soup.get_text())
    rule = re.search(r'高峰时段为北京时间周一至周五(\d{1,2}):(\d{2})[-–](\d{1,2}):(\d{2})、(\d{1,2}):(\d{2})[-–](\d{1,2}):(\d{2})（其余为空闲时段）', text)
    if not rule:
        raise ValueError('unrecognized peak schedule')
    values = list(map(int, rule.groups()))
    if any(values[i] > 23 or values[i+1] > 59 for i in (0, 2, 4, 6)):
        raise ValueError('invalid schedule time')
    minutes = [values[i]*60+values[i+1] for i in (0, 2, 4, 6)]
    if not 0 <= minutes[0] < minutes[1] <= minutes[2] < minutes[3] <= 1440:
        raise ValueError('overlapping schedule')
    result = []
    keys = {'百万tokens输入（缓存命中）': 'cached_input', '百万tokens输入（缓存未命中）': 'input', '百万tokens输出': 'output'}
    for index, model in models:
        periods = {'空闲时段': {}, '高峰时段': {}}
        for row in rows:
            if len(row) <= index or len(row) < 3 or row[1] not in keys or row[2] not in periods:
                continue
            key = keys[row[1]]
            if key in periods[row[2]] or not re.fullmatch(r'\d+(?:\.\d+)?元', row[index]):
                raise ValueError('ambiguous price or currency')
            amount = float(row[index][:-1])
            if not 0 < amount < 100000:
                raise ValueError('invalid rate')
            periods[row[2]][key] = amount
        if any(set(rates) != set(keys.values()) for rates in periods.values()):
            raise ValueError('incomplete rates')
        result.append({'provider': 'DeepSeek', 'model': model, 'currency': 'CNY', 'rates': periods['空闲时段'],
                       'schedule': {'timezone': 'Asia/Shanghai', 'weekdays': [0,1,2,3,4],
                                    'windows': [minutes[:2], minutes[2:]], 'peak_rates': periods['高峰时段']}})
    return result


def sync_status():
    with _conn() as conn:
        init(conn)
        conn.executescript(SCHEMA)
        row = conn.execute('SELECT payload FROM price_sync_state WHERE id=1').fetchone()
    return json.loads(row[0]) if row else {'status': 'not_synced', 'message': '尚未同步官方价格'}


def sync_prices(force=False):
    with _lock:
        state = sync_status()
        now = datetime.now(timezone.utc)
        if state.get('attempted_at'):
            elapsed = (now-datetime.fromisoformat(state['attempted_at'])).total_seconds()
            if elapsed < (60 if force else 6*3600):
                return state
        at = now.isoformat()
        try:
            # No credentials; fixed official host, no redirects to other origins.
            response = requests.get(URL, headers=REQUEST_HEADERS, timeout=15, allow_redirects=False)
            if response.status_code != 200 or len(response.content) > 2_000_000:
                raise ValueError('official page unavailable')
            html = response.content.decode('utf-8')
            parsed = parse_prices(html)
            digest = hashlib.sha256(json.dumps(parsed, sort_keys=True).encode()).hexdigest()
            page_hash = hashlib.sha256(response.content).hexdigest()
            with _conn() as conn:
                init(conn)
                conn.executescript(SCHEMA)
                conn.execute('BEGIN IMMEDIATE')
                from v2.data.billing_rules import deepseek_model_family

                # Bridge a failed-sync gap with the last verified rate until
                # this new official observation.  This is deliberately an
                # estimate and is recorded in the price payload.  The new rate
                # starts at the new observation, so changed prices are never
                # retroactively applied to the gap.
                current_families = []
                for item in parsed:
                    family = set(deepseek_model_family(item['model']))
                    if family not in current_families:
                        current_families.append(family)
                extended = 0
                prior_rows = [
                    (row, json.loads(row['payload']))
                    for row in conn.execute("SELECT id, payload FROM usage_prices WHERE provider='DeepSeek'").fetchall()
                ]
                for family in current_families:
                    candidates = [(row, price) for row, price in prior_rows if price.get('model') in family and price.get('effective_at', at) < at]
                    if not candidates:
                        continue
                    latest_effective = max(price['effective_at'] for _, price in candidates)
                    latest = [(row, price) for row, price in candidates if price['effective_at'] == latest_effective]
                    if any(price.get('review_after', at) >= at for _, price in latest):
                        continue
                    for row, previous in latest:
                        previous['review_after'] = at
                        previous['continuity_extended_at'] = at
                        previous['continuity_basis'] = '上一已核验价格延续至下一次官方价格观测；金额为估算'
                        conn.execute(
                            'UPDATE usage_prices SET review_after=?, payload=? WHERE id=?',
                            (at, json.dumps(previous), row['id']),
                        )
                        extended += 1
                for item in parsed:
                    # One observed snapshot per successful check. Its start time
                    # is observation time, NOT an invented official effective date.
                    ident = hashlib.sha256((at+item['model']).encode()).hexdigest()[:32]
                    price = {**item, 'id': ident, 'effective_at': at, 'created_at': at,
                             'review_after': (now+REVIEW_TTL).isoformat(),
                             'source': URL, 'automatic': True, 'observed_at': at,
                             'page_sha256': page_hash, 'page_snapshot': html,
                             'effective_basis': 'first_observed'}
                    conn.execute('INSERT INTO usage_prices VALUES (?,?,?,?,?,?,?)',
                                 (ident, 'DeepSeek', item['model'], at, price['review_after'], at, json.dumps(price)))
                state = {'status': 'ok', 'attempted_at': at, 'synced_at': at, 'models': [p['model'] for p in parsed],
                         'changed': state.get('digest') != digest, 'digest': digest,
                         'extended_versions': extended,
                         'message': f'已同步官方人民币价格及北京时间高峰／空闲时段；新价格从本次采集时间起适用，{extended} 个旧价格版本已连续延伸至本次观测。'}
                conn.execute('INSERT OR REPLACE INTO price_sync_state VALUES (1,?)', (json.dumps(state),))
        except Exception:
            state = {**state, 'status': 'error', 'attempted_at': at,
                     'message': '官方价格同步失败或页面规则无法识别；未覆盖旧价格。旧价格到期后金额待核算。'}
            with _conn() as conn:
                conn.executescript(SCHEMA)
                conn.execute('INSERT OR REPLACE INTO price_sync_state VALUES (1,?)', (json.dumps(state),))
        return state
