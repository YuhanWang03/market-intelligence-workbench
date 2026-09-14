"""Read-only account balance, deliberately separate from project spend."""
import os
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation

import requests

_lock = threading.Lock()
_cache = None


def deepseek_balance():
    global _cache
    with _lock:
        if _cache and _cache[0] > time.monotonic():
            return _cache[1]
        key = os.environ.get('DEEPSEEK_API_KEY', '').strip()
        if not key:
            return {'status': 'not_configured', 'message': '服务器未配置 DeepSeek API Key'}
        try:
            response = requests.get('https://api.deepseek.com/user/balance',
                                    headers={'Authorization': 'Bearer ' + key}, timeout=10)
            response.raise_for_status()
            data = response.json()
            balances = []
            for item in data.get('balance_infos', []):
                if item.get('currency') not in ('USD', 'CNY'):
                    continue
                values = {k: Decimal(str(item[k])) for k in ('total_balance', 'granted_balance', 'topped_up_balance')}
                if not all(v.is_finite() for v in values.values()):
                    raise ValueError('invalid balance')
                balances.append({'currency': item['currency'], **{k: str(v) for k, v in values.items()}})
            result = {'status': 'ok', 'balances': balances, 'fetched_at': datetime.now(timezone.utc).isoformat(),
                      'scope': 'account', 'message': '账户级官方余额，包含其他程序用量影响，不属于本项目费用统计'}
        except (requests.RequestException, ValueError, KeyError, TypeError, InvalidOperation):
            result = {'status': 'unavailable', 'message': '官方余额暂时无法获取，请检查服务器凭证或稍后重试'}
        _cache = (time.monotonic() + 60, result)
        return result
