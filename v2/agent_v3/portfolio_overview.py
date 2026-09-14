"""Full-snapshot portfolio arithmetic and a readable, model-independent answer."""
from decimal import Decimal, InvalidOperation
import hashlib
import json

from v2.agent_v2.catalog import CapabilityCatalog, CapabilitySpec
from v2.agent_v2.models import EvidenceItem, ResultStatus, ToolEnvelope


def number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        result = Decimal(str(value))
        return result if result.is_finite() else None
    except (InvalidOperation, ValueError):
        return None


def overview(base):
    positions = base.metadata.get("positions", [])
    account = base.metadata.get("account", {})
    rows, gaps = [], []
    for position in positions:
        ticker = position.get("ticker") or position.get("symbol") or "未知标的"
        qty, price = number(position.get("qty")), number(position.get("current_price"))
        short = bool(qty is not None and qty < 0) or "short" in str(position.get("side", "")).lower()
        value = number(position.get("market_value"))
        if value is None and qty is not None and price is not None:
            value = (-abs(qty) if short else qty) * price
        pnl = number(position.get("unrealized_pl"))
        cost = number(position.get("avg_entry_price"))
        # Keep the broker's reported P/L; missing values are not zero.
        pct = number(position.get("unrealized_pl_pct"))
        if value is None:
            gaps.append(f"{ticker} 缺少市值及可计算市值的价格/数量。")
        if pnl is None:
            gaps.append(f"{ticker} 缺少未实现盈亏。")
        rows.append(dict(ticker=ticker, value=value, pnl=pnl, pct=pct,
                         cost=abs(qty * cost) if qty is not None and cost is not None else None,
                         short=short))
    complete_value = all(row["value"] is not None for row in rows)
    complete_pnl = all(row["pnl"] is not None for row in rows)
    gross = sum((abs(row["value"]) for row in rows), Decimal(0)) if complete_value else None
    net = sum((row["value"] for row in rows), Decimal(0)) if complete_value else None
    pnl = sum((row["pnl"] for row in rows), Decimal(0)) if complete_pnl else None
    basis = sum((row["cost"] for row in rows), Decimal(0)) if all(row['cost'] is not None for row in rows) else None
    pnl_pct = pnl / basis * 100 if pnl is not None and basis else None
    equity = number(account.get("equity", account.get("portfolio_value")))
    cash = number(account.get("cash"))
    buying_power = number(account.get("buying_power"))
    by_size = sorted((r for r in rows if r["value"] is not None), key=lambda r: (-abs(r["value"]), r["ticker"]))
    losses = sorted((r for r in rows if r["pnl"] is not None and r["pnl"] < 0), key=lambda r: (r["pnl"], r["ticker"]))
    gains = sorted((r for r in rows if r["pnl"] is not None and r["pnl"] > 0), key=lambda r: (-r["pnl"], r["ticker"]))
    percent_losses = sorted((r for r in rows if r['pct'] is not None and r['pct'] < 0), key=lambda r: (r['pct'], r['ticker']))
    priorities = list(dict.fromkeys([r['ticker'] for r in by_size[:1] + losses[:2] + by_size]))[:3]
    key = 'portfolio-overview-' + hashlib.sha256(json.dumps(base.metadata, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]
    evidence = []
    def fact(suffix, text, metric=None, value=None, formula=None):
        item_id = key + '-' + suffix
        evidence.append(EvidenceItem(id=item_id, entity='account', claim=text, metric=metric,
            value=float(value) if value is not None else None, as_of=base.as_of, source_id='account.portfolio',
            metadata={'formula': formula, 'source_evidence_ids': [e.id for e in base.evidence]}))
        return f'{text} [{item_id}]'
    def money(value):
        return '数据缺失' if value is None else f'{value:,.2f} 美元'
    lines = ['**持仓组合总览**', fact('count', f'当前共有 {len(rows)} 个持仓。', 'position_count', len(rows))]
    if not rows and base.metadata.get('positions_available', True):
        lines.append('当前没有持仓，无需进行逐股研究。')
    if not base.metadata.get('positions_available', True):
        gaps.append('账户接口未提供持仓列表，不能认定为空仓。')
        net = gross = pnl = pnl_pct = None
        lines = ['**持仓组合总览**', '账户接口未提供持仓列表，暂时无法完成组合汇总。']
    for suffix, label, value, formula in (
        ('net', '持仓净市值', net, 'sum(position.market_value); missing values derived from signed qty * current_price'),
        ('gross', '持仓总敞口（市值绝对值之和）', gross, 'sum(abs(position.market_value))'),
        ('pnl', '未实现盈亏合计', pnl, 'sum(position.unrealized_pl)'),
        ('equity', '账户净值', equity, 'account.equity or account.portfolio_value'),
        ('cash', '现金', cash, 'account.cash'),
        ('power', '购买力（不可加作账户资产）', buying_power, 'account.buying_power'),
    ):
        if value is not None:
            lines.append(fact(suffix, f'{label}：{money(value)}。', suffix, value, formula))
    if pnl_pct is not None:
        lines.append(fact('pnl-percent', f'相对剩余持仓成本绝对值的浮盈亏比例：{pnl_pct:+.2f}%。', 'pnl_percent', pnl_pct, 'sum(unrealized_pl) / sum(abs(qty * avg_entry_price)) * 100'))
    if net is not None and equity is not None and cash is not None and abs(equity-net-cash)>Decimal('0.01'):
        difference=equity-net-cash
        lines.append(fact('reconciliation', f'账户净值与持仓净市值加现金相差 {money(difference)}，需要核对快照时点及其他资产负债；不将差额当作其他持仓市值。', 'reconciliation_difference', difference, 'account_equity - net_market_value - cash'))
    if gross:
        top = by_size[0]
        top3 = sum((abs(r['value']) for r in by_size[:3]), Decimal(0)) / gross * 100
        lines.extend(['\n**仓位集中度**', fact('concentration', f'最大仓位 {top["ticker"]} 占持仓总敞口 {abs(top["value"])/gross*100:.2f}%；前三大仓位合计占 {top3:.2f}%。分母为全部持仓总敞口，不含现金。', formula='abs(market_value) / gross_exposure * 100')])
    lines.append('\n**全部持仓（浮盈亏口径，不是当日涨跌）**')
    for i, row in enumerate(rows):
        weight = f'{abs(row["value"])/gross*100:.2f}%' if gross and row['value'] is not None else '未知'
        pct = f'{row["pct"]*100:+.2f}%' if row['pct'] is not None else '未知'
        lines.append('- ' + fact(f'position-{i}', f'{row["ticker"]}：市值 {money(row["value"])}，仓位占比 {weight}，浮盈亏 {money(row["pnl"])}（{pct}）。', formula='weight = abs(market_value) / gross_exposure; percentage = broker fraction * 100'))
    if losses:
        lines.extend(['\n**主要亏损贡献**', fact('losses', '按浮亏金额排序：' + '；'.join(f'{r["ticker"]} {money(r["pnl"])}' for r in losses[:3]) + '。亏损比例最大不等于对组合亏损贡献最大。')])
    if percent_losses:
        lines.append(fact('percent-losses','按浮亏比例排序：'+'；'.join(f'{r["ticker"]} {r["pct"]*100:+.2f}%' for r in percent_losses[:3])+'。'))
    if gains:
        lines.append(fact('gains', '主要浮盈贡献：' + '；'.join(f'{r["ticker"]} {money(r["pnl"])}' for r in gains[:3]) + '。'))
    if any(r['short'] for r in rows):
        gaps.append('包含空头；净市值与总敞口不同，仓位占比按市值绝对值计算。')
    gaps.extend(['仅依据当前账户快照；浮盈亏不能证明下跌原因，也不是已实现或当日收益。', '未穿透 ETF 成分股，未确认行业敞口或个股基本面；不能把全部标的统一当作普通股票分析。'])
    lines.append('\n**范围与限制**\n' + '\n'.join('- '+x for x in gaps))
    answer = '\n'.join(lines)
    return ToolEnvelope('account.overview', ResultStatus.PARTIAL_DATA if not complete_value or not complete_pnl or not base.metadata.get('positions_available',True) else ResultStatus.COMPLETED,
        subject='account', as_of=base.as_of, evidence=evidence, limitations=gaps,
        metrics={'position_count':len(rows),'net_market_value':float(net) if net is not None else None,'gross_exposure':float(gross) if gross is not None else None,'unrealized_pnl':float(pnl) if pnl is not None else None},
        metadata={'portfolio_overview_answer':answer,'priority_tickers':priorities,'positions_complete':complete_value and complete_pnl,'positions':positions})


def register_overview(registry):
    registry.catalog = CapabilityCatalog([*registry.catalog.specs(), CapabilitySpec('account.overview', 'account', 'Calculate all holdings, exposure concentration and unrealized P/L contributions from one account snapshot.', {'type':'object','properties':{},'additionalProperties':False})])
    def execute(args, context):
        return overview(registry.handlers['account.portfolio']({}, context))
    registry.register('account.overview', execute)
