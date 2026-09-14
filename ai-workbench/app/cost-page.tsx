'use client';
import { useState, type FormEvent } from 'react';
import { apiJson } from './lib/api';
import './cost-page.css';
import { BillingPanel, UsageBreakdown } from './billing-panel';

type Price = { automatic?: boolean; applied_period?: string; schedule?: { peak_rates: Record<string, number> }; currency: 'CNY' | 'USD'; id: string; provider: string; model: string; effective_at: string; review_after: string; source: string; rates: Record<string, number> };
type SyncStatus = { status: string; message: string; synced_at?: string; models?: string[] };
type Event = { requested_model?: string; pricing_model?: string; pricing_basis?: string; usage_note?: string; breakdown?: Record<string, { tokens: number; rate: number; amount: number }> | null; quota?: { free_credits: number; paid_credits: number; gross_amount: number }; quota_note?: string; amount: number | null; currency: 'CNY' | 'USD' | null; channel?: string; id: string; occurred_at: string; category: string; provider: string; model: string; ticker?: string; endpoint: string; source: string; state: string; status: string; reason: string; cost_usd: number | null; usage_basis: string; usage: { input_tokens?: number; output_tokens?: number; cached_tokens?: number; units?: number; search_depth?: string }; price: Price | null };
export type CostReport = { currencies: { currency: 'CNY' | 'USD'; today_amount: number; month_amount: number; total_amount: number }[]; timezone: string; today_cost_usd: number; month_cost_usd: number; total_cost_usd: number; total_requests: number; pending_requests: number; by_provider: { amounts: Record<string, number>; category: string; provider: string; cost_usd: number; requests: number; pending: number; input_tokens: number; output_tokens: number; credits: number }[]; prices: Price[]; price_sync?: SyncStatus; recent: Event[]; recent_filter: string; recent_total: number; recent_offset: number; recent_limit: number; recent_has_more: boolean; recent_detail_hours: number | null; recent_cutoff: string | null };
const money = (n: number | null | undefined, currency?: string | null) => n == null || !currency ? '待核算' : `${currency === 'CNY' ? '¥' : '$'}${n.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 6 })} ${currency}`;
const time = (s: string) => new Date(s).toLocaleString('zh-CN', { timeZone: 'America/New_York', hour12: false }) + ' ET';
const category: Record<string, string> = { data: '数据查询', llm: 'LLM', search: '搜索' };
const channels: Record<string, string> = { web: '网页', telegram: 'Telegram', background: '后台任务', unknown: '来源未知' };
const endpoints: Record<string, string> = { financial_metrics: '财务指标', company_facts: '公司事实', news: '新闻', earnings: '财报', insider_trades: '内部人交易', prices: '股价', filings: 'SEC 文件', line_items: '财务报表', search: '网页搜索' };

export function CostPage({ report, loading, error, refresh }: { report: CostReport | null; loading: boolean; error: string; refresh: (filter?: string, offset?: number, syncPrices?: boolean) => Promise<void> }) {
  const [filter, setFilter] = useState('all');
  const [offset, setOffset] = useState(0);
  const [provider, setProvider] = useState('DeepSeek');
  const [notice, setNotice] = useState('');
  const [saving, setSaving] = useState(false);
  const [syncBusy, setSyncBusy] = useState(false);
  const sync = report?.price_sync;
  async function syncPrices() {
    setSyncBusy(true);
    try { await refresh(filter, offset, true) }
    finally { setSyncBusy(false) }
  }
  const llm = provider === 'DeepSeek' || provider === 'Other LLM';
  async function savePrice(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); setSaving(true); setNotice('');
    const data = new FormData(event.currentTarget);
    try {
      const rates = Object.fromEntries((llm ? ['input', 'cached_input', 'output'] : ['unit']).map(k => [k, Number(data.get(k))]));
      await apiJson('/api/costs/prices', { method: 'POST', body: JSON.stringify({ provider, currency: data.get('currency'), model: data.get('model'), rates, source: data.get('source'), effective_at: new Date(String(data.get('effective'))).toISOString(), review_after: new Date(String(data.get('review'))).toISOString() }) });
      setNotice('价格版本已保存，新调用按生效版本计费。'); await refresh(filter, offset);
    } catch (e) { setNotice(e instanceof Error ? e.message : '保存失败') }
    finally { setSaving(false) }
  }
  const reportFilter = report?.recent_filter || 'all';
  const visibleEvents = reportFilter === filter ? report?.recent || [] : [];
  const total = reportFilter === filter ? report?.recent_total || 0 : 0;
  const pageStart = total ? offset + 1 : 0;
  const pageEnd = Math.min(offset + visibleEvents.length, total);
  const automaticPrices = report?.prices.filter((price, index, all) => price.automatic && all.findIndex(candidate => candidate.automatic && candidate.model === price.model) === index) || [];
  const loadPage = (nextOffset: number) => { const safeOffset = Math.max(0, nextOffset); setOffset(safeOffset); void refresh(filter, safeOffset) };
  const changeFilter = (nextFilter: string) => { setFilter(nextFilter); setOffset(0); void refresh(nextFilter, 0) };
  return <div className="cost-page unified-costs">
    <div className="page-heading"><div><span className="eyebrow">用量与费用</span><h1>花费</h1><p>统一查看网页、Telegram 与数据服务的调用费用</p></div><button className="run-button" disabled={loading || syncBusy} onClick={() => void syncPrices()}>{loading || syncBusy ? '同步刷新中…' : '刷新账本与收费标准'}</button></div>
    {error && <p role="alert" className="quality-warning">{error}</p>}
    <div className="cost-summary-grid">
      {([['今日估算', 'today_amount'], ['本月估算', 'month_amount'], ['累计估算', 'total_amount']] as const).map(([label, field]) => <section className="surface" key={label}><span className="metric-label">{label}</span>{report ? report.currencies.map(c => <div className="currency-metric" key={c.currency}><span>{c.currency}</span><strong>{money(c[field], c.currency).replace(` ${c.currency}`, '')}</strong></div>) : <strong>—</strong>}</section>)}
    </div>
    <p className="cost-basis">人民币与美元独立统计，不换算 · 累计费用永久保留 · 单次调用明细仅保留最近24小时 · 无法核算的调用会被清除</p>
    <div className="cost-ledger-grid">
      <div className="cost-ledger-left">
        <section className="surface engine-card"><div className="surface-header"><div><h2>按服务汇总</h2><span>累计估算及已记录用量</span></div></div><div className="cost-breakdown">{report?.by_provider.map(g => <div key={g.category + g.provider}><span>{category[g.category]} · {g.provider}<small>{g.category === 'llm' ? `输入 ${g.input_tokens.toLocaleString()} / 输出 ${g.output_tokens.toLocaleString()} Tokens` : g.category === 'search' ? `${g.credits.toLocaleString()} 已知 credits` : `${g.requests} 次请求`}</small></span><strong>{Object.entries(g.amounts).map(([code, amount]) => <span className="currency-line" key={code}>{money(amount, code)}</span>)}</strong><small>{g.requests.toLocaleString()} 条累计</small></div>)}{!report?.by_provider.length && <p className="insufficient-note">尚无记录</p>}</div></section>
        <section id="cost-recent" className="surface engine-card"><div className="surface-header"><div><h2>最近调用</h2><span>仅保留最近24小时 · 当前类型 {total} 条 · 点击展开明细</span></div><select aria-label="筛选费用类型" value={filter} onChange={e => changeFilter(e.target.value)}><option value="all">全部类型</option><option value="data">数据查询</option><option value="llm">LLM</option><option value="search">搜索</option></select></div>
          <div className="usage-columns" aria-hidden="true"><span>来源 / 时间</span><span>服务 / 模型</span><span>估算费用</span></div>
          <div className="usage-list">{visibleEvents.map(e => <details key={e.id}><summary><span><span className="cost-channel">{channels[e.channel || 'unknown'] || '来源未知'}</span>{time(e.occurred_at)}<small>{e.ticker || e.source || '—'}</small></span><span>{e.provider}<small>{category[e.category]} · {e.category === 'llm' ? (e.pricing_model || e.requested_model || e.model) : (endpoints[e.model] || e.model)}</small></span><strong>{money(e.amount, e.currency)}</strong></summary><div className="usage-detail"><UsageBreakdown event={e}/><p>{e.state === 'failed' ? '请求失败' : '请求已返回'} · {e.usage_basis === 'reported' ? '接口返回用量' : e.usage_basis === 'legacy' ? '旧账本记录' : '记录用量'}{e.reason && ` · ${e.reason}`}</p><p>{e.category === 'llm' ? `输入 ${e.usage.input_tokens ?? '未知'} · 缓存命中 ${e.usage.cached_tokens ?? '未知'} · 输出 ${e.usage.output_tokens ?? '未知'} Tokens` : `${e.category === 'search' ? 'Credits' : '请求数'}：${e.usage.units ?? '未知'}${e.usage.search_depth ? ` · 模式 ${e.usage.search_depth}` : ''}`}</p>{e.price ? <><p>价格版本：{e.price.id}</p><p>采用单价（{e.price.currency || 'USD'}）：{Object.entries(e.price.rates).map(([k,v]) => `${({ input: '输入/百万 Tokens', cached_input: '缓存输入/百万 Tokens', output: '输出/百万 Tokens', unit: e.category === 'search' ? '每 credit' : '每请求' } as Record<string,string>)[k] || k} ${v}`).join('；')}</p><p>来源：{e.price.source}</p></> : <p>使用固定端点单价。</p>}</div></details>)}{loading && reportFilter !== filter && <p className="insufficient-note">正在加载筛选结果…</p>}{!loading && reportFilter === filter && !visibleEvents.length && <p className="insufficient-note">最近24小时内没有符合条件的调用记录。</p>}</div>
          {total > 0 && <nav className="usage-pagination" aria-label="调用记录分页"><span>显示 {pageStart}–{pageEnd} / {total}</span><div><button disabled={loading || offset === 0} onClick={() => loadPage(offset - 100)}>上一页</button><button disabled={loading || !report?.recent_has_more} onClick={() => loadPage(offset + 100)}>下一页</button></div></nav>}
        </section>
      </div>
      <section className="surface engine-card service-fees"><div className="surface-header"><div><h2>服务收费详情</h2><span>刷新账本时同步核对当前收费标准</span></div><span className={`sync-pill ${sync?.status === 'ok' ? 'ok' : ''}`}>{sync?.status === 'ok' ? '已同步' : '待核对'}</span></div>
        <div className="fee-service-list">
          <article><div><strong>DeepSeek</strong><small>按调用发生时刻的北京时间区分高峰／空闲；模型名无法确认时按 deepseek-flash 估算</small></div>{automaticPrices.map(price => <div className="fee-rate" key={price.id}><b>{price.model}</b><span>空闲：输入 ¥{price.rates.input} · 缓存 ¥{price.rates.cached_input} · 输出 ¥{price.rates.output}</span>{price.schedule && <span>高峰：输入 ¥{price.schedule.peak_rates.input} · 缓存 ¥{price.schedule.peak_rates.cached_input} · 输出 ¥{price.schedule.peak_rates.output}</span>}<small>单位：每百万 Tokens</small></div>)}{!automaticPrices.length && <p>尚未同步到官方价格。</p>}</article>
          <article><div><strong>Tavily</strong><small>所有实际返回的 credits 统一计费，不抵扣免费额度</small></div><div className="fee-rate"><b>$0.008 / credit</b><span>调用费用 = 实际 credits × $0.008</span></div></article>
          <article><div><strong>Financial Datasets</strong><small>按成功的数据端点请求估算</small></div><div className="fee-rate"><b>默认 $0.02 / 请求</b><span>若服务器配置了端点专属价格，以该配置为准</span></div></article>
        </div>
        <BillingPanel report={report} refresh={() => refresh(filter, offset)}/>
        <div className="fee-sync-note"><strong>{sync?.status === 'ok' ? '收费标准最近同步成功' : '收费标准需要核对'}</strong><p>{sync?.message || '点击顶部按钮同步收费标准并刷新账本。'}</p>{sync?.synced_at && <small>{time(sync.synced_at)}</small>}</div>
      </section>
    </div>
    <details className="surface engine-card cost-settings"><summary className="surface-header"><div><h2>价格与计费设置</h2><span>官方价格同步 · 手动单价 · 历史版本</span></div><span className={`sync-pill ${sync?.status === 'ok' ? 'ok' : ''}`}>{sync?.status === 'ok' ? '最近同步成功' : sync?.status === 'error' ? '同步需关注' : '等待同步'} · 展开</span></summary>
      <div className="cost-note">
        <h3>DeepSeek 官方价格自动填写</h3>
        <p>服务启动时检查，此后每 6 小时同步一次。自动识别模型、人民币单价，并按每条调用发生时刻换算北京时间后判断高峰／空闲时段；连续 7 天未成功复核才暂停金额估算。</p>
        <button disabled={syncBusy} onClick={() => void syncPrices()}>{syncBusy ? '同步中…' : '立即同步官方价格'}</button>
        <p role="status">{sync?.message || '尚未取得同步状态'}</p>
        {sync?.synced_at && <small>最近成功：{time(sync.synced_at)} · 手动同步间隔至少 1 分钟</small>}
        <p>新价格从采集时刻起适用；同步中断期间使用上一已核验价格估算至下一次官方观测，并保留连续性标记。Financial Datasets 沿用固定端点配置。</p>
        {automaticPrices.map(p => <p key={p.id}><strong>{p.model}</strong><br/>每百万 Tokens（人民币）：未缓存输入 / 缓存输入 / 输出<br/>空闲：{p.rates.input} / {p.rates.cached_input} / {p.rates.output}{p.schedule && <>；高峰：{p.schedule.peak_rates.input} / {p.schedule.peak_rates.cached_input} / {p.schedule.peak_rates.output}</>}</p>)}
        <a href="https://api-docs.deepseek.com/zh-cn/quick_start/pricing/" target="_blank" rel="noreferrer">查看官方价格与时段规则 ↗</a>
      </div>
      <details className="price-editor"><summary>手动新增／更新价格版本（自动同步会采用后续官方版本）</summary><form onSubmit={event => void savePrice(event)}>
        <label>供应商<select value={provider} onChange={e => setProvider(e.target.value)}>{['DeepSeek', 'Financial Datasets', 'Other LLM'].map(p => <option key={p}>{p}</option>)}</select></label>
        <label>计价币种<select name="currency" required defaultValue="USD"><option value="USD">美元 USD</option><option value="CNY">人民币 CNY</option></select></label>
        <label>精确模型名／端点名<input key={provider} name="model" required maxLength={120} defaultValue={provider === 'Tavily' ? 'search' : ''} placeholder={llm ? '与调用记录返回的模型名一致' : '例如 financial_metrics'} /></label>
        {(llm ? [['input','未缓存输入 / 百万 Tokens'],['cached_input','缓存输入 / 百万 Tokens'],['output','输出 / 百万 Tokens']] : [['unit',provider === 'Tavily' ? '每 credit（套餐折算价）' : '每次请求']]).map(([k,label]) => <label key={k}>{label}<input name={k} type="number" min="0" max="100000" step="any" required /></label>)}
        <label>生效时间（本机时区）<input name="effective" type="datetime-local" required /></label><label>价格复核期限（本机时区）<input name="review" type="datetime-local" required /></label>
        <label className="price-source">价格来源／套餐说明<input name="source" required maxLength={500} placeholder="官方价格页面、核对日期及适用套餐" /></label><button disabled={saving} type="submit">{saving ? '保存中…' : '保存新版本'}</button>
      </form>{notice && <p role="status">{notice}</p>}</details>
      <div className="cost-price-list">{report?.prices.map(p => <div key={p.id}><strong>{p.provider} · {p.model} · {p.currency || 'USD'}</strong><span>{time(p.effective_at)} 生效 → {time(p.review_after)} 前复核</span><small>{p.source}</small></div>)}{!report?.prices.length && <p className="insufficient-note">尚未配置价格版本。Financial Datasets 沿用既有端点配置。</p>}</div>
      <div className="cost-note"><p>每笔调用冻结调用当时采用的单价；DeepSeek 模型名无法确认时按当时有效的 deepseek-flash 单价估算，只计算接口实际返回的 Token。刷新收费标准只影响新调用，不会改写已经核算的历史费用。</p><p>Tavily 只计算接口实际返回的 credits，不抵扣每月免费额度，统一按 $0.008/credit 估算。单次调用明细仅保留最近24小时，累计费用和累计用量仍永久保留。</p><p><a href="https://api-docs.deepseek.com/quick_start/pricing" target="_blank" rel="noreferrer">核对 DeepSeek 官方价格 ↗</a> · <a href="https://docs.tavily.com/documentation/api-credits" target="_blank" rel="noreferrer">核对 Tavily 价格 ↗</a></p></div>
    </details>
  </div>;
}
