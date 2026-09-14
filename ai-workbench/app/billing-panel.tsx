'use client';
import { useState, type FormEvent } from 'react';
import { apiJson } from './lib/api';

type Group = { provider: string; category: string; cached_tokens?: number; uncached_tokens?: number; output_tokens: number; unclassified_requests?: number; token_costs?: Record<string, Record<string, number>> };
export type BillingReport = { pending_reasons?: Record<string, number>; by_provider: Group[] };
const labels: Record<string, string> = { input: '未缓存输入', cached_input: '缓存输入', output: '输出' };
const amount = (n: number) => n.toLocaleString('en-US', { maximumFractionDigits: 8 });

export function BillingPanel({ report, refresh }: { report: BillingReport | null; refresh: () => Promise<void> }) {
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');
  async function action(path: string, body?: unknown) {
    setBusy(true); setMessage('');
    try {
      const result = await apiJson<{ message?: string }>(path, { method: 'POST', ...(body ? { body: JSON.stringify(body) } : {}) });
      setMessage(result.message || '配置已保存。');
      await refresh();
    } catch (error) { setMessage(error instanceof Error ? error.message : '操作失败') }
    finally { setBusy(false) }
  }
  function mapping(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const data = new FormData(event.currentTarget);
    void action('/api/costs/model-mappings', { model: data.get('model'), target: data.get('target'), source: data.get('source'), effective_at: new Date(String(data.get('start'))).toISOString(), review_after: new Date(String(data.get('end'))).toISOString(), confirmed: data.get('confirmed') === 'on' });
  }
  return <div className="merged-billing">
    <div className="merged-billing-head"><div><h3>用量拆分与核算状态</h3><span>累计分项用量与核算金额</span></div></div>
    <div className="cost-note">
      {report?.by_provider.filter(g => g.category === 'llm').map(g => <div key={g.provider} className="billing-token-grid"><strong>{g.provider}</strong><span>未缓存输入 {amount(g.uncached_tokens || 0)} Tokens</span><span>缓存输入 {amount(g.cached_tokens || 0)} Tokens</span><span>输出 {amount(g.output_tokens)} Tokens</span><small>缓存分类不完整：{g.unclassified_requests || 0} 条（未计入前两项）</small>{Object.entries(g.token_costs || {}).map(([currency, costs]) => <p key={currency}>{currency} 已核算分项：{Object.entries(costs).map(([key, value]) => `${labels[key]} ${amount(value)}`).join(' · ')}</p>)}</div>)}
      <h3>Tavily 用量</h3>
      {report?.by_provider.filter(g => g.category === 'search').map(g => <p key={g.provider}>本项目累计记录的 credits 按 $0.008/credit 统一估算，不抵扣每月免费额度。</p>)}
      <details className="price-editor"><summary>配置有依据的模型映射</summary><p>新调用会保留请求和返回模型名。确认映射及有效期间后，可按目标价格核算后续具有完整用量的数据。</p><form onSubmit={mapping}>
        <label>记录中的模型名<input name="model" defaultValue="deepseek-flash" required /></label><label>计价模型<input name="target" defaultValue="deepseek-v4-flash" required /></label>
        <label>有效起点（本机时间）<input name="start" type="datetime-local" required /></label><label>复核截止（本机时间）<input name="end" type="datetime-local" required /></label><label>确认依据<input name="source" required placeholder="实际请求配置／供应商确认" /></label><label><input type="checkbox" name="confirmed" required />已确认该期间实际使用目标模型及价格</label><button disabled={busy}>保存映射并补算</button>
      </form></details>
      {message && <p role="status">{message}</p>}
    </div>
  </div>;
}

export function UsageBreakdown({ event }: { event: { category: string; model?: string; requested_model?: string; pricing_model?: string; pricing_basis?: string; usage_note?: string; breakdown?: Record<string, { tokens: number; rate: number; amount: number }> | null; currency?: string | null; quota?: { free_credits: number; paid_credits: number; gross_amount: number }; quota_note?: string } }) {
  return <>{event.category === 'llm' && <p>原始返回模型：{event.model || '未记录'} · 请求模型：{event.requested_model || '历史未记录'} · 计价模型：{event.pricing_model || '未匹配'}{event.pricing_basis === 'requested_model' && '（按请求模型估算，返回名不同）'}{event.pricing_basis === 'confirmed_alias' && '（按已确认映射）'}{event.pricing_basis === 'official_model_family' && '（按 DeepSeek 官方兼容模型族估算）'}{event.pricing_basis === 'deepseek_flash_fallback' && '（模型名未确认，按 deepseek-flash 估算）'}</p>}{Object.entries(event.breakdown || {}).map(([key, row]) => <p key={key}>{labels[key]}：{amount(row.tokens)} Tokens × {amount(row.rate)} / 百万 = {amount(row.amount)} {event.currency}</p>)}{event.usage_note && <p>{event.usage_note}</p>}{event.quota && <p>本次 credits：免费 {event.quota.free_credits} · 付费 {event.quota.paid_credits}；抵扣前估算 ${amount(event.quota.gross_amount)}</p>}{event.quota_note && <p>{event.quota_note}</p>}</>;
}
