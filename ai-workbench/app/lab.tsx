'use client';

/* 实验室 — 一条主线：筛选 → 委员会 → 回测 → 观察 → 批准（加入 Watchlist）。
   每个工具左侧是真实可调的参数，右侧是该工具自己的结果；所有运行落库，可在「运行记录」里重开。 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ApiError, apiJson } from './lib/api';

// ----------------------------------------------------------------------------- types

export type LabTool = 'overview' | 'screening' | 'committee' | 'backtest' | 'event-study' | 'scoreboard' | 'runs';
export const labMenu: { id: LabTool; label: string; icon: string }[] = [
  { id: 'overview', label: '概览', icon: '⌘' },
  { id: 'screening', label: '股票筛选', icon: '⌕' },
  { id: 'committee', label: '投资人委员会', icon: '⚖' },
  { id: 'backtest', label: '策略回测', icon: '↗' },
  { id: 'event-study', label: '事件研究', icon: '∿' },
  { id: 'scoreboard', label: '观察记分板', icon: '◎' },
  { id: 'runs', label: '运行记录', icon: '◷' },
];

type Universe = 'custom' | 'tech30' | 'sp500' | 'nasdaq100' | 'dow30' | 'holdings' | 'watchlist' | 'holdings_watchlist';
const INDEX_UNIVERSES: Universe[] = ['sp500', 'nasdaq100', 'dow30'];
const UNIVERSES: { id: Universe; label: string }[] = [
  { id: 'holdings', label: '当前持仓' }, { id: 'watchlist', label: 'Watchlist' }, { id: 'holdings_watchlist', label: '持仓 + Watchlist' }, { id: 'tech30', label: 'TECH_30 监控池' }, { id: 'dow30', label: '道琼斯 30' }, { id: 'nasdaq100', label: '纳斯达克 100' }, { id: 'sp500', label: '标普 500' }, { id: 'custom', label: '自定义' },
];
const UNIVERSE_LABEL: Record<string, string> = Object.fromEntries(UNIVERSES.map(u => [u.id, u.label]));

type ScreenCandidate = { ticker: string; price: number; price_change: number | null; market_cap: number | null; revenue_growth: number | null; gross_margin: number | null; volatility: number | null; high_52w: number | null; return_1w?: number | null; revenue_actual?: number | null; revenue_estimate?: number | null; [key: string]: unknown };
type ScreenRule = { field: string; op: 'gte' | 'lte'; value: number };
type CriterionMeta = { label: string; unit: 'pct' | 'usd' | 'x'; source: 'metrics' | 'prices' };
type CriteriaResp = { items: Record<string, CriterionMeta>; defaults: ScreenRule[] };
type LabJob<T> = { job_id: string; status: 'running' | 'completed' | 'failed'; done: number; total: number; universe: string; error?: string; result?: T };
type ScreeningJob = LabJob<ScreeningResult>;
type UniverseInfo = { size: number; as_of: string | null; label: string };
type Pricing = { prices_usd: Record<string, number>; committee_per_ticker: { full: number; lean: number } };
type ScreeningResult = { kind: 'screening'; lab_run_id?: string; universe: string; universe_as_of?: string | null; skipped?: Record<string, string>; data_source?: 'yfinance' | 'fd'; with_earnings?: boolean; fd_requests?: Record<string, number>; fd_cost_usd?: number; tickers: string[]; rules?: ScreenRule[]; rules_text?: string[]; rejected_count?: number; no_data?: string[]; reject_reasons?: Record<string, number>; date: string; universe_size: number; candidates: ScreenCandidate[] };

type CommitteeSource = 'tickers' | 'holdings' | 'watchlist' | 'screening';
type PersonaMeta = { key: string; name: string; name_zh: string; style: string; period: string; lookback: number; needs: string[] };
type CommitteePart = { name: string; score: number; max_score: number; details: string };
type CommitteeSignal = { persona: string; ticker: string; as_of: string; signal: 'bullish' | 'bearish' | 'neutral'; confidence: number; score: number; max_score: number; parts: CommitteePart[]; facts: Record<string, unknown>; margin_of_safety: number | null; reasoning: string; narrative?: string | null; narrative_grounded?: boolean | null; abstained: boolean; data_gaps: string[] };
type CommitteeVerdict = { ticker: string; stance: 'bullish' | 'bearish' | 'neutral' | 'abstain'; consensus: number; bullish: number; bearish: number; neutral: number; abstained: number; voters: number; agreement: number; avg_confidence: number; rank: number | null; signals: CommitteeSignal[]; position?: { weight: number | null; market_value: number; current_price: number | null; unrealized_pl_pct: number | null }; action?: string; action_reason?: string; price?: number | null };
type CommitteeResult = { kind: 'committee'; run_id: string; lab_run_id?: string; source: CommitteeSource; as_of: string; personas: string[]; personas_meta: PersonaMeta[]; elapsed_s: number; errors: Record<string, string>; cache_hits: string[]; data_gaps?: { gap: string; tickers: string[] }[]; lean?: boolean; fd_requests?: Record<string, number>; fd_cost_usd?: number; verdicts: CommitteeVerdict[]; top: { rank: number; ticker: string; stance: string; consensus: number }[]; screening?: { universe_size: number | null; n_candidates: number } };

type Trade = { ticker: string; direction: string; entry_date: string; exit_date: string; entry_price: number; exit_price: number; pnl: number; return_pct: number; holding_days: number; metadata?: Record<string, unknown> };
type PeriodVote = { ticker: string; consensus: number; agreement: number; voters: number; abstained: number; bullish: number; bearish: number; neutral: number; gaps: number; reasons?: string[]; picked: boolean };
type BacktestAbort = { signal_date: string; as_of: string; failed: number; of: number; reason: string; remaining_dates: string[] };
type BacktestPeriod = { signal_date: string; as_of: string; picked: string[]; verdicts: PeriodVote[]; missing: string[] };
type Membership = { point_in_time: boolean; changes: number; mode?: 'full' | 'additions' | 'none'; history_from?: string | null; tickers_incl_former?: number };
type YearRow = { year: string; periods: number; trades: number; start: string; end: string; pnl: number; start_equity: number; return_pct: number | null; win_rate: number | null; benchmark_pct: number | null; excess_pct: number | null; return_on_deployed_pct?: number | null; excess_on_deployed_pct?: number | null };
type Deployment = { positions_per_period: number; per_trade: number; deployed_usd: number; capital: number; utilization: number; on_deployed: { total_return_pct: number; annualized_return_pct: number | null; max_drawdown_pct: number; excess_return_pct: number | null } };
type BacktestResult = { kind: 'backtest'; lab_run_id?: string; yearly?: YearRow[]; deployment?: Deployment | null; strategy: string; data_source?: string; universe: string; tickers: string[]; params: Record<string, unknown>; fd_requests?: Record<string, number>; fd_cost_usd?: number; notes?: { price_failures?: Record<string, string>; errors?: Record<string, string>; rebalance_dates?: string[]; periods?: BacktestPeriod[]; aborted?: BacktestAbort | null; no_data?: string[]; membership?: Membership | null }; benchmark?: { ticker: string; start: string; end: string; total_return_pct: number; annualized_return_pct: number | null } | null; excess_return_pct?: number | null; universe_as_of?: string | null; trades: Trade[]; metrics: { total_return_pct: number; annualized_return_pct: number; sharpe_ratio: number; max_drawdown_pct: number; win_rate: number; n_trades: number; n_long: number; n_short: number; avg_return_pct: number; avg_holding_days: number; n_periods?: number; sharpe_trade_level?: number; cost_bps?: number } | null; equity_curve: number[] };

type SweepRow = { top_n: number; holding_days: number; near_high_pct: number | null; per_trade?: number; n_trades: number; n_periods: number; total_return_pct: number | null; annualized_return_pct: number | null; sharpe_ratio: number | null; max_drawdown_pct: number | null; win_rate: number | null; avg_return_pct: number | null; benchmark_pct: number | null; excess_return_pct: number | null; start: string | null; end: string | null };
type SweepResult = { kind: 'sweep'; lab_run_id?: string; strategy: string; data_source?: string; universe: string; universe_as_of?: string | null; tickers: string[]; params: { history_days: number; lookback_days: number; skip_days: number; capital: number; sizing?: string; cost_bps: number; grid: { top_ns: number[]; holding_days_list: number[]; near_high_pcts: (number | null)[] } }; fd_requests?: Record<string, number>; fd_cost_usd?: number; notes?: { price_failures?: Record<string, string>; membership?: Membership | null; no_data?: string[] }; rows: SweepRow[] };
type BacktestPanelResult = BacktestResult | SweepResult;

type WindowStats = { window: string; n_events: number; mean_car: number; std_car: number; t_stat: number; p_value: number; ci: { lower: number; upper: number; confidence: number } };
type EventCAR = { ticker: string; event_date: string; source_type: string; eps_surprise: string | null; car_0_1: number | null; car_0_5: number | null; car_0_20: number | null; car_2_20?: number | null; market_model: { alpha: number; beta: number; r_squared: number; n_obs?: number } };
type EventStudyResult = { kind: 'event_study'; lab_run_id?: string; universe: string; tickers: string[]; data_source?: string; fd_requests?: Record<string, number>; fd_cost_usd?: number; price_failures?: Record<string, string>; params: Record<string, unknown>; events: EventCAR[]; aggregates: { source_type: string; group?: string; n_events: number; windows: WindowStats[] }[]; skipped_tickers: string[]; dedupe?: boolean; group_by?: string };
const GROUP_LABEL: Record<string, string> = { ALL: '全部事件', BEAT: '超预期（BEAT）', MISS: '不及预期（MISS）', MEET: '符合预期（MEET）', UNLABELED: '无 EPS 标注', REACT_UP: '公告日反应最强的 1/3', REACT_MID: '公告日反应中间 1/3', REACT_DOWN: '公告日反应最弱的 1/3' };

type ScoreboardRow = { persona: string; name_zh?: string; n: number; hits: number; hit_rate: number | null; avg_directional_1m: number | null; ci_low?: number | null; ci_high?: number | null; baseline_1m?: number | null; edge_1m?: number | null; n_3m?: number; hits_3m?: number; hit_rate_3m?: number | null; avg_directional_3m?: number | null; baseline_3m?: number | null; edge_3m?: number | null; neutral?: number; abstained?: number; votes?: number };
type ScoreBaseline = { n_1m: number; up_rate_1m: number | null; avg_return_1m: number | null; n_3m: number; up_rate_3m: number | null; avg_return_3m: number | null };
type Scoreboard = { items: ScoreboardRow[]; counts: { runs: number; tickers: number; votes: number; scored_1m: number; scored_3m: number; due_1m: number; due_3m: number }; baseline?: ScoreBaseline };
/** Parameters of a stored run, handed back to a tool so its form matches the result it shows. */
type Restore = { params: Record<string, unknown>; nonce: number };
type RunSummary = { id: string; kind: string; ran_at: string; tickers?: string[]; [key: string]: unknown };
type WatchlistItem = { ticker: string; added_at: string; note: string };

type ToolResult = ScreeningResult | CommitteeResult | BacktestPanelResult | EventStudyResult;
type Handoff = { tickers: string[]; from: string };

// --------------------------------------------------------------------------- helpers

const pct = (v: number | null | undefined, d = 1) => v == null || !Number.isFinite(v) ? '—' : `${v >= 0 ? '+' : ''}${(v * 100).toFixed(d)}%`;
const pctAbs = (v: number | null | undefined, d = 0) => v == null || !Number.isFinite(v) ? '—' : `${(v * 100).toFixed(d)}%`;
const num = (v: number | null | undefined, d = 2) => v == null || !Number.isFinite(v) ? '—' : v.toFixed(d);
const usd = (v: number | null | undefined) => v == null || !Number.isFinite(v) ? '—' : `$${v.toFixed(2)}`;
const money = (v: number | null | undefined) => v == null || !Number.isFinite(v) ? '—' : Math.abs(v) >= 1e12 ? `$${(v / 1e12).toFixed(2)}T` : Math.abs(v) >= 1e9 ? `$${(v / 1e9).toFixed(1)}B` : Math.abs(v) >= 1e6 ? `$${(v / 1e6).toFixed(0)}M` : `$${v.toFixed(2)}`;
const when = (iso: string) => { try { return new Intl.DateTimeFormat('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false }).format(new Date(iso)) } catch { return iso.slice(5, 16) } };
const parseTickers = (text: string) => Array.from(new Set(text.split(/[\s,，;]+/).map(t => t.trim().toUpperCase()).filter(Boolean)));
const errorText = (error: unknown) => error instanceof ApiError ? (error.detail || `HTTP ${error.status}`) : error instanceof Error ? error.message : String(error);
const KIND_LABEL: Record<string, string> = { screening: '股票筛选', committee: '投资人委员会', backtest: '策略回测', event_study: '事件研究', backfill: '收益回填' };
const SIGNAL_GLYPH: Record<string, string> = { bullish: '▲', bearish: '▼', neutral: '·', abstain: '—' };
const SIGNAL_LABEL: Record<string, string> = { bullish: '看多', bearish: '看空', neutral: '中性', abstain: '弃权' };
const STANCE_LABEL: Record<string, string> = { bullish: '偏多', bearish: '偏空', neutral: '中性', abstain: '无票' };

function useLabData<T>(path: string | null, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null); const [error, setError] = useState('');
  const reload = useCallback(() => { if (!path) return; apiJson<T>(path).then(d => { setData(d); setError('') }).catch(e => setError(errorText(e))) }, [path]);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => { const t = window.setTimeout(reload, 0); return () => window.clearTimeout(t) }, [reload, ...deps]);
  return { data, error, reload };
}

/** Poll a Lab job until it settles. A status poll that fails transiently (nginx 502/504 while the
 *  backend is saturated, a dropped connection) is retried with backoff — the job keeps running
 *  server-side, so giving up on the first error would orphan a result that lands minutes later. */
async function pollJob<T>(first: T | LabJob<T>, path: string, onJob: (j: LabJob<T>) => void, failText: string): Promise<T> {
  let r = first; let misses = 0;
  while (typeof r === 'object' && r !== null && 'job_id' in r) {
    const j = r as LabJob<T>; onJob(j);
    if (j.status === 'failed') throw new Error(j.error || failText);
    if (j.status === 'completed' && j.result) { r = j.result; break }
    await new Promise<void>(resolve => globalThis.setTimeout(resolve, Math.min(2000 + misses * 3000, 15000)));
    try { r = await apiJson<LabJob<T>>(`${path}/${encodeURIComponent(j.job_id)}`); misses = 0 }
    catch (e) {
      if (e instanceof ApiError && e.status === 404) throw new Error('任务已不在后端内存里（服务可能重启过）；如果它跑完了，结果会在「运行记录」里。');
      misses += 1;
      if (misses >= 20) throw new Error(`连续 ${misses} 次查询任务状态失败（${errorText(e)}）。任务仍在后台运行，稍后到「运行记录」里查看结果。`);
    }
  }
  return r as T;
}
/** Apply a stored run's parameters to a tool's form once per reopen; `ready` defers until lookups the mapping needs have loaded. */
function useRestore(restore: Restore | undefined, apply: (p: Record<string, unknown>) => void, ready = true) {
  const done = useRef(0);
  useEffect(() => { if (restore && ready && done.current !== restore.nonce) { done.current = restore.nonce; apply(restore.params) } }, [restore, ready, apply]);
}
const str = (v: unknown, fallback: string) => v == null ? fallback : String(v);
function Field({ label, children, hint, block }: { label: string; children: React.ReactNode; hint?: string; block?: boolean }) { const inner = <><span>{label}</span>{children}{hint ? <small>{hint}</small> : null}</>; return block ? <div className="lab-field">{inner}</div> : <label className="lab-field">{inner}</label> }
function NumberInput({ value, onChange, min, max, step }: { value: string; onChange: (v: string) => void; min?: number; max?: number; step?: number }) { return <input type="number" value={value} min={min} max={max} step={step} onChange={e => onChange(e.target.value)}/> }
function Chips<T extends string>({ options, value, onChange }: { options: { id: T; label: string }[]; value: T; onChange: (v: T) => void }) { return <div className="lab-chips">{options.map(o => <button key={o.id} type="button" className={o.id === value ? 'active' : ''} onClick={() => onChange(o.id)}>{o.label}</button>)}</div> }
function UniversePicker({ universe, setUniverse, tickers, setTickers, exclude = [], info, limit = 60 }: { universe: Universe; setUniverse: (u: Universe) => void; tickers: string; setTickers: (t: string) => void; exclude?: Universe[]; limit?: number; info?: Record<string, UniverseInfo> | null }) {
  const meta = info?.[universe];
  return <><Field label="股票池" hint={meta ? `${meta.size} 只${meta.as_of ? ` · 成分股快照 ${meta.as_of}` : ''}${INDEX_UNIVERSES.includes(universe) ? ' · 超过 40 只会转为后台任务，可离开页面' : ''}` : undefined}><Chips options={UNIVERSES.filter(u => !exclude.includes(u.id))} value={universe} onChange={setUniverse}/></Field>{universe === 'custom' && <Field label="股票代码" hint={`逗号或空格分隔，最多 ${limit} 只${parseTickers(tickers).length > limit ? `（当前 ${parseTickers(tickers).length} 只，超出部分会被拒绝）` : ''}`}><input value={tickers} onChange={e => setTickers(e.target.value.toUpperCase())} placeholder="AAPL, MSFT, NVDA"/></Field>}</>;
}
function Empty({ glyph, title, text }: { glyph: string; title: string; text: string }) { return <div className="lab-empty"><span>{glyph}</span><h3>{title}</h3><p>{text}</p></div> }
function ErrorBox({ text }: { text: string }) { return <div className="lab-error"><strong>请求失败</strong><span>{text}</span></div> }
function Stat({ label, value, tone }: { label: string; value: string; tone?: string }) { return <div className="lab-stat"><span>{label}</span><strong className={tone || ''}>{value}</strong></div> }
function RawJson({ data }: { data: unknown }) { return <details className="lab-raw"><summary>原始结果</summary><pre>{JSON.stringify(data, null, 2)}</pre></details> }
function AddToWatchlist({ ticker, watchlist, onAdded }: { ticker: string; watchlist: Set<string>; onAdded: () => void }) {
  const [busy, setBusy] = useState(false);
  if (watchlist.has(ticker)) return <em className="lab-tag-ok">已在 Watchlist</em>;
  return <button type="button" className="lab-link" disabled={busy} onClick={async () => { setBusy(true); try { await apiJson('/api/watchlist', { method: 'POST', body: JSON.stringify({ ticker, note: '来自实验室' }) }); onAdded() } finally { setBusy(false) } }}>{busy ? '…' : '+ Watchlist'}</button>;
}

// ------------------------------------------------------------------------------ page

type Ask = (prompt: string, context: string) => void;
type ToolProps = { ask: Ask; selectTool: (t: LabTool) => void; watchlist: Set<string>; refreshWatchlist: () => void };

export function LabPage({ tool, selectTool, ask }: { tool: LabTool; selectTool: (t: LabTool) => void; ask: Ask; askNow?: Ask }) {
  const [results, setResults] = useState<Partial<Record<string, ToolResult>>>({});
  const [restores, setRestores] = useState<Partial<Record<string, Restore>>>({});
  const [handoff, setHandoff] = useState<Handoff | null>(null);
  const wl = useLabData<{ items: WatchlistItem[] }>('/api/watchlist');
  const watchlist = useMemo(() => new Set((wl.data?.items || []).map(i => i.ticker)), [wl.data]);
  const setResult = useCallback((kind: string, r: ToolResult | undefined) => setResults(current => ({ ...current, [kind]: r })), []);
  const hand = useCallback((tickers: string[], from: string, to: LabTool) => { setHandoff({ tickers, from }); selectTool(to) }, [selectTool]);
  const common: ToolProps = { ask, selectTool, watchlist, refreshWatchlist: wl.reload };
  const heading: Record<LabTool, [string, string]> = {
    overview: ['实验室', '筛选 → 委员会 → 回测 → 观察 → 批准。每一步都是确定性的引擎，结果全部落库；实验室不写任何生产状态。'],
    screening: ['股票筛选', '基本面硬规则过滤股票池，得到候选名单，可直接送入委员会。'],
    committee: ['投资人委员会', '13 位模拟投资人各按一份确定性清单打分，按置信度加权投票；LLM 只在你点「解读」时写文字。'],
    backtest: ['策略回测', '四个策略（PEAD、价格动量、内部人集中买入、投资人委员会）在历史上模拟交易，给出收益、回撤和胜率。'],
    'event-study': ['事件研究', '财报公布后的累计异常收益（CAR）及其显著性。'],
    scoreboard: ['观察记分板', '委员会每一票在 1 个月 / 3 个月后对不对：无前视的逐人命中率。'],
    runs: ['运行记录', '所有工具的历史运行，点开可原样重看。'],
  };
  return <div className={`page lab-page lab-big${tool === 'screening' || tool === 'backtest' || tool === 'event-study' ? ' lab-page-fill' : ''}`}>
    <div className="page-heading"><div><h1>{heading[tool][0]}</h1><p>{heading[tool][1]}</p></div><span className="lab-tag">ISOLATED LAB</span></div>
    {tool === 'overview' && <OverviewTool {...common}/>}
    {tool === 'screening' && <ScreeningTool {...common} result={results.screening as ScreeningResult | undefined} setResult={r => setResult('screening', r)} onHand={hand} restore={restores.screening}/>}
    {tool === 'committee' && <CommitteeTool {...common} result={results.committee as CommitteeResult | undefined} setResult={r => setResult('committee', r)} handoff={handoff} clearHandoff={() => setHandoff(null)} onHand={hand} restore={restores.committee}/>}
    {tool === 'backtest' && <BacktestTool {...common} result={results.backtest as BacktestPanelResult | undefined} setResult={r => setResult('backtest', r)} handoff={handoff} clearHandoff={() => setHandoff(null)} restore={restores.backtest}/>}
    {tool === 'event-study' && <EventStudyTool {...common} result={results.event_study as EventStudyResult | undefined} setResult={r => setResult('event_study', r)} handoff={handoff} clearHandoff={() => setHandoff(null)} restore={restores.event_study}/>}
    {tool === 'scoreboard' && <ScoreboardTool {...common}/>}
    {tool === 'runs' && <RunsTool {...common} onOpen={(kind, result, params) => { setResult(kind, result as ToolResult); setRestores(current => ({ ...current, [kind]: { params, nonce: Date.now() } })); selectTool(kind === 'event_study' ? 'event-study' : kind as LabTool) }}/>}
  </div>;
}

function HandoffBanner({ handoff, onUse, onClear }: { handoff: Handoff; onUse: () => void; onClear: () => void }) {
  return <div className="lab-handoff"><span>来自「{handoff.from}」的 {handoff.tickers.length} 只：{handoff.tickers.slice(0, 12).join(' ')}{handoff.tickers.length > 12 ? ' …' : ''}</span><button type="button" onClick={onUse}>用作股票池</button><button type="button" className="lab-link" onClick={onClear}>忽略</button></div>;
}

// -------------------------------------------------------------------------- overview

function OverviewTool({ selectTool, watchlist, ask }: ToolProps) {
  const runs = useLabData<{ items: RunSummary[]; counts: Record<string, number> }>('/api/lab/runs?limit=8');
  const board = useLabData<Scoreboard>('/api/lab/committee/scoreboard');
  const latest = (kind: string) => runs.data?.items.find(r => r.kind === kind);
  const scr = latest('screening'); const com = latest('committee'); const bt = latest('backtest'); const counts = board.data?.counts;
  const steps: { id: LabTool; title: string; line: string; count: string }[] = [
    { id: 'screening', title: '① 筛选', line: scr ? `${when(scr.ran_at)} · ${UNIVERSE_LABEL[String(scr.universe)] || scr.universe} ${scr.universe_size} → ${scr.n_candidates} 只候选` : '还没跑过', count: String(runs.data?.counts.screening || 0) },
    { id: 'committee', title: '② 委员会', line: com ? `${when(com.ran_at)} · ${com.n_tickers} 只 · 偏多 ${(com.stances as Record<string, number>)?.bullish ?? 0} / 偏空 ${(com.stances as Record<string, number>)?.bearish ?? 0}` : '还没跑过', count: String(runs.data?.counts.committee || 0) },
    { id: 'backtest', title: '③ 回测', line: bt ? `${when(bt.ran_at)} · ${bt.n_trades} 笔 · 总收益 ${pct(bt.total_return_pct as number)} · 夏普 ${num(bt.sharpe_ratio as number)}` : '还没跑过', count: String(runs.data?.counts.backtest || 0) },
    { id: 'scoreboard', title: '④ 观察', line: counts ? `${counts.votes} 票 · 已评分 ${counts.scored_1m} / 待评分 ${counts.due_1m}（1 月）` : '…', count: String(counts?.scored_1m ?? 0) },
    { id: 'runs', title: '⑤ 批准', line: `Watchlist ${watchlist.size} 只 · 生产阈值只读`, count: String(watchlist.size) },
  ];
  return <>
    <section className="surface lab-pipeline">{steps.map(s => <button key={s.id} type="button" onClick={() => selectTool(s.id)}><strong>{s.title}</strong><em>{s.count}</em><span>{s.line}</span></button>)}</section>
    <div className="lab-two">
      <section className="surface"><div className="surface-header"><div><h2>最近运行</h2><span>所有工具 · 落库</span></div><button type="button" onClick={runs.reload}>刷新</button></div>
        {runs.error ? <ErrorBox text={runs.error}/> : !runs.data?.items.length ? <Empty glyph="◷" title="还没有运行记录" text="从「股票筛选」或「投资人委员会」开始。"/> : <div className="lab-runs">{runs.data.items.map(r => <RunRow key={r.id} run={r} onOpen={() => selectTool('runs')}/>)}</div>}
      </section>
      <section className="surface"><div className="surface-header"><div><h2>引擎</h2><span>全部确定性，无 LLM 判决</span></div></div>
        <div className="lab-engines">
          {[['股票筛选', '市值 / 营收增长 / 毛利率 / 波动率硬规则', 'screening'], ['投资人委员会', '13 套打分清单 · 置信度加权投票', 'committee'], ['策略回测', 'PEAD · 动量 · 内部人 · 委员会 · 逐笔交易', 'backtest'], ['事件研究', '市场模型 + bootstrap 置信区间', 'event-study'], ['前向记分', '每天 02:30 ET 回填真实收益', 'scoreboard']].map(([n, d, t]) => <button key={n} type="button" onClick={() => selectTool(t as LabTool)}><strong>{n}</strong><span>{d}</span></button>)}
        </div>
        <div className="guardrail"><strong>生产保护</strong><span>实验室不创建告警、不下单、不改监控阈值；唯一的「批准」动作是把股票加进 Watchlist。</span></div>
      </section>
    </div>
    <button type="button" className="explain-button lab-ask" onClick={() => ask('解释实验室五步流程（筛选、委员会、回测、观察、批准）各自解决什么问题', '实验室 · 概览')}>问 AI：这套流程怎么用</button>
  </>;
}

function RunRow({ run, onOpen, onDelete }: { run: RunSummary; onOpen?: () => void; onDelete?: () => void }) {
  const costTag = typeof run.fd_cost_usd === 'number' ? ` · FD ${usd(run.fd_cost_usd)}` : '';
  const pool = UNIVERSE_LABEL[String(run.universe)] || String(run.universe || '');
  const line = run.kind === 'screening' ? `${pool} ${run.universe_size} → ${run.n_candidates} 只${costTag}`
    : run.kind === 'committee' ? `${SOURCE_LABEL[String(run.source)] || run.source || ''} · ${run.n_tickers} 只 · 偏多 ${(run.stances as Record<string, number>)?.bullish ?? 0} 偏空 ${(run.stances as Record<string, number>)?.bearish ?? 0}${(run.top as string[])?.length ? ` · 榜首 ${(run.top as string[])[0]}` : ''}${costTag}`
    : run.kind === 'backtest' && run.sweep ? `动量参数扫描 · ${pool} · ${run.n_combos} 组 · 最佳夏普 ${num(run.sharpe_ratio as number)}${run.best ? `（每期 ${(run.best as SweepRow).top_n} 只 · 持有 ${(run.best as SweepRow).holding_days} 日）` : ''}`
    : run.kind === 'backtest' ? `${STRATEGY_LABEL[String(run.strategy)] || run.strategy} · ${pool} · ${run.n_trades} 笔 · ${pct(run.total_return_pct as number)} · 夏普 ${num(run.sharpe_ratio as number)}${typeof run.excess_return_pct === 'number' ? ` · 超额 ${pct(run.excess_return_pct)}` : ''}${costTag}`
    : run.kind === 'event_study' ? `${pool} · ${run.n_events} 个事件 · ${run.n_groups} 组${costTag}`
    : run.kind === 'backfill' ? `回填 ${run.filled} / ${run.checked}` : '';
  return <div className={`lab-run ${onOpen ? '' : 'static'}`} role={onOpen ? 'button' : undefined} tabIndex={onOpen ? 0 : undefined} onClick={onOpen} onKeyDown={e => { if (onOpen && (e.key === 'Enter' || e.key === ' ')) onOpen() }}>
    <em className={`kind-${run.kind}`}>{KIND_LABEL[run.kind] || run.kind}</em><span>{line}</span>
    <span className="lab-run-end"><time>{when(run.ran_at)}</time>{onDelete && <button type="button" className="lab-run-del" title="删除这条记录" onClick={e => { e.stopPropagation(); onDelete() }}>×</button>}</span>
  </div>;
}
const SOURCE_LABEL: Record<string, string> = { tickers: '指定股票', holdings: '当前持仓', watchlist: 'Watchlist', screening: '筛选结果' };

// ------------------------------------------------------------------------- screening

/** Order, default operator and default value for each criterion; labels/units come from the backend catalog. */
const CRITERIA_UI: { field: string; op: 'gte' | 'lte'; def: string; hint?: string }[] = [
  { field: 'market_cap', op: 'gte', def: '10' }, { field: 'price', op: 'gte', def: '5' },
  { field: 'revenue_growth', op: 'gte', def: '5' }, { field: 'earnings_growth', op: 'gte', def: '10' },
  { field: 'gross_margin', op: 'gte', def: '50' }, { field: 'operating_margin', op: 'gte', def: '15' }, { field: 'net_margin', op: 'gte', def: '10' },
  { field: 'return_on_equity', op: 'gte', def: '15' }, { field: 'return_on_invested_capital', op: 'gte', def: '10' },
  { field: 'debt_to_equity', op: 'lte', def: '1' }, { field: 'current_ratio', op: 'gte', def: '1' },
  { field: 'price_to_earnings_ratio', op: 'lte', def: '30' }, { field: 'price_to_sales_ratio', op: 'lte', def: '10' }, { field: 'price_to_book_ratio', op: 'lte', def: '10' },
  { field: 'free_cash_flow_yield', op: 'gte', def: '3' }, { field: 'payout_ratio', op: 'lte', def: '60' },
  { field: 'volatility', op: 'lte', def: '60' }, { field: 'return_1w', op: 'gte', def: '0' }, { field: 'return_1m', op: 'gte', def: '0' }, { field: 'return_3m', op: 'gte', def: '0' },
  { field: 'pct_from_52w_high', op: 'gte', def: '-15' }, { field: 'pct_from_52w_low', op: 'gte', def: '20' },
];
const DEFAULT_ENABLED = ['market_cap', 'revenue_growth', 'gross_margin', 'volatility'];
const unitLabel = (unit: string, field: string) => unit === 'pct' ? '%' : unit === 'usd' ? (field === 'market_cap' ? '十亿美元' : '美元') : '倍';
const toBackend = (field: string, unit: string, raw: string) => { const n = Number(raw); return unit === 'pct' ? n / 100 : unit === 'usd' && field === 'market_cap' ? n * 1e9 : n };
const fmtCell = (unit: string, field: string, v: unknown) => typeof v !== 'number' ? '—' : unit === 'pct' ? pct(v) : unit === 'usd' ? (field === 'market_cap' ? money(v) : `$${num(v)}`) : num(v);


function ScreeningTool({ result, setResult, onHand, watchlist, refreshWatchlist, ask, restore }: ToolProps & { result?: ScreeningResult; setResult: (r?: ScreeningResult) => void; onHand: (t: string[], from: string, to: LabTool) => void; restore?: Restore }) {
  const [universe, setUniverse] = useState<Universe>('tech30'); const [tickers, setTickers] = useState('');
  const criteria = useLabData<CriteriaResp>('/api/lab/screening/criteria');
  const [enabled, setEnabled] = useState<Set<string>>(new Set(DEFAULT_ENABLED));
  const [values, setValues] = useState<Record<string, string>>(() => Object.fromEntries(CRITERIA_UI.map(c => [c.field, c.def])));
  const [ops, setOps] = useState<Record<string, 'gte' | 'lte'>>(() => Object.fromEntries(CRITERIA_UI.map(c => [c.field, c.op])));
  const activeRules = (): ScreenRule[] => CRITERIA_UI.filter(c => enabled.has(c.field) && values[c.field] !== '' && Number.isFinite(Number(values[c.field]))).map(c => ({ field: c.field, op: ops[c.field], value: toBackend(c.field, criteria.data?.items[c.field]?.unit || 'x', values[c.field]) }));
  const [busy, setBusy] = useState(false); const [error, setError] = useState(''); const [picked, setPicked] = useState<Set<string>>(new Set()); const [job, setJob] = useState<ScreeningJob | null>(null);
  const [dataSource, setDataSource] = useState<'yfinance' | 'fd'>('yfinance'); const [withEarnings, setWithEarnings] = useState(false);
  const info = useLabData<{ items: Record<string, UniverseInfo> }>('/api/lab/universes');
  const pricing = useLabData<Pricing>('/api/lab/committee/pricing');
  const catalog = criteria.data?.items;
  useRestore(restore, useCallback((p: Record<string, unknown>) => {
    if (typeof p.universe === 'string') setUniverse(p.universe as Universe);
    if (Array.isArray(p.tickers)) setTickers((p.tickers as string[]).join(', '));
    if (p.data_source === 'fd' || p.data_source === 'yfinance') setDataSource(p.data_source);
    setWithEarnings(!!p.with_earnings);
    const rules = p.rules as ScreenRule[] | null | undefined;
    if (rules && catalog) {
      setEnabled(new Set(rules.map(r => r.field)));
      setOps(current => ({ ...current, ...Object.fromEntries(rules.map(r => [r.field, r.op])) }));
      setValues(current => ({ ...current, ...Object.fromEntries(rules.map(r => { const unit = catalog[r.field]?.unit || 'x'; return [r.field, String(unit === 'pct' ? Math.round(r.value * 1000) / 10 : unit === 'usd' && r.field === 'market_cap' ? r.value / 1e9 : r.value)] })) }));
    }
  }, [catalog]), !!catalog);
  const poolSize = universe === 'custom' ? parseTickers(tickers).length : (info.data?.items[universe]?.size ?? 0);
  const estMetrics = dataSource === 'fd' ? poolSize * (pricing.data?.prices_usd.financial_metrics ?? 0.02) : 0;
  const run = async () => {
    setBusy(true); setError(''); setJob(null);
    try {
      const body = { universe, tickers: universe === 'custom' ? parseTickers(tickers) : [], data_source: dataSource, with_earnings: dataSource === 'fd' && withEarnings, rules: activeRules() };
      const first = await apiJson<ScreeningResult | ScreeningJob>('/api/lab/screening', { method: 'POST', body: JSON.stringify(body) });
      const done = await pollJob<ScreeningResult>(first, '/api/lab/screening/jobs', setJob, '筛选任务失败'); setResult(done); setPicked(new Set(done.candidates.map(c => c.ticker)));
    } catch (e) { setError(errorText(e)) } finally { setBusy(false); setJob(null) }
  };
  const chosen = result ? result.candidates.filter(c => picked.has(c.ticker)).map(c => c.ticker) : [];
  // the event study caps at 20 tickers; the interesting ones for PEAD are the smallest, so hand those over
  const smallest = result ? [...result.candidates].filter(c => picked.has(c.ticker)).sort((a, b) => (Number(a.market_cap) || 0) - (Number(b.market_cap) || 0)).slice(0, 20).map(c => c.ticker) : [];
  return <div className="lab-tool lab-tool-fill">
    <section className="surface lab-config lab-config-fill"><div className="surface-header"><div><h2>筛选条件</h2><span>全部阈值可改，缺数据的股票不通过</span></div></div>
      <div className="lab-config-body lab-scroll">
      <UniversePicker universe={universe} setUniverse={setUniverse} tickers={tickers} setTickers={setTickers} info={info.data?.items}/>
      <Field label="数据源" hint={dataSource === 'yfinance' ? '市值、营收增长、毛利率来自 yfinance，免费；口径：营收增长为最近一季同比，毛利率为 TTM' : `Financial Datasets 按请求计费，约 ${usd(pricing.data?.prices_usd.financial_metrics ?? 0.02)}/只`}><Chips options={[{ id: 'yfinance', label: 'yfinance（免费）' }, { id: 'fd', label: 'Financial Datasets（付费）' }]} value={dataSource} onChange={setDataSource}/></Field>
      {dataSource === 'fd' && <Field label="候选的华尔街财报预期" hint={`每只候选约 ${usd(pricing.data?.prices_usd.earnings ?? 0.02)}`}><Chips options={[{ id: 'no', label: '不取' }, { id: 'yes', label: '取' }]} value={withEarnings ? 'yes' : 'no'} onChange={v => setWithEarnings(v === 'yes')}/></Field>}
      <Field block label={`筛选条件（已启用 ${enabled.size}）`} hint="勾选的条件全部满足才通过；某只股票缺该字段即不通过该条。yfinance 可能缺少部分财务比率，缺失的会显示为 —。"><div className="lab-criteria">{CRITERIA_UI.map(c => { const meta = criteria.data?.items[c.field]; const on = enabled.has(c.field); const unit = meta?.unit || 'x'; return <div key={c.field} className={on ? 'on' : ''}><input type="checkbox" checked={on} onChange={() => setEnabled(cur => { const next = new Set(cur); if (next.has(c.field)) next.delete(c.field); else next.add(c.field); return next })}/><span className="lab-crit-label">{meta?.label || c.field}</span><button type="button" className="lab-op" disabled={!on} onClick={() => setOps(cur => ({ ...cur, [c.field]: cur[c.field] === 'gte' ? 'lte' : 'gte' }))}>{ops[c.field] === 'gte' ? '≥' : '≤'}</button><input type="number" disabled={!on} value={values[c.field]} onChange={e => setValues(cur => ({ ...cur, [c.field]: e.target.value }))}/><small>{unitLabel(unit, c.field)}</small></div> })}</div></Field>
      </div>
      <div className="lab-config-foot">
      <p className="lab-note">预计 Financial Datasets 费用：{estMetrics > 0 ? `≈ ${usd(estMetrics)}（${poolSize} 次指标请求）` : '$0.00'}{withEarnings ? ` + 每只候选 ${usd(pricing.data?.prices_usd.earnings ?? 0.02)}` : ''}。价格来自 /api/lab/committee/pricing，可用 FD_PRICES 环境变量校正。</p>
      <button className="run-button" disabled={busy || !enabled.size || (universe === 'custom' && !parseTickers(tickers).length)} onClick={() => void run()}>{busy ? (job ? `筛选中… ${job.done} / ${job.total}` : '筛选中…') : '运行筛选'}</button>
      {job && <div className="lab-progress"><i style={{ width: `${job.total ? Math.round((job.done / job.total) * 100) : 0}%` }}/></div>}
      </div>
    </section>
    <section className="surface lab-result lab-result-clamp">
      {error ? <ErrorBox text={error}/> : !result ? <Empty glyph="⌕" title="等待筛选" text="选一个股票池、勾选条件，结果是全部通过的候选名单。可以整单送进委员会。"/> : <>
        <div className="surface-header"><div><h2>{result.candidates.length} / {result.universe_size} 只通过</h2><span>{UNIVERSE_LABEL[result.universe] || result.universe}{result.universe_as_of ? `（成分股 ${result.universe_as_of}）` : ''} · {result.date} · {result.data_source === 'fd' ? 'Financial Datasets' : 'yfinance'} · FD 费用 {usd(result.fd_cost_usd ?? 0)}{(result.no_data?.length || 0) + Object.keys(result.skipped || {}).length ? ` · 无数据跳过 ${new Set([...(result.no_data || []), ...Object.keys(result.skipped || {})]).size} 只` : ''} · 条件：{(result.rules_text || []).join('，') || '无'}</span></div>
          <div className="lab-actions"><button type="button" disabled={!chosen.length} onClick={() => onHand(chosen, '股票筛选', 'committee')}>送入委员会（{chosen.length}）</button><button type="button" disabled={!chosen.length} onClick={() => onHand(chosen, '股票筛选', 'backtest')}>送入回测</button><button type="button" disabled={!chosen.length} onClick={() => onHand(smallest, chosen.length > 20 ? '股票筛选 · 市值最小的 20 只' : '股票筛选', 'event-study')}>送入事件研究{chosen.length > 20 ? '（最小 20 只）' : ''}</button></div></div>
        {result.candidates.length === 0 ? <div className="lab-note">没有股票通过。{result.reject_reasons && Object.keys(result.reject_reasons).length ? ` 最常见的不通过原因：${Object.entries(result.reject_reasons).slice(0, 3).map(([r, n]) => `${r}（${n} 只）`).join('，')}。` : ''}放宽条件，或换一个股票池。</div> : (() => { const core = new Set(['market_cap', 'revenue_growth', 'gross_margin', 'volatility']); const extra = (result.rules || []).map(r => r.field).filter((f, i, a) => !core.has(f) && f !== 'price' && a.indexOf(f) === i); return <div className="lab-table-wrap lab-scroll"><table className="lab-table"><thead><tr><th><input type="checkbox" checked={picked.size === result.candidates.length} onChange={e => setPicked(e.target.checked ? new Set(result.candidates.map(c => c.ticker)) : new Set())}/></th><th>股票</th><th>价格</th><th>1 日</th><th>1 周</th><th>市值</th><th>营收增长</th><th>毛利率</th><th>波动率</th>{extra.map(f => <th key={f}>{criteria.data?.items[f]?.label || f}</th>)}<th></th></tr></thead><tbody>
          {result.candidates.map(c => <tr key={c.ticker}><td><input type="checkbox" checked={picked.has(c.ticker)} onChange={() => setPicked(cur => { const next = new Set(cur); if (next.has(c.ticker)) next.delete(c.ticker); else next.add(c.ticker); return next })}/></td><td><strong>{c.ticker}</strong></td><td>${num(c.price)}</td><td className={(c.price_change || 0) >= 0 ? 'positive' : 'negative'}>{pct(c.price_change)}</td><td className={(c.return_1w || 0) >= 0 ? 'positive' : 'negative'}>{pct(c.return_1w)}</td><td>{money(c.market_cap)}</td><td>{pct(c.revenue_growth)}</td><td>{pctAbs(c.gross_margin)}</td><td>{pctAbs(c.volatility)}</td>{extra.map(f => <td key={f}>{fmtCell(criteria.data?.items[f]?.unit || 'x', f, c[f])}</td>)}<td><AddToWatchlist ticker={c.ticker} watchlist={watchlist} onAdded={refreshWatchlist}/></td></tr>)}
        </tbody></table></div> })()}
        <div className="lab-foot"><button type="button" className="explain-button" onClick={() => ask(`股票筛选结果：${result.candidates.slice(0, 40).map(c => c.ticker).join(', ') || '无'}${result.candidates.length > 40 ? ` 等 ${result.candidates.length} 只` : ''}（股票池 ${result.universe_size} 只，条件：${(result.rules_text || []).join('，')}）。请点评这批候选的共同点和明显遗漏。`, '实验室 · 股票筛选')}>问 AI 点评这批候选</button><RawJson data={result}/></div>
      </>}
    </section>
  </div>;
}

// ------------------------------------------------------------------------- committee

const COMMITTEE_SOURCES: { id: CommitteeSource; label: string; hint: string }[] = [
  { id: 'holdings', label: '当前持仓', hint: '审视每只持仓：增持 / 持有 / 减持候选' },
  { id: 'watchlist', label: 'Watchlist', hint: '观察名单全部排名' },
  { id: 'tickers', label: '自定义', hint: '逗号分隔，最多 60 只，全部排名' },
  { id: 'screening', label: '先筛选再评审', hint: '用默认阈值跑一遍股票筛选，再对候选评审并选出前 N 只' },
];
const FALLBACK_PERSONAS: PersonaMeta[] = [['warren_buffett','Warren Buffett','沃伦·巴菲特'],['charlie_munger','Charlie Munger','查理·芒格'],['ben_graham','Ben Graham','本杰明·格雷厄姆'],['peter_lynch','Peter Lynch','彼得·林奇'],['phil_fisher','Phil Fisher','菲利普·费雪'],['bill_ackman','Bill Ackman','比尔·阿克曼'],['cathie_wood','Cathie Wood','凯茜·伍德'],['michael_burry','Michael Burry','迈克尔·伯里'],['mohnish_pabrai','Mohnish Pabrai','莫尼什·帕伯莱'],['stanley_druckenmiller','Stanley Druckenmiller','斯坦利·德鲁肯米勒'],['aswath_damodaran','Aswath Damodaran','阿斯瓦斯·达摩达兰'],['nassim_taleb','Nassim Taleb','纳西姆·塔勒布'],['rakesh_jhunjhunwala','Rakesh Jhunjhunwala','拉克什·金君瓦拉']].map(([key, name, name_zh]) => ({ key, name, name_zh, style: '', period: '', lookback: 0, needs: [] }));

function CommitteeTool({ result, setResult, handoff, clearHandoff, onHand, watchlist, refreshWatchlist, ask, restore }: ToolProps & { result?: CommitteeResult; setResult: (r?: CommitteeResult) => void; handoff: Handoff | null; clearHandoff: () => void; onHand: (t: string[], from: string, to: LabTool) => void; restore?: Restore }) {
  const [source, setSource] = useState<CommitteeSource>('holdings'); const [tickers, setTickers] = useState('AAPL, MSFT, NVDA');
  const [topN, setTopN] = useState('15'); const [maxWeight, setMaxWeight] = useState('15'); const [useCache, setUseCache] = useState(true); const [lean, setLean] = useState(true);
  const pricing = useLabData<Pricing>('/api/lab/committee/pricing');
  const perTicker = lean ? pricing.data?.committee_per_ticker.lean : pricing.data?.committee_per_ticker.full;
  const knownCount = source === 'tickers' ? parseTickers(tickers).length : source === 'screening' ? Number(topN) || 15 : null;
  const personasData = useLabData<{ items: PersonaMeta[] }>('/api/lab/committee/personas');
  const personas = personasData.data?.items?.length ? personasData.data.items : FALLBACK_PERSONAS;
  const [selected, setSelected] = useState<string[]>(FALLBACK_PERSONAS.map(p => p.key));
  useRestore(restore, useCallback((p: Record<string, unknown>) => {
    if (typeof p.source === 'string') setSource(p.source as CommitteeSource);
    if (Array.isArray(p.tickers) && (p.tickers as string[]).length) setTickers((p.tickers as string[]).join(', '));
    setSelected(Array.isArray(p.personas) && (p.personas as string[]).length ? p.personas as string[] : personas.map(m => m.key));
    if (typeof p.use_cache === 'boolean') setUseCache(p.use_cache);
    if (typeof p.lean === 'boolean') setLean(p.lean);
    if (typeof p.max_weight === 'number') setMaxWeight(String(Math.round(p.max_weight * 100)));
    if (p.top_n != null) setTopN(String(p.top_n));
  }, [personas]));
  const [busy, setBusy] = useState(false); const [error, setError] = useState('');
  const history = useLabData<{ items: { run_id: string; created_at: string; source: string; n_tickers: number; tickers: string[] }[] }>('/api/lab/committee/runs?limit=10', [result?.run_id]);
  const allSelected = selected.length === personas.length;
  const run = async () => {
    setBusy(true); setError('');
    try {
      const body: Record<string, unknown> = { source, personas: allSelected ? undefined : selected, use_cache: useCache, lean, max_weight: (Number(maxWeight) || 15) / 100, ...(source === 'tickers' ? { tickers: parseTickers(tickers) } : {}), ...(source === 'screening' ? { top_n: Number(topN) || 15 } : {}) };
      setResult(await apiJson<CommitteeResult>('/api/lab/committee', { method: 'POST', body: JSON.stringify(body) }));
    } catch (e) { setError(errorText(e)) } finally { setBusy(false) }
  };
  const reopen = async (runId: string) => { setBusy(true); setError(''); try { setResult(await apiJson<CommitteeResult>(`/api/lab/committee/runs/${encodeURIComponent(runId)}`)) } catch (e) { setError(errorText(e)) } finally { setBusy(false) } };
  return <>
    {handoff && <HandoffBanner handoff={handoff} onUse={() => { setSource('tickers'); setTickers(handoff.tickers.join(', ')); clearHandoff() }} onClear={clearHandoff}/>}
    <div className="lab-notice lab-notice-neutral"><strong>打分有效，但依据比设计的浅，评价还要等</strong><span>委员会不是回测：它对当前数据实时评审，每一票对不对由「观察记分板」在 1 个月 / 3 个月后用真实收益判定，这个流程本身没有前视。两条局限要记住：一是 Financial Datasets 的财务报表只有 2–3 个财年，巴菲特、芒格、林奇这类看 10 年的投资人，其增长和一致性指标实际是用 TTM 推算的年度序列打的分，只能看到约 3 年，不是清单要求的 10 年；二是记分板需要时间积累，一位投资人至少要 20 票、经过 1 个月才有可解读的命中率，现在还在积累期，榜上的差异暂时是噪声。</span></div>
    <div className="lab-tool">
      <section className="surface lab-config"><div className="surface-header"><div><h2>评审设置</h2><span>每位投资人一份确定性打分清单 · 按置信度加权投票</span></div></div>
        <Field label="输入来源" hint={COMMITTEE_SOURCES.find(s => s.id === source)?.hint}><Chips options={COMMITTEE_SOURCES} value={source} onChange={setSource}/></Field>
        {source === 'tickers' && <Field label="股票代码"><input value={tickers} onChange={e => setTickers(e.target.value.toUpperCase())}/></Field>}
        {source === 'holdings' && <Field label="单只持仓权重上限（%）" hint="共识看多但权重已达上限时标为「持有」而不是「增持候选」"><NumberInput value={maxWeight} onChange={setMaxWeight} min={1} max={100}/></Field>}
        {source === 'screening' && <Field label="从候选中选出前 N 只"><NumberInput value={topN} onChange={setTopN} min={1} max={60}/></Field>}
        <Field label={`参与投票的投资人（${selected.length}/${personas.length}）`}><div className="persona-chips">{personas.map(p => <button key={p.key} type="button" className={selected.includes(p.key) ? 'active' : ''} title={p.style} onClick={() => setSelected(cur => cur.includes(p.key) ? cur.filter(k => k !== p.key) : [...cur, p.key])}>{p.name_zh || p.name}</button>)}<button type="button" className="lab-link" onClick={() => setSelected(allSelected ? [] : personas.map(p => p.key))}>{allSelected ? '全不选' : '全选'}</button></div></Field>
        <Field label="数据"><Chips options={[{ id: 'cache', label: '复用当日快照' }, { id: 'fresh', label: '重新取数' }]} value={useCache ? 'cache' : 'fresh'} onChange={v => setUseCache(v === 'cache')}/></Field>
        <Field label="省流模式" hint={lean ? '跳过新闻和内部人交易两路付费请求；只影响几位投资人的情绪小分项' : '取全部数据，每只多两次付费请求'}><Chips options={[{ id: 'lean', label: '开（省流）' }, { id: 'full', label: '关（全量）' }]} value={lean ? 'lean' : 'full'} onChange={v => setLean(v === 'lean')}/></Field>
      <p className="lab-note">预计 Financial Datasets 费用：每只约 {usd(perTicker)}{knownCount ? `，${knownCount} 只约 ${usd((perTicker ?? 0) * knownCount)}` : ''}；当日已有快照的股票不再计费。</p>
        <button className="run-button" disabled={busy || !selected.length || (source === 'tickers' && !parseTickers(tickers).length)} onClick={() => void run()}>{busy ? '评审中…（首次取数约 5 秒/只）' : '开始评审'}</button>
        {history.data?.items.length ? <div className="lab-history"><span>历史运行</span>{history.data.items.map(h => <button key={h.run_id} type="button" onClick={() => void reopen(h.run_id)} className={result?.run_id === h.run_id ? 'active' : ''}>{when(h.created_at)} · {COMMITTEE_SOURCES.find(s => s.id === h.source)?.label || h.source} · {h.n_tickers} 只</button>)}</div> : null}
      </section>
      <section className="surface lab-result">
        {error ? <ErrorBox text={error}/> : !result ? <Empty glyph="⚖" title="等待评审" text="结果是一张矩阵：行是投资人，列是股票，格子里是多空与置信度。点任意格子看依据。"/> : <CommitteeView result={result} personas={personas} watchlist={watchlist} refreshWatchlist={refreshWatchlist} ask={ask} onHand={onHand}/>}
      </section>
    </div>
  </>;
}

type NarrativeState = { text: string; grounded: boolean | null } | { error: string } | 'loading';

function CommitteeView({ result, personas, watchlist, refreshWatchlist, ask, onHand }: { result: CommitteeResult; personas: PersonaMeta[]; watchlist: Set<string>; refreshWatchlist: () => void; ask: Ask; onHand: (t: string[], from: string, to: LabTool) => void }) {
  const [picked, setPicked] = useState<{ ticker: string; persona: string } | null>(null);
  const [narratives, setNarratives] = useState<Record<string, NarrativeState>>({});
  const verdicts = useMemo(() => result.verdicts || [], [result.verdicts]);
  const meta = result.personas_meta?.length ? result.personas_meta : personas.filter(p => result.personas.includes(p.key));
  const grid = useMemo(() => { const m = new Map<string, CommitteeSignal>(); verdicts.forEach(v => v.signals.forEach(s => m.set(`${s.persona}|${v.ticker}`, s))); return m }, [verdicts]);
  const pickedSignal = picked ? grid.get(`${picked.persona}|${picked.ticker}`) : undefined; const pickedVerdict = picked ? verdicts.find(v => v.ticker === picked.ticker) : undefined;
  const isHoldings = result.source === 'holdings';
  const narrate = async (ticker: string, persona: string) => {
    const key = `${persona}|${ticker}`; setNarratives(c => ({ ...c, [key]: 'loading' }));
    try { const d = await apiJson<{ narrative: string; narrative_grounded: boolean | null }>('/api/lab/committee/narrate', { method: 'POST', body: JSON.stringify({ run_id: result.run_id, ticker, persona, language: 'zh' }) }); setNarratives(c => ({ ...c, [key]: { text: d.narrative, grounded: d.narrative_grounded } })) }
    catch (e) { setNarratives(c => ({ ...c, [key]: { error: errorText(e) } })) }
  };
  const bullishTickers = verdicts.filter(v => v.stance === 'bullish').map(v => v.ticker);
  return <>
    <div className="surface-header"><div><h2>{verdicts.length} 只 · {meta.length} 位投资人</h2><span>{COMMITTEE_SOURCES.find(s => s.id === result.source)?.label || result.source} · as of {result.as_of} · {num(result.elapsed_s, 1)}s · FD 费用 {usd(result.fd_cost_usd ?? 0)}{result.lean ? '（省流）' : ''}{result.cache_hits?.length ? ` · ${result.cache_hits.length} 只命中当日缓存` : ''}{result.screening ? ` · 初筛 ${result.screening.universe_size ?? '?'} → ${result.screening.n_candidates}` : ''}</span></div>
      <div className="lab-actions"><button type="button" disabled={!bullishTickers.length} onClick={() => onHand(bullishTickers, '委员会偏多', 'backtest')}>偏多的送入回测（{bullishTickers.length}）</button></div></div>
    {Object.keys(result.errors || {}).length > 0 && <div className="committee-errors">{Object.entries(result.errors).map(([t, e]) => <span key={t}><strong>{t}</strong> {e}</span>)}</div>}
    {result.data_gaps && result.data_gaps.length > 0 && <div className="lab-gaps"><strong>数据缺口</strong>{result.data_gaps.map(g => <div key={g.gap}><span>{g.gap}</span><small>{g.tickers.length} 只：{g.tickers.slice(0, 12).join(' ')}{g.tickers.length > 12 ? ' …' : ''}</small></div>)}<p>缺少所需输入的投资人会弃权而不是打低分；弃权票不参与共识。</p></div>}
    <div className="lab-verdicts">{verdicts.map(v => <div key={v.ticker} className={`lab-verdict stance-${v.stance}`}><strong>{v.ticker}</strong><em>{isHoldings ? (v.action || '—') : `#${v.rank} ${STANCE_LABEL[v.stance]}`}</em><span>共识 {v.consensus >= 0 ? '+' : ''}{v.consensus.toFixed(2)} · {v.bullish}▲ {v.bearish}▼ {v.neutral}· {v.abstained ? `${v.abstained}弃` : ''}</span>{isHoldings && v.position ? <span>权重 {pctAbs(v.position.weight, 1)}{v.position.unrealized_pl_pct != null ? ` · 浮盈 ${pct(v.position.unrealized_pl_pct)}` : ''}</span> : null}{isHoldings && v.action_reason ? <small>{v.action_reason}</small> : null}<AddToWatchlist ticker={v.ticker} watchlist={watchlist} onAdded={refreshWatchlist}/></div>)}</div>
    <div className="committee-matrix-wrap"><table className="committee-matrix"><thead><tr><th>投资人</th>{verdicts.map(v => <th key={v.ticker}>{v.ticker}</th>)}</tr></thead><tbody>
      {meta.map(p => <tr key={p.key}><th title={p.style}>{p.name_zh || p.name}</th>{verdicts.map(v => { const s = grid.get(`${p.key}|${v.ticker}`); const kind = !s || s.abstained ? 'abstain' : s.signal; const active = picked?.ticker === v.ticker && picked?.persona === p.key; return <td key={v.ticker}><button type="button" className={`sig sig-${kind} ${active ? 'active' : ''}`} onClick={() => setPicked(active ? null : { ticker: v.ticker, persona: p.key })} title={s ? s.reasoning : '无数据'}>{SIGNAL_GLYPH[kind]}{kind !== 'abstain' && s ? <b>{s.confidence}</b> : null}</button></td> })}</tr>)}
      <tr className="committee-foot"><th>共识</th>{verdicts.map(v => <td key={v.ticker}><strong className={`stance-${v.stance}`}>{v.consensus >= 0 ? '+' : ''}{v.consensus.toFixed(2)}</strong></td>)}</tr>
      <tr className="committee-foot"><th>一致度</th>{verdicts.map(v => <td key={v.ticker}>{pctAbs(v.agreement)}</td>)}</tr>
    </tbody></table></div>
    {pickedSignal && pickedVerdict && <div className="committee-detail"><div className="committee-detail-head"><strong>{meta.find(p => p.key === pickedSignal.persona)?.name_zh || pickedSignal.persona} · {pickedVerdict.ticker}</strong><em className={`sig-text-${pickedSignal.abstained ? 'abstain' : pickedSignal.signal}`}>{SIGNAL_LABEL[pickedSignal.abstained ? 'abstain' : pickedSignal.signal]} {pickedSignal.abstained ? '' : `${pickedSignal.confidence}%`}</em><span>得分 {num(pickedSignal.score)} / {pickedSignal.max_score}{pickedSignal.margin_of_safety != null ? ` · 安全边际 ${pct(pickedSignal.margin_of_safety, 0)}` : ''}</span></div>
      {pickedSignal.abstained ? <p className="lab-note">{pickedSignal.reasoning}</p> : null}
      {pickedSignal.parts.length > 0 && <div className="committee-parts">{pickedSignal.parts.map(part => <div key={part.name}><div className="committee-part-head"><span>{part.name.replaceAll('_', ' ')}</span><strong>{num(part.score, part.score % 1 ? 2 : 0)} / {part.max_score}</strong></div><div className="risk-track"><i style={{ width: `${part.max_score > 0 ? Math.max(0, Math.min(100, (part.score / part.max_score) * 100)) : 0}%` }}/></div><p>{part.details || '—'}</p></div>)}</div>}
      {(() => { const key = `${pickedSignal.persona}|${pickedVerdict.ticker}`; const state = narratives[key]; const stored = pickedSignal.narrative ? { text: pickedSignal.narrative, grounded: pickedSignal.narrative_grounded ?? null } : null; const shown = state && state !== 'loading' && 'text' in state ? state : stored;
        return <div className="committee-narrate">{shown ? <p className="committee-narrative">{shown.text}<em className={shown.grounded === false ? 'ungrounded' : ''}>{shown.grounded === false ? '⚠ 有数字无法溯源' : shown.grounded ? '✓ 数字已溯源' : ''}</em></p> : null}{state && state !== 'loading' && 'error' in state ? <small className="committee-gaps">解读失败：{state.error}</small> : null}{!pickedSignal.abstained ? <button type="button" className="text-action" disabled={state === 'loading'} onClick={() => void narrate(pickedVerdict.ticker, pickedSignal.persona)}>{state === 'loading' ? '解读中…' : shown ? '重新解读' : 'LLM 解读'}</button> : null}</div> })()}
      {pickedSignal.data_gaps.length > 0 && <small className="committee-gaps">数据缺口：{pickedSignal.data_gaps.join('；')}</small>}
    </div>}
    <div className="lab-foot"><button type="button" className="explain-button" onClick={() => ask(`投资人委员会结果（${result.as_of}）：${verdicts.slice(0, 8).map(v => `${v.ticker} 共识${v.consensus >= 0 ? '+' : ''}${v.consensus.toFixed(2)}（${v.bullish}多/${v.bearish}空/${v.neutral}中）`).join('；')}。请解释分歧最大的股票为什么投资人意见不一。`, '实验室 · 投资人委员会')}>问 AI 解释分歧</button><RawJson data={result}/></div>
  </>;
}

// -------------------------------------------------------------------------- backtest

function EquityLine({ values }: { values: number[] }) {
  if (values.length < 2) return null;
  const min = Math.min(...values); const max = Math.max(...values); const span = Math.max(max - min, 1e-9);
  const pts = values.map((v, i) => `${(i / (values.length - 1)) * 760},${150 - ((v - min) / span) * 130}`).join(' ');
  return <svg className="lab-equity" viewBox="0 0 760 160" role="img" aria-label="回测净值曲线"><polyline className="chart-line" points={pts}/></svg>;
}

type StrategyId = 'pead' | 'momentum' | 'insider' | 'committee';
type DataTier = 'free' | 'paid';
const STRATEGIES: { id: StrategyId; tier: DataTier; label: string; short: string; holding: number; hint: string; empty: string }[] = [
  { id: 'pead', tier: 'paid', label: 'PEAD 财报后漂移', short: 'PEAD', holding: 5, hint: '财报 EPS 超预期做多、不及预期做空，财报日后入场持有 N 日。财报历史来自 Financial Datasets，每只 1 次请求。', empty: '在选定股票池的历史财报事件上按 PEAD 规则入场、持有 N 日出场，得到逐笔交易与净值曲线。' },
  { id: 'momentum', tier: 'free', label: '价格动量', short: '动量', holding: 21, hint: '12-1 动量：按「跳过最近 M 日后的 N 日涨幅」排名，每期买入前 K 只、持有一期。只用价格，yfinance 下免费，所以可以用整个标普 500 / 纳斯达克 100 / 道琼斯 30 做股票池。可选只买接近 52 周新高的股票。', empty: '每个换仓日按过去一年的涨幅排名，买入最强的几只，持有一期后换仓。' },
  { id: 'insider', tier: 'paid', label: '内部人集中买入', short: '内部人', holding: 63, hint: '窗口期内有多位不同内部人（高管、董事）净买入且合计金额达标，即在最后一笔申报日次日买入。内部人交易来自 Financial Datasets，每只 1 次请求。', empty: '找出历史上多位内部人在短窗口内集中买入的时点，买入并持有一段时间。' },
  { id: 'committee', tier: 'paid', label: '投资人委员会', short: '委员会', holding: 63, hint: '每个换仓日让 13 位模拟投资人按当时可得的财务数据打分，买入共识最强的前 K 只。每只股票每个时点约 5 次付费请求（省流模式：指标 ×2、财务科目 ×2、市值），快照会缓存，重跑同一时点不再计费；某个时点超过一半股票取数失败会自动停止。', empty: '在历史上每个季度让投资人委员会投票，买入共识最强的几只，看这套打分规则过去是否赚钱。' },
];
const STRATEGY_LABEL: Record<string, string> = Object.fromEntries(STRATEGIES.map(s => [s.id, s.label]));

function signalText(strategy: string, meta?: Record<string, unknown>): string {
  if (!meta) return '';
  const n = (k: string) => typeof meta[k] === 'number' ? (meta[k] as number) : null;
  if (strategy === 'momentum') { const m = n('momentum'); const h = n('pct_from_52w_high'); return `动量 ${pct(m)}${h != null ? ` · 距高点 ${pct(h)}` : ''}`; }
  if (strategy === 'insider') return `${n('insiders') ?? '?'} 位内部人 · $${Math.round(n('cluster_value_usd') ?? 0).toLocaleString()}`;
  if (strategy === 'committee') return `共识 ${num(n('consensus'))} · ${n('bullish') ?? 0}多/${n('bearish') ?? 0}空 · 数据截至 ${String(meta.as_of || '')}`;
  return `${String(meta.eps_surprise || '')} ${String(meta.source_type || '')}`.trim();
}

const SWEEP_GRID = { top_ns: [10, 20, 30], holding: [21, 42, 63], nearHigh: [null, 0.10] as (number | null)[] };
type SweepSort = 'sharpe_ratio' | 'total_return_pct' | 'excess_return_pct' | 'max_drawdown_pct';
function SweepView({ result, ask }: { result: SweepResult; ask: Ask }) {
  const [sort, setSort] = useState<SweepSort>('sharpe_ratio');
  const rows = [...result.rows].sort((a, b) => sort === 'max_drawdown_pct' ? (a.max_drawdown_pct ?? 1) - (b.max_drawdown_pct ?? 1) : (b[sort] ?? -Infinity) - (a[sort] ?? -Infinity));
  const best = rows[0]; const g = result.params.grid; const ms = result.notes?.membership;
  const combo = (r: SweepRow) => `每期 ${r.top_n} 只 · 持有 ${r.holding_days} 日 · ${r.near_high_pct == null ? '不限' : `距新高 ≤ ${Math.round(r.near_high_pct * 100)}%`}`;
  const beat = result.rows.filter(r => (r.excess_return_pct ?? 0) > 0).length;
  return <>
    <div className="surface-header"><div><h2>动量参数扫描 · {result.rows.length} 组 · {result.tickers.length} 只</h2><span>{UNIVERSE_LABEL[result.universe] || result.universe}{result.universe_as_of ? `（成分股 ${result.universe_as_of}）` : ''} · 回看历史 {result.params.history_days} 天 · 动量回看 {result.params.lookback_days} 日 · 跳过 {result.params.skip_days} 日 · 初始资金 ${Number(result.params.capital).toLocaleString()}，每组满仓（单笔 = 资金 ÷ 只数） · 成本 {result.params.cost_bps} bp/边 · 价格 {result.data_source === 'fd' ? 'Financial Datasets' : 'yfinance'} · FD 费用 {usd(result.fd_cost_usd ?? 0)}</span></div></div>
    <div className="lab-scroll">
      {best && <><div className="lab-subhead lab-best">最佳（按{SWEEP_SORT_LABEL[sort]}）：{combo(best)}</div><div className="lab-stats"><Stat label="总收益" value={pct(best.total_return_pct)} tone={(best.total_return_pct ?? 0) >= 0 ? 'positive' : 'negative'}/><Stat label="夏普（按期）" value={num(best.sharpe_ratio)}/><Stat label="最大回撤" value={pct(best.max_drawdown_pct)} tone="negative"/><Stat label="超额收益" value={pct(best.excess_return_pct)} tone={(best.excess_return_pct ?? 0) >= 0 ? 'positive' : 'negative'}/></div></>}
      <p className="lab-note">{beat} / {result.rows.length} 组跑赢同期 SPY。网格：每期 {g.top_ns.join(' / ')} 只 × 持有 {g.holding_days_list.join(' / ')} 日 × 52 周高点过滤 {g.near_high_pcts.map(v => v == null ? '不限' : `≤ ${Math.round(v * 100)}%`).join(' / ')}。所有组用同一份价格数据和同一股票池；{ms?.point_in_time ? `已按历史成分股回测（变动记录 ${ms.changes} 条）` : INDEX_UNIVERSES.includes(result.universe as Universe) ? '股票池是当前成分股快照，存在幸存者偏差' : '自定义股票池'}{result.notes?.no_data?.length ? `，${result.notes.no_data.length} 只无价格数据被跳过` : ''}。注意：在同一段历史上挑选表现最好的参数本身就是过拟合，相邻参数的结果是否平滑比最佳值更重要。</p>
      <div className="lab-table-wrap"><div className="lab-subhead lab-subhead-row"><span>对比表</span><div className="lab-chips">{(Object.keys(SWEEP_SORT_LABEL) as SweepSort[]).map(k => <button key={k} type="button" className={k === sort ? 'active' : ''} onClick={() => setSort(k)}>按{SWEEP_SORT_LABEL[k]}</button>)}</div></div>
        <table className="lab-table"><thead><tr><th>#</th><th>参数组合</th><th>期数</th><th>笔数</th><th>总收益</th><th>年化</th><th>夏普（按期）</th><th>最大回撤</th><th>胜率</th><th>平均单笔</th><th>SPY 同期</th><th>超额</th></tr></thead>
          <tbody>{rows.map((r, i) => <tr key={`${r.top_n}-${r.holding_days}-${r.near_high_pct}`} className={i === 0 ? 'picked' : ''}><td>{i + 1}</td><td>{combo(r)}</td><td>{r.n_periods}</td><td>{r.n_trades}</td><td className={(r.total_return_pct ?? 0) >= 0 ? 'positive' : 'negative'}>{pct(r.total_return_pct)}</td><td>{pct(r.annualized_return_pct)}</td><td><strong>{num(r.sharpe_ratio)}</strong></td><td className="negative">{pct(r.max_drawdown_pct)}</td><td>{pctAbs(r.win_rate)}</td><td>{pct(r.avg_return_pct, 2)}</td><td>{pct(r.benchmark_pct)}</td><td className={(r.excess_return_pct ?? 0) >= 0 ? 'positive' : 'negative'}><strong>{pct(r.excess_return_pct)}</strong></td></tr>)}</tbody></table></div>
    </div>
    <div className="lab-foot"><button type="button" className="explain-button" onClick={() => ask(`动量策略参数扫描（${UNIVERSE_LABEL[result.universe] || result.universe}，${result.tickers.length} 只，回看历史 ${result.params.history_days} 天，成本 ${result.params.cost_bps} bp/边，每组满仓）。${result.rows.length} 组里 ${beat} 组跑赢 SPY。按夏普排序前 5 组：${[...result.rows].sort((a, b) => (b.sharpe_ratio ?? -9) - (a.sharpe_ratio ?? -9)).slice(0, 5).map(r => `${combo(r)}：总收益 ${pct(r.total_return_pct)}，夏普 ${num(r.sharpe_ratio)}，回撤 ${pct(r.max_drawdown_pct)}，超额 ${pct(r.excess_return_pct)}`).join('；')}。请判断参数面是否平滑、哪些结论稳健、哪些只是过拟合。`, '实验室 · 参数扫描')}>问 AI 评价参数稳健性</button><RawJson data={result}/></div>
  </>;
}
const SWEEP_SORT_LABEL: Record<SweepSort, string> = { sharpe_ratio: '夏普', total_return_pct: '总收益', excess_return_pct: '超额', max_drawdown_pct: '回撤（小→大）' };

function BacktestTool({ result: panel, setResult, handoff, clearHandoff, ask, restore }: ToolProps & { result?: BacktestPanelResult; setResult: (r?: BacktestPanelResult) => void; handoff: Handoff | null; clearHandoff: () => void; restore?: Restore }) {
  const result = panel?.kind === 'backtest' ? panel : undefined; const sweep = panel?.kind === 'sweep' ? panel : undefined;
  const [universe, setUniverse] = useState<Universe>('custom'); const [tickers, setTickers] = useState('AAPL, MSFT, NVDA');
  const [tier, setTierRaw] = useState<DataTier>('free'); const [strategy, setStrategyRaw] = useState<StrategyId>('momentum'); const [dataSource, setDataSource] = useState<'yfinance' | 'fd'>('yfinance');
  const [holding, setHolding] = useState('21'); const [earnings, setEarnings] = useState('8'); const [capital, setCapital] = useState('100000'); const [perTrade, setPerTrade] = useState('10000'); const [costBps, setCostBps] = useState('10');
  const [history, setHistory] = useState('730'); const [topN, setTopN] = useState('5'); const [lookback, setLookback] = useState('252'); const [skip, setSkip] = useState('21'); const [nearHigh, setNearHigh] = useState('');
  const [window, setWindowDays] = useState('30'); const [minInsiders, setMinInsiders] = useState('2'); const [minValue, setMinValue] = useState('100000');
  const [minConsensus, setMinConsensus] = useState('0.2'); const [minAgreement, setMinAgreement] = useState('0.5'); const [lean, setLean] = useState(true); const [lag, setLag] = useState('45');
  const [busy, setBusy] = useState(false); const [error, setError] = useState(''); const [job, setJob] = useState<LabJob<BacktestResult> | null>(null);
  const info = useLabData<{ items: Record<string, UniverseInfo> }>('/api/lab/universes');
  const pricing = useLabData<Pricing>('/api/lab/committee/pricing');
  const meta = STRATEGIES.find(s => s.id === strategy)!;
  const setStrategy = (id: StrategyId) => { setStrategyRaw(id); setHolding(String(STRATEGIES.find(s => s.id === id)!.holding)); if (id !== 'momentum' && INDEX_UNIVERSES.includes(universe)) setUniverse('custom'); };
  const setTier = (t: DataTier) => { setTierRaw(t); setStrategy(STRATEGIES.find(s => s.tier === t)!.id); if (t === 'free') setDataSource('yfinance'); };
  useRestore(restore, useCallback((p: Record<string, unknown>) => {
    // a stored backtest (has `strategy`) or a stored sweep (has `top_ns`); both share the universe / pricing / momentum fields
    const id = (typeof p.strategy === 'string' ? p.strategy : 'momentum') as StrategyId;
    const meta = STRATEGIES.find(s => s.id === id) || STRATEGIES[0];
    setTierRaw(meta.tier); setStrategyRaw(meta.id);
    if (typeof p.universe === 'string') setUniverse(p.universe as Universe);
    if (Array.isArray(p.tickers) && (p.tickers as string[]).length) setTickers((p.tickers as string[]).join(', '));
    if (p.data_source === 'fd' || p.data_source === 'yfinance') setDataSource(p.data_source);
    setHolding(str(p.holding_days, String(meta.holding))); setCapital(str(p.capital, '100000')); setPerTrade(str(p.per_trade, '10000')); setCostBps(str(p.cost_bps, '10'));
    setEarnings(str(p.earnings_limit, '8')); setHistory(str(p.history_days, '730')); setTopN(str(p.top_n, '5')); setLookback(str(p.lookback_days, '252')); setSkip(str(p.skip_days, '21'));
    setNearHigh(typeof p.near_high_pct === 'number' ? String(Math.round(p.near_high_pct * 100)) : '');
    setWindowDays(str(p.window_days, '30')); setMinInsiders(str(p.min_insiders, '2')); setMinValue(str(p.min_value_usd, '100000'));
    setMinConsensus(str(p.min_consensus, '0.2')); setMinAgreement(str(p.min_agreement, '0.5')); if (typeof p.lean === 'boolean') setLean(p.lean); setLag(str(p.filing_lag_days, '45'));
  }, []));
  const tierStrategies = STRATEGIES.filter(s => s.tier === tier);
  const price = (k: string) => pricing.data?.prices_usd[k] ?? 0.02;
  const poolSize = universe === 'custom' ? parseTickers(tickers).length : (info.data?.items[universe]?.size ?? 0);
  const step = Math.max(1, Math.round(Number(holding) * 365 / 252));
  const periods = Math.max(0, Math.floor((Number(history) - step) / step) + 1);
  const priceChunks = Math.ceil((Number(history) + (strategy === 'momentum' ? Number(lookback) * 1.6 : 0) + 10) / 90);
  const estimate = (() => {
    if (!poolSize) return null;
    let events = 0; let note = '';
    if (strategy === 'pead') { events = poolSize * price('earnings'); note = '财报历史'; }
    else if (strategy === 'insider') { events = poolSize * price('insider_trades'); note = '内部人交易'; }
    else if (strategy === 'committee') { events = poolSize * periods * (lean ? 5 : 7) * price('financial_metrics'); note = `${periods} 个换仓日 × ${poolSize} 只 × ${lean ? 5 : 7} 次`; }
    const prices = dataSource === 'fd' ? poolSize * priceChunks * price('prices') : 0;
    return { total: events + prices, note, prices };
  })();
  const body = () => ({
    universe, tickers: universe === 'custom' ? parseTickers(tickers) : [], strategy, data_source: dataSource,
    holding_days: Number(holding), capital: Number(capital), per_trade: Number(perTrade), cost_bps: Number(costBps), earnings_limit: Number(earnings),
    history_days: Number(history), top_n: Number(topN), lookback_days: Number(lookback), skip_days: Number(skip), near_high_pct: nearHigh === '' ? null : Number(nearHigh) / 100,
    window_days: Number(window), min_insiders: Number(minInsiders), min_value_usd: Number(minValue),
    min_consensus: Number(minConsensus), min_agreement: Number(minAgreement), lean, filing_lag_days: Number(lag),
  });
  const poll = <T,>(first: T | LabJob<T>) => pollJob<T>(first, '/api/lab/backtest/jobs', j => setJob(j as LabJob<BacktestResult>), '回测任务失败');
  const run = async () => {
    setBusy(true); setError(''); setJob(null);
    try { setResult(await poll(await apiJson<BacktestResult | LabJob<BacktestResult>>('/api/lab/backtest', { method: 'POST', body: JSON.stringify(body()) }))) }
    catch (e) { setError(errorText(e)) } finally { setBusy(false); setJob(null) }
  };
  const runSweep = async () => {
    setBusy(true); setError(''); setJob(null);
    const sweepBody = { universe, tickers: universe === 'custom' ? parseTickers(tickers) : [], data_source: dataSource, history_days: Number(history), lookback_days: Number(lookback), skip_days: Number(skip),
      capital: Number(capital), cost_bps: Number(costBps), top_ns: SWEEP_GRID.top_ns, holding_days_list: SWEEP_GRID.holding, near_high_pcts: SWEEP_GRID.nearHigh };
    try { setResult(await poll(await apiJson<LabJob<SweepResult>>('/api/lab/backtest/sweep', { method: 'POST', body: JSON.stringify(sweepBody) }))) }
    catch (e) { setError(errorText(e)) } finally { setBusy(false); setJob(null) }
  };
  const sweepCombos = SWEEP_GRID.top_ns.length * SWEEP_GRID.holding.length * SWEEP_GRID.nearHigh.length;
  const utilization = (strategy === 'momentum' || strategy === 'committee') && Number(topN) > 0 && Number(capital) > 0 ? (Number(topN) * Number(perTrade)) / Number(capital) : null;
  const m = result?.metrics;
  const p = (result?.params || {}) as Record<string, unknown>;
  const paramText = result ? (
    result.strategy === 'pead' ? `每只 ${String(p.earnings_limit)} 份财报` :
    result.strategy === 'momentum' ? `回看 ${String(p.lookback_days)} 日 · 跳过 ${String(p.skip_days)} 日 · 每期 ${String(p.top_n)} 只${p.near_high_pct != null ? ` · 距 52 周高点 ≤ ${Math.round(Number(p.near_high_pct) * 100)}%` : ''}` :
    result.strategy === 'insider' ? `${String(p.window_days)} 日内 ≥ ${String(p.min_insiders)} 人 · ≥ $${Number(p.min_value_usd || 0).toLocaleString()}` :
    `每期前 ${String(p.top_n)} 只 · 共识 ≥ ${String(p.min_consensus)} · 一致度 ≥ ${String(p.min_agreement)} · ${p.lean ? '省流' : '全量'}`
  ) : '';
  const failures = Object.keys(result?.notes?.price_failures || {}).length; const errs = Object.keys(result?.notes?.errors || {}).length;
  const dep = result?.deployment || null; const offFull = !!dep && Math.abs(dep.utilization - 1) >= 0.01;
  return <>
    {handoff && <HandoffBanner handoff={handoff} onUse={() => { setUniverse('custom'); setTickers(handoff.tickers.join(', ')); clearHandoff() }} onClear={clearHandoff}/>}
    <div className="lab-notice"><strong>数据不足，结果不理想</strong><span>四个策略都在现有数据上验证过：财报事件在标普 500 里没有可用的漂移；等权动量在标普 500 里 10 年跑不赢 SPY（18 组参数全部落后，52 周新高过滤是负贡献）；委员会和内部人策略受财务与交易历史深度限制，样本太小。根本原因是数据：yfinance 只有价格，退市和被收购的前成分股取不到（10 年回测 695 只里缺 101 只）；Financial Datasets 的财报只有约 5 个季度、财务报表只有 2–3 个财年，做不了更长的事件研究和基本面回测。想要有参考价值的结果，需要接入更多数据：含退市股票的无幸存者偏差价格库（如 Sharadar、Norgate、CRSP、Polygon），10 年以上的时点化财务与财报历史，以及更完整的内部人交易记录。在此之前，这一页的数字只适合检验流程，不适合据以决策。</span></div>
    <div className="lab-tool lab-tool-fill">
      <section className="surface lab-config lab-config-fill"><div className="surface-header"><div><h2>回测参数</h2><span>先选免费还是付费数据，再选该档位下的策略；四个策略共用一个引擎</span></div></div>
        <div className="lab-config-body lab-scroll">
        <Field label="数据" hint={tier === 'free' ? '只用 yfinance 的日线价格，不产生任何费用；可用的策略是纯价格策略。' : '财报、内部人交易和财务数据来自 Financial Datasets，按请求计费（每次约 $0.02）；下方会给出费用预估。'}><Chips options={[{ id: 'free', label: '免费（yfinance）' }, { id: 'paid', label: '付费（Financial Datasets）' }]} value={tier} onChange={setTier}/></Field>
        <Field label="策略" hint={meta.hint}><Chips options={tierStrategies.map(s => ({ id: s.id, label: s.label }))} value={strategy} onChange={setStrategy}/></Field>
        <UniversePicker universe={universe} setUniverse={setUniverse} tickers={tickers} setTickers={setTickers} exclude={strategy === 'momentum' ? [] : INDEX_UNIVERSES} limit={strategy === 'momentum' ? 600 : 60}/>
        {strategy === 'momentum' && INDEX_UNIVERSES.includes(universe) && <p className="lab-note">整个指数作为股票池：每只一次 yfinance 请求，标普 500 约 5–10 分钟，在后台运行可看进度。「每期买入只数」建议 10–20，让排名真正起作用。</p>}
        {tier === 'paid' && <Field label="价格来源" hint={dataSource === 'yfinance' ? '日线价格仍用 yfinance，免费；只有事件和财务数据付费。' : `日线价格也从 Financial Datasets 取，每只每 90 天一段、每段 ${usd(price('prices'))}；只在需要与线上口径完全一致时用。`}><Chips options={[{ id: 'yfinance', label: 'yfinance（免费）' }, { id: 'fd', label: 'Financial Datasets（付费）' }]} value={dataSource} onChange={setDataSource}/></Field>}
        <div className="lab-grid2">
          <Field label="持有交易日" hint={strategy === 'momentum' || strategy === 'committee' ? '也是换仓周期' : undefined}><NumberInput value={holding} onChange={setHolding} min={1} max={252}/></Field>
          {strategy === 'pead' ? <Field label="每只回看财报数" hint="每份财报是一个入场事件"><NumberInput value={earnings} onChange={setEarnings} min={1} max={20}/></Field>
            : <Field label="回看历史（天）" hint="在这段历史里产生信号"><NumberInput value={history} onChange={setHistory} min={60} max={3650} step={30}/></Field>}
          {strategy === 'momentum' && <><Field label="动量回看（交易日）"><NumberInput value={lookback} onChange={setLookback} min={20} max={504}/></Field><Field label="跳过最近（交易日）" hint="避开短期反转"><NumberInput value={skip} onChange={setSkip} min={0} max={120}/></Field><Field label="每期买入只数"><NumberInput value={topN} onChange={setTopN} min={1} max={60}/></Field><Field label="距 52 周高点 ≤ %" hint="留空 = 不限制；填 5 即只买离新高 5% 以内的"><NumberInput value={nearHigh} onChange={setNearHigh} min={0} max={100}/></Field></>}
          {strategy === 'insider' && <><Field label="窗口（天）"><NumberInput value={window} onChange={setWindowDays} min={1} max={180}/></Field><Field label="最少内部人数"><NumberInput value={minInsiders} onChange={setMinInsiders} min={1} max={20}/></Field><Field label="合计买入 ≥ $"><NumberInput value={minValue} onChange={setMinValue} min={0} step={50000}/></Field></>}
          {strategy === 'committee' && <><Field label="每期买入只数"><NumberInput value={topN} onChange={setTopN} min={1} max={60}/></Field><Field label="最低共识" hint="-1 到 1"><NumberInput value={minConsensus} onChange={setMinConsensus} min={-1} max={1} step={0.1}/></Field><Field label="最低一致度" hint="多数派占投票人的比例"><NumberInput value={minAgreement} onChange={setMinAgreement} min={0} max={1} step={0.1}/></Field><Field label="财报滞后（天）" hint="只用信号日之前这么多天已公布的财务数据"><NumberInput value={lag} onChange={setLag} min={0} max={120}/></Field></>}
          <Field label="初始资金（$）"><NumberInput value={capital} onChange={setCapital} min={1000} step={10000}/></Field>
          <Field label="单笔资金（$）" hint={utilization == null ? undefined : `每期 ${topN} 只 × $${Number(perTrade).toLocaleString()} = $${(Number(topN) * Number(perTrade)).toLocaleString()}，资金使用率 ${Math.round(utilization * 100)}%${Math.abs(utilization - 1) < 0.01 ? '（满仓）' : utilization > 1 ? '，超过初始资金 = 隐含杠杆，收益和回撤都会放大' : '，其余闲置，收益会被稀释；满仓请填 $' + Math.round(Number(capital) / Number(topN)).toLocaleString()}`}><NumberInput value={perTrade} onChange={setPerTrade} min={100} step={1000}/></Field>
          <Field label="交易成本（基点 / 单边）" hint="10 = 每边 0.1%，买卖各扣一次"><NumberInput value={costBps} onChange={setCostBps} min={0} max={200}/></Field>
        </div>
        {strategy === 'committee' && <Field label="省流模式" hint={lean ? '跳过新闻和内部人交易两路请求；只影响几位投资人的情绪小分项' : '取全部数据，每个时点每只约 6 次请求'}><Chips options={[{ id: 'on', label: '开（省流）' }, { id: 'off', label: '关（全量）' }]} value={lean ? 'on' : 'off'} onChange={v => setLean(v === 'on')}/></Field>}
        </div>
        <div className="lab-config-foot">
        {tier === 'paid' && <p className="lab-note">预计 Financial Datasets 费用：{estimate ? `≈ ${usd(estimate.total)}${estimate.note ? `（${estimate.note}${estimate.prices ? ` + 价格 ${usd(estimate.prices)}` : ''}）` : ''}` : '$0.00'}{strategy === 'committee' ? '。已缓存的时点不重复计费；委员会回测在后台运行，可以看进度。' : strategy === 'pead' && dataSource === 'fd' ? '，另加每笔交易 1 次价格请求。' : '。'}</p>}
        {tier === 'free' && <p className="lab-note">免费档不调用 Financial Datasets，费用 $0.00。</p>}
        <button className="run-button" disabled={busy || (universe === 'custom' && !parseTickers(tickers).length)} onClick={() => void run()}>{busy ? (job ? `回测中… ${job.done} / ${job.total}` : '回测中…') : '运行回测'}</button>
        {strategy === 'momentum' && <><button type="button" className="run-button lab-secondary" disabled={busy || (universe === 'custom' && !parseTickers(tickers).length)} onClick={() => void runSweep()}>{busy ? '运行中…' : `参数扫描（${sweepCombos} 组）`}</button>
          <p className="lab-note">扫描固定网格：每期 {SWEEP_GRID.top_ns.join(' / ')} 只 × 持有 {SWEEP_GRID.holding.join(' / ')} 日 × 是否要求距 52 周高点 ≤ 10%；沿用上面的股票池、回看历史、动量回看、跳过、初始资金和成本；每组都满仓（单笔 = 初始资金 ÷ 每期只数），价格只取一次。</p></>}
        {job && <div className="lab-progress"><i style={{ width: `${job.total ? Math.round((job.done / job.total) * 100) : 0}%` }}/></div>}
        </div>
      </section>
      <section className="surface lab-result lab-result-clamp">
        {error ? <ErrorBox text={error}/> : sweep ? <SweepView result={sweep} ask={ask}/> : !result ? <Empty glyph="↗" title="等待回测" text={meta.empty}/> : <>
          <div className="surface-header"><div><h2>{STRATEGY_LABEL[result.strategy] || result.strategy.toUpperCase()} · {result.tickers.length} 只 · {m?.n_trades ?? 0} 笔{m?.n_periods ? ` · ${m.n_periods} 期` : ''}</h2><span>{UNIVERSE_LABEL[result.universe] || result.universe}{result.universe_as_of ? `（成分股 ${result.universe_as_of}）` : ''} · 持有 {String(p.holding_days)} 日 · {paramText} · 单笔 ${Number(p.per_trade || 0).toLocaleString()} · 成本 {String(p.cost_bps ?? 0)} bp/边 · 价格 {result.data_source === 'fd' ? 'Financial Datasets' : 'yfinance'} · FD 费用 {usd(result.fd_cost_usd ?? 0)}</span></div></div>
          <div className="lab-scroll">
          {m ? <div className="lab-stats"><Stat label="总收益" value={pct(m.total_return_pct)} tone={m.total_return_pct >= 0 ? 'positive' : 'negative'}/><Stat label="年化" value={pct(m.annualized_return_pct)}/><Stat label="夏普（按期）" value={`${num(m.sharpe_ratio)}${m.sharpe_trade_level ? ` / 逐笔 ${num(m.sharpe_trade_level)}` : ''}`}/><Stat label="最大回撤" value={pct(m.max_drawdown_pct)} tone="negative"/><Stat label="胜率" value={pctAbs(m.win_rate)}/><Stat label="平均单笔" value={pct(m.avg_return_pct)}/><Stat label="多 / 空" value={`${m.n_long} / ${m.n_short}`}/><Stat label="平均持有" value={`${num(m.avg_holding_days, 1)} 日`}/><Stat label={`${result.benchmark?.ticker || 'SPY'} 同期`} value={pct(result.benchmark?.total_return_pct)}/><Stat label="超额收益" value={pct(result.excess_return_pct)} tone={(result.excess_return_pct ?? 0) >= 0 ? 'positive' : 'negative'}/></div> : <p className="lab-note">没有产生交易：{result.strategy === 'pead' ? '股票池里可能没有可用的财报事件。' : result.strategy === 'insider' ? '这段历史里没有满足条件的内部人集中买入。' : result.strategy === 'committee' ? '没有股票达到共识与一致度门槛，或财务数据不足；下方「每期投票」列出了每个换仓日的情况。' : '没有股票满足动量条件，或价格历史不够长。'}</p>}
          {m && dep && <div className={`lab-deploy ${offFull ? 'off' : ''}`}><div className="lab-subhead">资金使用率 {Math.round(dep.utilization * 100)}%<small>每期最多 {dep.positions_per_period} 只 × ${Number(dep.per_trade).toLocaleString()} = ${Number(dep.deployed_usd).toLocaleString()}，初始资金 ${Number(dep.capital).toLocaleString()}{offFull ? (dep.utilization > 1 ? '。超过初始资金，上面的收益和回撤含隐含杠杆；下面按实际投入的资金重算。' : '。其余资金闲置，上面的收益被稀释、与满仓的 SPY 不可比；下面按实际投入的资金重算。') : '，满仓。'}</small></div>
            {offFull && <div className="lab-stats"><Stat label="总收益（按投入）" value={pct(dep.on_deployed.total_return_pct)} tone={dep.on_deployed.total_return_pct >= 0 ? 'positive' : 'negative'}/><Stat label="年化（按投入）" value={pct(dep.on_deployed.annualized_return_pct)}/><Stat label="最大回撤（按投入）" value={pct(dep.on_deployed.max_drawdown_pct)} tone="negative"/><Stat label="超额（按投入）" value={pct(dep.on_deployed.excess_return_pct)} tone={(dep.on_deployed.excess_return_pct ?? 0) >= 0 ? 'positive' : 'negative'}/></div>}</div>}
          {result.strategy === 'momentum' && INDEX_UNIVERSES.includes(result.universe as Universe) && (result.notes?.membership?.point_in_time
            ? (result.notes.membership.mode === 'additions'
              ? <p className="lab-note">已按加入日期过滤：信号日之后才纳入指数的股票不参与该期排名（消除了主要的前视偏差）。变动表尚未获取，已被剔除的股票无法还原，仍有少量幸存者偏差。</p>
              : <p className="lab-note">已按历史成分股回测：每个换仓日只在当时的成分股里排名（变动记录 {result.notes.membership.changes} 条，可回溯到 {result.notes.membership.history_from}；含已退出指数的股票共 {result.notes.membership.tickers_incl_former} 只，其中 {result.notes?.no_data?.length || 0} 只已无价格数据被跳过）。</p>)
            : <div className="lab-error"><strong>幸存者偏差</strong><span>股票池是当前成分股快照，没有历史变动记录：回测在早期就「知道」哪些股票后来会被纳入指数，收益会被高估。在 VPS 上运行 <code>python -m v2.screening.universes --refresh</code> 获取标普 500 的变动表后重跑。</span></div>)}
          {result.notes?.aborted && <div className="lab-error"><strong>提前停止</strong><span>{result.notes.aborted.signal_date} 这期 {result.notes.aborted.of} 只里有 {result.notes.aborted.failed} 只取不到核心数据（{result.notes.aborted.reason}），后面 {result.notes.aborted.remaining_dates.length} 个换仓日没有跑。常见原因是 Financial Datasets 余额用完；充值后重跑只会补取失败的时点。</span></div>}
          {(failures > 0 || errs > 0) && <p className="lab-note">{failures > 0 ? `${failures} 只取不到价格已跳过` : ''}{failures > 0 && errs > 0 ? '；' : ''}{errs > 0 ? `${errs} 个（股票, 时点）取数失败` : ''}，详见原始结果。</p>}
          <EquityLine values={result.equity_curve || []}/>
          {(result.yearly?.length || 0) > 0 && <div className="lab-table-wrap"><div className="lab-subhead">按年拆分（按入场年份归类；{result.benchmark?.ticker || 'SPY'} 取同一时间段）</div><table className="lab-table"><thead><tr><th>年份</th><th>时间段</th><th>期数</th><th>笔数</th><th>胜率</th><th>年初权益</th><th>盈亏</th><th>策略收益</th>{offFull && <th>按投入</th>}<th>{result.benchmark?.ticker || 'SPY'}</th><th>超额</th>{offFull && <th>超额（按投入）</th>}</tr></thead><tbody>{result.yearly!.map(y => <tr key={y.year}><td><strong>{y.year}</strong></td><td>{y.start.slice(5)} → {y.end}</td><td>{y.periods}</td><td>{y.trades}</td><td>{pctAbs(y.win_rate)}</td><td>${Math.round(y.start_equity).toLocaleString()}</td><td className={y.pnl >= 0 ? 'positive' : 'negative'}>${Math.round(y.pnl).toLocaleString()}</td><td className={(y.return_pct ?? 0) >= 0 ? 'positive' : 'negative'}>{pct(y.return_pct)}</td>{offFull && <td className={(y.return_on_deployed_pct ?? 0) >= 0 ? 'positive' : 'negative'}>{pct(y.return_on_deployed_pct)}</td>}<td>{pct(y.benchmark_pct)}</td><td className={(y.excess_pct ?? 0) >= 0 ? 'positive' : 'negative'}><strong>{pct(y.excess_pct)}</strong></td>{offFull && <td className={(y.excess_on_deployed_pct ?? 0) >= 0 ? 'positive' : 'negative'}><strong>{pct(y.excess_on_deployed_pct)}</strong></td>}</tr>)}</tbody></table><p className="lab-note">策略收益 = 当年盈亏 ÷ 年初权益（每笔固定 ${Number(p.per_trade || 0).toLocaleString()}）{offFull ? `；「按投入」= 当年盈亏 ÷ 实际投入的 $${Number(dep!.deployed_usd).toLocaleString()}，这才是和满仓 SPY 的公平比较` : ''}；12 月入场的仓位在次年 1 月平仓，仍计入前一年。</p></div>}
          {result.strategy === 'committee' && (result.notes?.periods?.length || 0) > 0 && <div className="lab-table-wrap"><div className="lab-subhead">每期投票（{result.notes!.periods!.length} 个换仓日）</div><table className="lab-table"><thead><tr><th>换仓日</th><th>数据截至</th><th>股票</th><th>投票 / 弃权</th><th>多 / 空 / 中</th><th>共识</th><th>一致度</th><th>入选</th><th>弃权原因</th></tr></thead><tbody>{result.notes!.periods!.flatMap(pd => [...pd.verdicts.map(v => <tr key={`${pd.signal_date}-${v.ticker}`} className={v.picked ? 'picked' : ''}><td>{pd.signal_date}</td><td>{pd.as_of}</td><td><strong>{v.ticker}</strong></td><td>{v.voters} / {v.abstained}{v.gaps ? ` · ${v.gaps} 处缺数据` : ''}</td><td>{v.bullish} / {v.bearish} / {v.neutral}</td><td className={v.consensus >= 0 ? 'positive' : 'negative'}>{num(v.consensus)}</td><td>{pctAbs(v.agreement)}</td><td>{v.picked ? '✓' : v.voters === 0 ? '全部弃权' : '—'}</td><td className="lab-reasons">{(v.reasons || []).join('；')}</td></tr>), ...pd.missing.map(t => <tr key={`${pd.signal_date}-${t}-x`}><td>{pd.signal_date}</td><td>{pd.as_of}</td><td><strong>{t}</strong></td><td colSpan={6}>取数失败</td></tr>)])}</tbody></table></div>}
          {result.trades.length > 0 && <div className="lab-table-wrap"><div className="lab-subhead">成交明细</div><table className="lab-table"><thead><tr><th>股票</th><th>方向</th><th>入场</th><th>出场</th><th>入场价</th><th>出场价</th><th>收益</th><th>盈亏</th><th>信号</th></tr></thead><tbody>{result.trades.slice(0, 60).map((t, i) => <tr key={i}><td><strong>{t.ticker}</strong></td><td>{t.direction === 'long' ? '多' : '空'}</td><td>{t.entry_date}</td><td>{t.exit_date}</td><td>${num(t.entry_price)}</td><td>${num(t.exit_price)}</td><td className={t.return_pct >= 0 ? 'positive' : 'negative'}>{pct(t.return_pct)}</td><td className={t.pnl >= 0 ? 'positive' : 'negative'}>${t.pnl.toFixed(0)}</td><td>{signalText(result.strategy, t.metadata)}</td></tr>)}</tbody></table>{result.trades.length > 60 && <p className="lab-note">只显示前 60 笔，共 {result.trades.length} 笔；完整列表在原始结果里。</p>}</div>}
          </div>
          <div className="lab-foot"><button type="button" className="explain-button" onClick={() => ask(`${STRATEGY_LABEL[result.strategy] || result.strategy} 回测结果：${result.tickers.length > 12 ? `${result.tickers.length} 只（${UNIVERSE_LABEL[result.universe] || result.universe}）` : result.tickers.join(', ')}，${m?.n_trades ?? 0} 笔，总收益 ${pct(m?.total_return_pct)}，同期 SPY ${pct(result.benchmark?.total_return_pct)}，超额 ${pct(result.excess_return_pct)}，夏普 ${num(m?.sharpe_ratio)}，最大回撤 ${pct(m?.max_drawdown_pct)}，胜率 ${pctAbs(m?.win_rate)}（${paramText}）。请评价这组指标的稳健性和样本量问题。`, '实验室 · 策略回测')}>问 AI 评价稳健性</button><RawJson data={result}/></div>
        </>}
      </section>
    </div>
  </>;
}

// ----------------------------------------------------------------------- event study

function EventStudyTool({ result, setResult, handoff, clearHandoff, ask, restore }: ToolProps & { result?: EventStudyResult; setResult: (r?: EventStudyResult) => void; handoff: Handoff | null; clearHandoff: () => void; restore?: Restore }) {
  const [universe, setUniverse] = useState<Universe>('custom'); const [tickers, setTickers] = useState('AAPL, MSFT, NVDA');
  const [earnings, setEarnings] = useState('8'); const [boot, setBoot] = useState('2000'); const [surprise, setSurprise] = useState(true); const [dataSource, setDataSource] = useState<'yfinance' | 'fd'>('yfinance'); const [dedupe, setDedupe] = useState(true); const [groupBy, setGroupBy] = useState<'surprise' | 'reaction' | 'source'>('surprise');
  const [busy, setBusy] = useState(false); const [error, setError] = useState('');
  useRestore(restore, useCallback((p: Record<string, unknown>) => {
    if (typeof p.universe === 'string') setUniverse(p.universe as Universe);
    if (Array.isArray(p.tickers) && (p.tickers as string[]).length) setTickers((p.tickers as string[]).join(', '));
    if (p.data_source === 'fd' || p.data_source === 'yfinance') setDataSource(p.data_source);
    setEarnings(str(p.earnings_limit, '8')); setBoot(str(p.n_bootstrap, '2000'));
    if (typeof p.require_eps_surprise === 'boolean') setSurprise(p.require_eps_surprise);
    if (typeof p.dedupe === 'boolean') setDedupe(p.dedupe);
    if (p.group_by === 'surprise' || p.group_by === 'reaction' || p.group_by === 'source') setGroupBy(p.group_by);
  }, []));
  const info = useLabData<{ items: Record<string, UniverseInfo> }>('/api/lab/universes');
  const pricing = useLabData<Pricing>('/api/lab/committee/pricing');
  const price = (k: string) => pricing.data?.prices_usd[k] ?? 0.02;
  const poolSize = Math.min(20, universe === 'custom' ? parseTickers(tickers).length : (info.data?.items[universe]?.size ?? 0));
  const estEarnings = poolSize * price('earnings');
  const estPrices = dataSource === 'fd' ? (poolSize * 5 + 11) * price('prices') : 0;  // ~400 days per ticker + SPY since 2023, in 90-day chunks
  const run = async () => { setBusy(true); setError(''); try { setResult(await apiJson<EventStudyResult>('/api/lab/event-study', { method: 'POST', body: JSON.stringify({ universe, tickers: universe === 'custom' ? parseTickers(tickers) : [], data_source: dataSource, earnings_limit: Number(earnings), n_bootstrap: Number(boot), require_eps_surprise: surprise, dedupe, group_by: groupBy }) })) } catch (e) { setError(errorText(e)) } finally { setBusy(false) } };
  return <>
    {handoff && <HandoffBanner handoff={handoff} onUse={() => { setUniverse('custom'); setTickers(handoff.tickers.join(', ')); clearHandoff() }} onClear={clearHandoff}/>}
    <div className="lab-notice"><strong>数据不足，结果不理想</strong><span>三轮实验都没有发现财报后的漂移：标普 500 大盘股、从中筛出的 20 只最小市值股票、以及按公告日反应分组，[+2, +20] 窗口的累计异常收益都不显著。限制是硬的：Financial Datasets 只有约 5 个季度的财报历史，每只股票最多 5 个事件，20 只股票凑不出 100 个事件，而标普 500 成分股的财报又是市场上被消化得最快的。想要有机会看到漂移，需要 10 年以上的财报日期与 EPS 意外历史，并把股票池扩到中小市值。在此之前，这一页只适合检验事件研究的流程，不适合据以决策。</span></div>
    <div className="lab-tool lab-tool-fill">
      <section className="surface lab-config lab-config-fill"><div className="surface-header"><div><h2>事件研究参数</h2><span>事件 = 财报公布；基准 = 市场模型（SPY）</span></div></div>
        <div className="lab-config-body lab-scroll">
        <UniversePicker universe={universe} setUniverse={setUniverse} tickers={tickers} setTickers={setTickers} exclude={INDEX_UNIVERSES}/>
        <div className="lab-grid2"><Field label="每只回看财报数"><NumberInput value={earnings} onChange={setEarnings} min={1} max={20}/></Field><Field label="Bootstrap 次数"><NumberInput value={boot} onChange={setBoot} min={100} max={10000} step={500}/></Field></div>
        <Field label="同一季度只算一次" hint={dedupe ? '同一次财报的 8-K 公告和几天后的 10-Q / 10-K 视为一个事件，取最早的公告日。' : '每份申报都算一个事件；8-K 和 10-Q / 10-K 会重复计入同一次财报。'}><Chips options={[{ id: 'yes', label: '是（去重）' }, { id: 'no', label: '否' }]} value={dedupe ? 'yes' : 'no'} onChange={v => setDedupe(v === 'yes')}/></Field>
        <Field label="分组" hint={groupBy === 'surprise' ? '全部事件一组，再按 EPS 超预期 / 不及预期 / 符合预期分组，这是 PEAD 关心的对比。' : groupBy === 'reaction' ? '不看 EPS 标签，按公告后两天的异常收益把事件分成强 / 中 / 弱三档，再看 [+2,+20] 窗口有没有延续。市场自己的反应比 EPS 标签可靠。' : '按申报类型（8-K、10-Q、10-K）分组，用来检查公告日和报表日的反应差异。'}><Chips options={[{ id: 'surprise', label: 'BEAT / MISS' }, { id: 'reaction', label: '公告日反应' }, { id: 'source', label: '申报类型' }]} value={groupBy} onChange={setGroupBy}/></Field>
        <Field label="只统计有 EPS surprise 标注的事件"><Chips options={[{ id: 'yes', label: '是' }, { id: 'no', label: '否' }]} value={surprise ? 'yes' : 'no'} onChange={v => setSurprise(v === 'yes')}/></Field>
        <Field label="价格来源" hint={dataSource === 'yfinance' ? '股票和 SPY 的日线价格来自 yfinance，免费且已复权；财报历史始终来自 Financial Datasets。' : `日线价格也从 Financial Datasets 取（未复权），每只每 90 天一段、每段 ${usd(price('prices'))}。`}><Chips options={[{ id: 'yfinance', label: 'yfinance（免费）' }, { id: 'fd', label: 'Financial Datasets（付费）' }]} value={dataSource} onChange={setDataSource}/></Field>
        </div>
        <div className="lab-config-foot">
        <p className="lab-note">预计 Financial Datasets 费用：≈ {usd(estEarnings + estPrices)}（财报历史 {usd(estEarnings)}{estPrices ? ` + 价格 ${usd(estPrices)}` : ''}）。股票池最多取前 20 只。</p>
        <button className="run-button" disabled={busy || (universe === 'custom' && !parseTickers(tickers).length)} onClick={() => void run()}>{busy ? '计算中…' : '开始计算'}</button>
        </div>
      </section>
      <section className="surface lab-result lab-result-clamp">
        {error ? <ErrorBox text={error}/> : !result ? <Empty glyph="∿" title="等待计算" text="对每次财报公布，用市场模型剔除大盘影响后累计 [0,+1]、[0,+5]、[0,+20] 日的异常收益，并给出 t 检验和 bootstrap 置信区间。"/> : <>
          <div className="surface-header"><div><h2>{result.events.length} 个事件 · {result.aggregates.length} 组</h2><span>{UNIVERSE_LABEL[result.universe] || result.universe} · {result.tickers.join(' ')}{result.skipped_tickers.length ? ` · 跳过 ${result.skipped_tickers.join(' ')}` : ''} · {result.dedupe ? '每季度一个事件' : '未去重'} · {result.group_by === 'source' ? '按申报类型分组' : result.group_by === 'reaction' ? '按公告日反应分组' : '按 BEAT / MISS 分组'} · 价格 {result.data_source === 'fd' ? 'Financial Datasets' : 'yfinance'} · FD 费用 {usd(result.fd_cost_usd ?? 0)}</span></div></div>
          <div className="lab-scroll">
          {Object.keys(result.price_failures || {}).length > 0 && <p className="lab-note">{Object.keys(result.price_failures!).length} 只取不到价格：{Object.keys(result.price_failures!).join(' ')}。</p>}
          {result.aggregates.map(g => <div key={g.group || g.source_type} className="lab-table-wrap"><div className="lab-subhead">{GROUP_LABEL[g.group || g.source_type] || g.group || g.source_type} · {g.n_events} 个事件</div><table className="lab-table"><thead><tr><th>窗口</th><th>n</th><th>平均 CAR</th><th>标准差</th><th>t</th><th>p</th><th>95% CI</th></tr></thead><tbody>{g.windows.map(w => <tr key={w.window}><td><strong>{w.window}</strong></td><td>{w.n_events}</td><td className={w.mean_car >= 0 ? 'positive' : 'negative'}>{pct(w.mean_car, 2)}</td><td>{pctAbs(w.std_car, 2)}</td><td>{num(w.t_stat)}</td><td className={w.p_value < 0.05 ? 'positive' : ''}>{num(w.p_value, 3)}</td><td>[{pct(w.ci.lower, 2)}, {pct(w.ci.upper, 2)}]</td></tr>)}</tbody></table></div>)}
          {result.events.length > 0 && <div className="lab-table-wrap"><div className="lab-subhead">逐事件</div><table className="lab-table"><thead><tr><th>股票</th><th>日期</th><th>类型</th><th>EPS</th><th>CAR [0,1]</th><th>CAR [0,5]</th><th>CAR [0,20]</th><th>漂移 [+2,+20]</th><th>β</th><th>R²</th></tr></thead><tbody>{result.events.slice(0, 60).map((e, i) => <tr key={i} className={e.market_model.r_squared < 0.05 ? 'weak' : ''} title={e.market_model.r_squared < 0.05 ? '市场模型拟合很差（R² < 0.05），β 不可信' : undefined}><td><strong>{e.ticker}</strong></td><td>{e.event_date}</td><td>{e.source_type}</td><td className={e.eps_surprise === 'BEAT' ? 'positive' : e.eps_surprise === 'MISS' ? 'negative' : ''}><strong>{e.eps_surprise || '—'}</strong></td><td className={(e.car_0_1 || 0) >= 0 ? 'positive' : 'negative'}>{pct(e.car_0_1, 2)}</td><td className={(e.car_0_5 || 0) >= 0 ? 'positive' : 'negative'}>{pct(e.car_0_5, 2)}</td><td className={(e.car_0_20 || 0) >= 0 ? 'positive' : 'negative'}>{pct(e.car_0_20, 2)}</td><td className={(e.car_2_20 || 0) >= 0 ? 'positive' : 'negative'}>{pct(e.car_2_20, 2)}</td><td>{num(e.market_model.beta)}</td><td>{num(e.market_model.r_squared)}</td></tr>)}</tbody></table></div>}
          </div>
          <div className="lab-foot"><button type="button" className="explain-button" onClick={() => ask(`事件研究：${result.tickers.join(', ')}，${result.events.length} 个财报事件。${result.aggregates.map(g => `${GROUP_LABEL[g.group || g.source_type] || g.source_type}: ${g.windows.map(w => `${w.window} 平均CAR ${pct(w.mean_car, 2)} (p=${num(w.p_value, 3)})`).join('，')}`).join('；')}。请解释这些窗口的统计显著性意味着什么。`, '实验室 · 事件研究')}>问 AI 解释显著性</button><RawJson data={result}/></div>
        </>}
      </section>
    </div>
  </>;
}

// ------------------------------------------------------------------------ scoreboard

type ScoreSort = 'n' | 'edge_1m' | 'edge_3m';
const SCORE_SORT_LABEL: Record<ScoreSort, string> = { n: '票数', edge_1m: '1 月超出基准', edge_3m: '3 月超出基准' };
const WEAK_N = 20;
function ScoreboardTool({ ask }: ToolProps) {
  const board = useLabData<Scoreboard>('/api/lab/committee/scoreboard');
  const thresholds = useLabData<{ monitoring: Record<string, number>; intraday: Record<string, number> }>('/api/lab/signals');
  const [busy, setBusy] = useState(false); const [report, setReport] = useState<{ checked: number; filled: number; skipped_no_price: number; errors: Record<string, string> } | null>(null); const [error, setError] = useState('');
  const [sort, setSort] = useState<ScoreSort>('n');
  const backfill = async () => { setBusy(true); setError(''); try { setReport(await apiJson('/api/lab/committee/backfill', { method: 'POST', body: JSON.stringify({}) })); board.reload() } catch (e) { setError(errorText(e)) } finally { setBusy(false) } };
  const c = board.data?.counts; const b = board.data?.baseline;
  const rows = [...(board.data?.items || [])].sort((x, y) => sort === 'n' ? y.n - x.n || (y.hit_rate ?? 0) - (x.hit_rate ?? 0) : ((y[sort] ?? -9) - (x[sort] ?? -9)));
  const strong = rows.filter(r => r.n >= WEAK_N);
  return <div className="lab-tool">
    <section className="surface lab-config"><div className="surface-header"><div><h2>观察进度</h2><span>投票时只有当天数据，收益是之后才发生的</span></div></div>
      {c ? <div className="lab-stats one"><Stat label="委员会运行" value={String(c.runs)}/><Stat label="覆盖股票" value={String(c.tickers)}/><Stat label="有效票" value={String(c.votes)}/><Stat label="已评 1 月 / 待评" value={`${c.scored_1m} / ${c.due_1m}`}/><Stat label="已评 3 月 / 待评" value={`${c.scored_3m} / ${c.due_3m}`}/></div> : board.error ? <ErrorBox text={board.error}/> : null}
      <button className="run-button" disabled={busy} onClick={() => void backfill()}>{busy ? '回填中…' : '立即回填到期的票'}</button>
      <p className="lab-note">调度器每天 02:30 ET 自动回填一次；这里只是手动触发同一段代码。</p>
      {report && <p className="lab-note">本次检查 {report.checked}，回填 {report.filled}，缺价格 {report.skipped_no_price}{Object.keys(report.errors).length ? `，错误 ${Object.keys(report.errors).length}` : ''}。</p>}
      {error && <ErrorBox text={error}/>}
      {b && (b.n_1m > 0 || b.n_3m > 0) && <div className="lab-thresholds"><div className="lab-subhead">基准：被评分的股票自己走了多少</div>
        <div><span>1 个月上涨比例（= 永远看多的命中率）</span><strong>{pctAbs(b.up_rate_1m)}（{b.n_1m} 个股票·日）</strong></div>
        <div><span>1 个月平均收益</span><strong>{pct(b.avg_return_1m, 2)}</strong></div>
        <div><span>3 个月上涨比例</span><strong>{pctAbs(b.up_rate_3m)}（{b.n_3m} 个）</strong></div>
        <div><span>3 个月平均收益</span><strong>{pct(b.avg_return_3m, 2)}</strong></div>
        <div><span>永远看空的命中率</span><strong>{b.up_rate_1m == null ? '—' : pctAbs(1 - b.up_rate_1m)}</strong></div></div>}
      <p className="lab-note">每位投资人的「基准」= 按他自己看多 / 看空的比例，在同一批股票上随机投票能拿到的命中率；「超出」= 命中率 − 基准。只有超出为正、且票数够多（≥ {WEAK_N}）的行才说明点什么。</p>
      {thresholds.data && <div className="lab-thresholds"><div className="lab-subhead">生产监控阈值（只读）</div>{Object.entries({ ...thresholds.data.intraday, volume_spike_threshold: thresholds.data.monitoring.volume_spike_threshold, insider_buy_min_value: thresholds.data.monitoring.insider_buy_min_value }).map(([k, v]) => <div key={k}><span>{k.replaceAll('_', ' ')}</span><strong>{typeof v === 'number' && v < 1 ? pctAbs(v, 1) : typeof v === 'number' && v >= 10000 ? money(v) : String(v)}</strong></div>)}</div>}
    </section>
    <section className="surface lab-result">
      {!board.data ? null : board.data.items.length === 0 ? <Empty glyph="◎" title="还没有可评分的票" text={c && c.votes ? `已有 ${c.votes} 票在观察中，最早的一批在投票 30 天后进入评分。` : '先在「投资人委员会」跑几次评审，票会在 30 天和 91 天后被真实收益评分。'}/> : <>
        <div className="surface-header"><div><h2>逐人命中率</h2><span>命中 = 看多且上涨，或看空且下跌；中性和弃权不计。票数少于 {WEAK_N} 的行灰显，区间是 1 月命中率的 95% Wilson 区间</span></div><div className="lab-chips">{(Object.keys(SCORE_SORT_LABEL) as ScoreSort[]).map(k => <button key={k} type="button" className={k === sort ? 'active' : ''} onClick={() => setSort(k)}>按{SCORE_SORT_LABEL[k]}</button>)}</div></div>
        <div className="lab-table-wrap"><table className="lab-table"><thead><tr><th>投资人</th><th>票（1 月）</th><th>命中率</th><th>95% 区间</th><th>基准</th><th>超出</th><th>方向收益</th><th>票（3 月）</th><th>命中率 3 月</th><th>超出 3 月</th><th>方向收益 3 月</th><th>中性 / 弃权</th></tr></thead><tbody>{rows.map(r => <tr key={r.persona} className={r.n < WEAK_N ? 'weak' : ''}><td><strong>{r.name_zh || r.persona}</strong></td><td>{r.n}</td><td>{pctAbs(r.hit_rate)}</td><td>{r.ci_low == null ? '—' : `${pctAbs(r.ci_low)} – ${pctAbs(r.ci_high)}`}</td><td>{pctAbs(r.baseline_1m)}</td><td className={(r.edge_1m ?? 0) >= 0 ? 'positive' : 'negative'}><strong>{r.edge_1m == null ? '—' : `${r.edge_1m >= 0 ? '+' : ''}${(r.edge_1m * 100).toFixed(0)} 点`}</strong></td><td className={(r.avg_directional_1m || 0) >= 0 ? 'positive' : 'negative'}>{pct(r.avg_directional_1m, 2)}</td><td>{r.n_3m ?? 0}</td><td>{pctAbs(r.hit_rate_3m)}</td><td className={(r.edge_3m ?? 0) >= 0 ? 'positive' : 'negative'}>{r.edge_3m == null ? '—' : `${r.edge_3m >= 0 ? '+' : ''}${(r.edge_3m * 100).toFixed(0)} 点`}</td><td className={(r.avg_directional_3m || 0) >= 0 ? 'positive' : 'negative'}>{pct(r.avg_directional_3m, 2)}</td><td>{r.neutral ?? 0} / {r.abstained ?? 0}</td></tr>)}</tbody></table></div>
        <p className="lab-note">{strong.length ? `${strong.length} 位投资人票数达到 ${WEAK_N}，其中 ${strong.filter(r => (r.edge_1m ?? 0) > 0).length} 位 1 月命中率高于自己的基准。` : `还没有投资人的票数达到 ${WEAK_N}，目前的差异主要是噪声。`}</p>
        <div className="lab-foot"><button type="button" className="explain-button" onClick={() => ask(`投资人 1 个月命中率榜（基准 = 同批股票上涨比例 ${pctAbs(b?.up_rate_1m)}）：${rows.map(r => `${r.name_zh || r.persona} ${pctAbs(r.hit_rate)}，基准 ${pctAbs(r.baseline_1m)}，超出 ${r.edge_1m == null ? '—' : (r.edge_1m * 100).toFixed(0) + ' 点'}（${r.n} 票，95% 区间 ${pctAbs(r.ci_low)}–${pctAbs(r.ci_high)}）`).join('；')}。哪些差异是真实的，哪些只是样本量小？`, '实验室 · 观察记分板')}>问 AI 怎么解读</button></div>
      </>}
    </section>
  </div>;
}

// ------------------------------------------------------------------------------ runs

function RunsTool({ onOpen }: ToolProps & { onOpen: (kind: string, result: unknown, params: Record<string, unknown>) => void }) {
  const [kind, setKind] = useState<string>('all');
  const runs = useLabData<{ items: RunSummary[]; counts: Record<string, number> }>(`/api/lab/runs?limit=100${kind === 'all' ? '' : `&kind=${kind}`}`, [kind]);
  const [error, setError] = useState(''); const [notice, setNotice] = useState(''); const [busy, setBusy] = useState(false);
  const open = async (run: RunSummary) => { setError(''); try { const row = await apiJson<{ kind: string; params: Record<string, unknown>; result: unknown }>(`/api/lab/runs/${encodeURIComponent(run.id)}`); onOpen(row.kind, row.result, row.params || {}) } catch (e) { setError(errorText(e)) } };
  const remove = async (run: RunSummary) => {
    if (!globalThis.confirm(`删除这条${KIND_LABEL[run.kind] || run.kind}记录？结果会一并删除，不可恢复。`)) return;
    setError(''); try { await apiJson(`/api/lab/runs/${encodeURIComponent(run.id)}`, { method: 'DELETE' }); runs.reload() } catch (e) { setError(errorText(e)) }
  };
  const cleanup = async () => {
    if (!globalThis.confirm('删除 30 天前的全部运行记录？不可恢复。')) return;
    setBusy(true); setError(''); setNotice('');
    try { const r = await apiJson<{ deleted: number }>('/api/lab/runs/cleanup', { method: 'POST', body: JSON.stringify({ older_than_days: 30 }) }); setNotice(`已删除 ${r.deleted} 条 30 天前的记录。`); runs.reload() } catch (e) { setError(errorText(e)) } finally { setBusy(false) }
  };
  const kinds = ['all', ...Object.keys(runs.data?.counts || {})];
  return <section className="surface">
    <div className="surface-header"><div><h2>运行记录</h2><span>后端重启不丢；点一条在对应工具里原样重开，参数也一起回填</span></div><div className="lab-chips">{kinds.map(k => <button key={k} type="button" className={k === kind ? 'active' : ''} onClick={() => setKind(k)}>{k === 'all' ? `全部 ${Object.values(runs.data?.counts || {}).reduce((a, b) => a + b, 0)}` : `${KIND_LABEL[k] || k} ${runs.data?.counts[k] ?? ''}`}</button>)}<button type="button" className="lab-chip-danger" disabled={busy} onClick={() => void cleanup()}>{busy ? '清理中…' : '清理 30 天前的记录'}</button></div></div>
    {error && <ErrorBox text={error}/>}{runs.error && <ErrorBox text={runs.error}/>}{notice && <p className="lab-note">{notice}</p>}
    {!runs.data?.items.length ? <Empty glyph="◷" title="还没有运行记录" text="每个工具跑完都会记在这里。"/> : <div className="lab-runs">{runs.data.items.map(r => <RunRow key={r.id} run={r} onOpen={r.kind === 'backfill' ? undefined : () => void open(r)} onDelete={() => void remove(r)}/>)}</div>}
  </section>;
}
