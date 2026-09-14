'use client';

/* eslint-disable @next/next/no-assign-module-variable, @typescript-eslint/no-unused-vars */

import { FormEvent, type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import Image from 'next/image';
import { apiJson, askAgentV2, authGuest, authLogin, authLogout, authStatus, getAgentV2Job, setApiAccessRole, type AccessStatus, type PageContext, type PageSelection, type AgentV2Evidence, type AgentV2Job, type AgentV2Response, type AgentV2SubAgent, type AgentV2TraceStep } from './lib/api';
import { LabPage, labMenu, type LabTool } from './lab';
import { RelationshipMap } from './relationship-map';
import { MoneyflowPanel, type FlowAnalysis } from './moneyflow-panel';
import { CostPage, type CostReport } from './cost-page';
import { ResearchHelp } from './research-help';

type MainSection = 'core' | 'research' | 'lab' | 'cost';
type ResearchTool = 'stock' | 'fundamentals' | 'valuation' | 'earnings' | 'expectations' | 'institutional' | 'moneyflow' | 'macro' | 'chain' | 'risk';
type ChatMode = 'agent_v2' | 'agent_v3';
const CHAT_LABELS: Record<ChatMode, string> = { agent_v2: 'Agent V2', agent_v3: 'Agent V3' };
type AgentChatMeta = { status: string; route: string; answerMode: string; elapsedMs: number; verified: boolean; capabilities: string[]; webRequested: boolean; webEnabled: boolean; webAllowed: boolean; warnings: string[]; synthesis: string; synthesisFallback: boolean; rewritten: string; subAgents: AgentV2SubAgent[] };
const STOP_LABELS: Record<string, string> = { finished: '完成', rounds: '轮次用尽', time: '超时', no_model: '无模型', no_budget: '无预算' };
const CALL_LABELS: Record<string, string> = { news: '新闻', filing_events: '读申报', memory: '记忆', search: '搜索', read: '读正文', filings: '申报', sections_read: '读节', events: '事件', objections: '反对' };
function callSummary(calls: Record<string, number | null | undefined>): string { return Object.entries(calls).filter(([, value]) => value).map(([key, value]) => `${CALL_LABELS[key] || key} ${value}`).join('、'); }
function SubAgentTrace({ label, rounds, elapsedMs, stop, calls, trace, intraday, notes }: { label: string; rounds: number; elapsedMs: number; stop: string; calls: Record<string, number | null | undefined>; trace: AgentV2TraceStep[]; intraday?: boolean; notes?: string[] }) {
  const summary = [label, `${rounds} 轮`, `${(elapsedMs / 1000).toFixed(1)}s`, callSummary(calls), STOP_LABELS[stop] || stop, intraday ? '盘中' : ''].filter(Boolean).join(' · ');
  return <details className="agent-trace"><summary>{summary}</summary>{notes && notes.length > 0 && <ul className="agent-notes">{notes.map((note, index) => <li key={index}>{note}</li>)}</ul>}<ol>{trace.map((step, index) => <li key={`${step.round}-${index}`}><span>第 {step.round} 轮</span><code>{step.action}</code>{step.detail && <em>{step.detail}</em>}<small>{step.ms} ms</small></li>)}{trace.length === 0 && <li><em>没有记录到模型轮次</em></li>}</ol></details>;
}
const SYNTHESIS_LABELS: Record<string, string> = { clean: '模型回答', repaired: '模型回答（修复一轮）', fallback: '兜底摘要（模型草稿未通过校验）', knowledge: '知识回答', deterministic: '规则摘要' };
function synthesisWarnings(synthesis: AgentV2Response['synthesis']): string[] {
  if (!synthesis) return [];
  return synthesis.attempts.filter(attempt => !attempt.ok).flatMap(attempt => {
    const stage = attempt.stage === 'draft' ? '草稿' : attempt.stage === 'repair' ? '修复稿' : '模型调用';
    return [...(attempt.warnings || []), ...(attempt.unknown_citations || []).map(id => `未知证据引用：${id}`), ...(attempt.ungrounded_numbers || []).map(value => `未落地数字：${value}`)].map(item => `${stage}被拒：${item}`);
  });
}
type ChatMessage = { id: number; role: 'user' | 'assistant'; text: string; image?: string; meta?: string; mode?: ChatMode; agent?: AgentChatMeta; evidence?: AgentV2Evidence[] };
type Position = { symbol: string; current_price: number; market_value: number; unrealized_pl: number; unrealized_pl_pct: number };
type PortfolioResponse = { account: { cash: number; portfolio_value: number; paper: boolean }; positions: Position[]; pnl: { intraday_pl_pct: number; portfolio_value: number; cash: number }; history?: { timestamp: number[]; equity: number[] } };
type RiskResponse = { pnl: { daily_pnl_pct: number | null }; concentration: { top_1_pct: number; top_3_pct: number }; exposure: { largest_sector: string; largest_sector_pct: number }; drawdown: { current_drawdown_pct: number | null }; earnings_risk: { ticker: string; days_until: number }[] };
type TickerItem = { key: string; label: string; value: number | null; change_pct: number | null; unit: string };
type ActivityItem = { id: number; ts: string; agent: string; msg_type: string; tickers: string | null; preview: string; title: string | null; priority_tier: string | null; importance_score: number | null };
type ActivityResponse = { items: ActivityItem[]; warning?: string };
type MonitoringUniverse = { intraday: string[]; source: string; scan_interval_seconds: number; price_pct_threshold: number; volume_pace_threshold: number };
type WatchlistItem = { ticker: string; added_at: string; note: string };
type PriceAlertItem = { id: number; ticker: string; direction: 'above' | 'below'; target_price: number; created_at: string; fired_at: string | null; fired_price: number | null };
type PriceRange = '1W' | '1M' | '3M' | '6M' | '1Y' | '3Y';
type PriceInterval = '30m' | '1h' | '1d' | '1w';
type PriceChartView = 'line' | 'candles';
type MarketSession = 'PRE_MARKET' | 'REGULAR' | 'AFTER_HOURS' | 'CLOSED';
type BarStatus = 'FORMING' | 'FINAL' | 'STALE';
type ChartLayerKey = 'sma20' | 'sma50' | 'sma200' | 'currentPrice' | 'sr1' | 'sr2' | 'volume' | 'volumeMA20' | 'atr14' | 'volumeRatio' | 'ohlc' | 'levelLabels' | 'grid';
type ChartDisplaySettings = Record<ChartLayerKey, boolean>;
type ChartPreset = 'simple' | 'standard' | 'professional';
type ActiveChartPreset = ChartPreset | 'custom';
type ChartPreference = { preset: ActiveChartPreset; layers: ChartDisplaySettings };
type BarIndicators = { timeframe: PriceInterval; sma20: number | null; sma50: number | null; sma200: number | null; atr14: number | null; volumeMA20: number | null; volumeRatio: number | null };
type PricePoint = { symbol: string; timestamp: string; open: number; high: number; low: number; close: number; volume: number; barReturn: number | null; timeframe: PriceInterval; session: MarketSession; source: string; isDelayed: boolean; adjustmentMode: string; isFinal: boolean; status: BarStatus; isStale: boolean; quoteSynchronized?: boolean; finalizationSource?: string; indicators: BarIndicators };
type PriceQuote = { symbol: string; timestamp: string; price: number; regularClose: number | null; previousClose: number | null; dailyChangePct: number | null; dayReturn: number | null; adjustedDayReturn: number | null; extendedHoursChangePct: number | null; source: string; session: MarketSession; isDelayed: boolean; isFinal: boolean; regularCloseIsOfficial: boolean; adjustmentMode: string; returnAdjustmentMode: string };
type PriceZone = { id: string; role: 'support' | 'resistance'; originalRole: 'high' | 'low'; lower: number; upper: number; midpoint: number; score: number; recencyScore: number; touchCount: number; lastTouchedAt: string; reversalMagnitudeAtr: number; volumeConfirmation: number; timeframe: PriceInterval; distancePct: number; roleReversal: boolean; major: boolean };
type TechnicalAnalysis = { symbol: string; timestamp: string; source: string; currentPrice: number; regularClose: number | null; previousRegularClose: number | null; dailyChangePct: number | null; dayReturn: number | null; adjustedDayReturn: number | null; barReturn: number | null; timeframe: PriceInterval; marketSession: MarketSession; isDelayed: boolean; adjustmentMode: string; SMA20: number | null; SMA50: number | null; SMA200: number | null; ATR14: number | null; volume: number; volumeMA20: number | null; volumeRatio: number | null; supports: PriceZone[]; resistances: PriceZone[]; trend: 'BULLISH' | 'BEARISH' | 'MIXED'; breakoutStatus: string; latestBarIsFinal: boolean };
type PriceHistoryResponse = { symbol: string; range: PriceRange; timeframe: PriceInterval; periodDays: number; visibleStart: string; visibleEnd: string; warmupBars: number; source: string; adjustmentMode: string; includesExtendedHours: boolean; quote: PriceQuote; bars: PricePoint[]; technicalAnalysis: TechnicalAnalysis };
type ChartSnapshotRefreshStatus = { started?: boolean; status: 'idle' | 'running' | 'completed' | 'completed_with_errors'; total: number; published: number; failed: number };
type PositionChartState = { symbol: string; portfolioPrice: number; range: PriceRange; points: PricePoint[]; quote: PriceQuote | null; analysis: TechnicalAnalysis | null; source: string; adjustmentMode: string; warmupBars: number; includesExtendedHours: boolean; loading: boolean; error: string };
type ChainLabel = { seed: string; category: 'supplier' | 'customer' | 'smaller_peer' | 'beneficiary'; reason: string };
type ChainNeighbor = { ticker: string; labels: ChainLabel[]; exists: boolean; already_in_universe: boolean; relation_verified: boolean; relation_checked: boolean };
type ChainData = { date: string; seeds: string[]; neighbors: ChainNeighbor[]; llm_tokens: number; api_calls: number; tavily_calls: number };
type ToolResult = { text: string; image?: string; intent?: string; data?: ChainData | null };
type ResearchResultState = { tool: ResearchTool; ticker: string; result: ToolResult; engine?: StockResearchResult | null; progress?: Record<string,string> };
type ResearchRun = (tool: ResearchTool, ticker: string, retry?: { runId: string; modules: string[] }) => Promise<void>;
type EngineFinding = { title?: string; name?: string; description?: string; reason?: string; level?: string; confidence?: number; source_ids?: string[]; evidence?: string[] };
type ProviderError = { provider: string; type: string; message: string; retryable: boolean };
type EngineModule = { status: string; ticker: string; module: string; summary: string; score: number | null; metrics: Record<string, unknown>; findings: EngineFinding[]; risks: EngineFinding[]; sources: string[]; confidence: number; completeness?: number; missing_fields?: string[]; warnings?: string[]; errors?: string[]; provider_errors?: ProviderError[]; data_sources_used?: string[]; cache_hit?: boolean; cache_expires_at?: string | null; details: Record<string, unknown>; error?: string | null };
type ResearchDriver = { id: string; driver_type: string; title: string; direction: string; importance: string; confidence: number; findings: string[]; evidence: string[] };
type ResearchInvalidation = { driver_id: string; driver: string; monitor: string; condition: string; finding_ids: string[]; evidence_ids: string[] };
type DebateItem = { claim: string; driver_id?: string | null; finding_id?: string; finding_ids?: string[]; evidence_ids?: string[]; type?: string };
type StockResearchResult = { run_id: string; status: string; ticker: string; from_cache: boolean; generated_at: string; dataset: { errors: Record<string,string>; available: Record<string,number | boolean> }; module_status: Record<string,string>; modules: Record<string,EngineModule>; scores: Record<string,number | null>; risk_level: string; investment_thesis: string; core_thesis?: string; why_now?: string; bull_case: string; base_case: string; bear_case: string; key_catalysts: Record<string,unknown>[]; key_risks: EngineFinding[]; thesis_invalidation: (string | ResearchInvalidation)[]; key_drivers?: { positive: ResearchDriver[]; negative: ResearchDriver[] }; research_confidence?: { score: number; band: string; modules: Record<string,{score:number;band:string}> }; research_support_tier?: string; quality_gate?: { status: string; coverage: number; warnings: string[] }; conflicts?: { id:string; type:string; description:string; severity:string }[]; industry_context?: { profile:string; description:string }; industry_metrics?: { profile:string; available_count:number; required_count:number; completeness:number; required_coverage?:number; optional_available_count?:number; optional_count?:number; optional_coverage?:number; available_metrics:string[]; required_metrics:string[] }; main_tension?: { statement:string }; investment_debate?: { what_bulls_believe:DebateItem[]; what_bears_believe:DebateItem[]; what_matters_most:DebateItem[] }; confidence_contributors?: string[]; confidence_limitations?: string[]; sources: { id: string; provider: string; type: string; title: string; url?: string | null; published_at?: string | null; fetched_at: string }[] };
type ResearchJob = { job_id: string; ticker: string; status: string; module_status: Record<string,string>; result: StockResearchResult | null; error?: string | null };
type ChatApiResponse = { html?: string; chart_b64?: string; intent?: string; data?: ChainData | null; job_id?: string; status?: 'running' | 'completed' | 'failed'; error?: string; deduplicated?: boolean };

const demoPortfolio: PortfolioResponse = { account: { cash: 6640, portfolio_value: 100348, paper: true }, pnl: { intraday_pl_pct: .0094, portfolio_value: 100348, cash: 6640 }, positions: [
  { symbol: 'NVDA', current_price: 143.46, market_value: 18463, unrealized_pl: 2675, unrealized_pl_pct: .1694 }, { symbol: 'IVV', current_price: 770.03, market_value: 17721, unrealized_pl: 2180, unrealized_pl_pct: .0289 }, { symbol: 'MU', current_price: 1001.24, market_value: 9554, unrealized_pl: 4, unrealized_pl_pct: .0005 }, { symbol: 'LRCX', current_price: 151.6, market_value: 7455, unrealized_pl: -237, unrealized_pl_pct: -.031 },
] };
const demoRisk: RiskResponse = { pnl: { daily_pnl_pct: .0094 }, concentration: { top_1_pct: .184, top_3_pct: .457 }, exposure: { largest_sector: 'Technology', largest_sector_pct: .421 }, drawdown: { current_drawdown_pct: -.025 }, earnings_risk: [{ ticker: 'NVDA', days_until: 12 }, { ticker: 'CRM', days_until: 6 }] };
const demoTape: TickerItem[] = [{ key: 'sp500', label: '标普500', value: 6835.12, change_pct: .0109, unit: '' }, { key: 'nasdaq100', label: '纳指100', value: 25182.7, change_pct: .0127, unit: '' }, { key: 'vix', label: 'VIX', value: 18.4, change_pct: -.021, unit: '' }, { key: 'us10y', label: '美债10Y', value: 4.12, change_pct: null, unit: '%' }];
const TAPE_STORAGE_KEY = 'workbench:ticker-tape-selection';
const CHART_DISPLAY_STORAGE_KEY = 'workbench:kline-display-v1';
const DEFAULT_TAPE_KEYS = ['sp500', 'nasdaq100', 'dow', 'russell2000', 'semiconductor_etf', 'vix', 'gold', 'wti', 'dollar', 'bitcoin', 'us10y'];
const PRICE_RANGES: { key: PriceRange; label: string; interval: PriceInterval; intervalLabel: string }[] = [{ key: '1W', label: '1周', interval: '30m', intervalLabel: '30分钟线' }, { key: '1M', label: '1月', interval: '1h', intervalLabel: '1小时线' }, { key: '3M', label: '3月', interval: '1d', intervalLabel: '日线' }, { key: '6M', label: '6月', interval: '1d', intervalLabel: '日线' }, { key: '1Y', label: '1年', interval: '1d', intervalLabel: '日线' }, { key: '3Y', label: '3年', interval: '1w', intervalLabel: '周线' }];
const CHART_PRESETS: Record<ChartPreset, ChartDisplaySettings> = {
  simple: { sma20: false, sma50: false, sma200: false, currentPrice: true, sr1: false, sr2: false, volume: true, volumeMA20: false, atr14: false, volumeRatio: false, ohlc: false, levelLabels: false, grid: true },
  standard: { sma20: true, sma50: true, sma200: true, currentPrice: true, sr1: true, sr2: false, volume: true, volumeMA20: false, atr14: false, volumeRatio: false, ohlc: true, levelLabels: true, grid: true },
  professional: { sma20: true, sma50: true, sma200: true, currentPrice: true, sr1: true, sr2: true, volume: true, volumeMA20: true, atr14: true, volumeRatio: true, ohlc: true, levelLabels: true, grid: true },
};
const CHART_PRESET_META: Record<ChartPreset, { label: string; description: string }> = {
  simple: { label: '简洁', description: '只看价格、当前价与成交量' },
  standard: { label: '标准', description: '均线 + 最近支撑阻力，适合大多数用户' },
  professional: { label: '专业', description: '完整指标与多级关键价位' },
};
const RESEARCH_TOOL_MODULES: Record<ResearchTool, string[] | undefined> = { stock: undefined, fundamentals: ['fundamental'], valuation: ['valuation'], earnings: ['earnings','sec','expectations'], expectations: ['expectations','catalyst'], institutional: ['institutional'], moneyflow: ['fund_flow'], macro: ['macro'], chain: ['supply_chain'], risk: ['risk'] };
const CHART_LAYER_GROUPS: { title: string; items: { key: ChartLayerKey; label: string }[] }[] = [
  { title: '均线', items: [{ key: 'sma20', label: 'SMA20' }, { key: 'sma50', label: 'SMA50' }, { key: 'sma200', label: 'SMA200' }] },
  { title: '关键价位', items: [{ key: 'currentPrice', label: '当前价格线' }, { key: 'sr1', label: 'S1 / R1' }, { key: 'sr2', label: 'S2 / R2' }] },
  { title: '成交量', items: [{ key: 'volume', label: 'Volume' }, { key: 'volumeMA20', label: 'Volume MA20' }] },
  { title: '辅助指标', items: [{ key: 'atr14', label: 'ATR14' }, { key: 'volumeRatio', label: '量比' }] },
  { title: '图表元素', items: [{ key: 'ohlc', label: '顶部 OHLC' }, { key: 'levelLabels', label: '价位标签' }, { key: 'grid', label: '网格线' }] },
];
const researchMenu: { id: ResearchTool; label: string; icon: string }[] = [{ id: 'stock', label: '个股工作台', icon: '⌁' }, { id: 'fundamentals', label: '公司与基本面', icon: '▤' }, { id: 'valuation', label: '估值与同业', icon: '◇' }, { id: 'earnings', label: '财报与 SEC', icon: '▧' }, { id: 'expectations', label: '预期与催化剂', icon: '◷' }, { id: 'institutional', label: '机构与内部人', icon: '▦' }, { id: 'moneyflow', label: '资金流', icon: '≈' }, { id: 'macro', label: '宏观环境', icon: '◎' }, { id: 'chain', label: '产业链', icon: '⌘' }, { id: 'risk', label: '风险雷达', icon: '⚑' }];

function formatMoney(value: number) { return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(value) }
function formatUnitPrice(value: number | null | undefined) { return typeof value === 'number' && Number.isFinite(value) && value > 0 ? new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(value) : '—' }
function formatPct(value: number | null | undefined) { return value == null ? '—' : `${value >= 0 ? '+' : ''}${(value * 100).toFixed(2)}%` }
function formatTickerValue(value: number | null | undefined) { return typeof value === 'number' && Number.isFinite(value) ? value.toLocaleString() : '—' }
function formatClock(value: Date) { const pad = (part: number) => String(part).padStart(2, '0'); return `${value.getFullYear()}-${pad(value.getMonth() + 1)}-${pad(value.getDate())} ${pad(value.getHours())}:${pad(value.getMinutes())}:${pad(value.getSeconds())}` }
function formatClockInZone(value: Date, timeZone: string) { const parts = new Intl.DateTimeFormat('en-CA', { timeZone, year:'numeric', month:'2-digit', day:'2-digit', hour:'2-digit', minute:'2-digit', second:'2-digit', hour12:false }).formatToParts(value); const get = (type:string) => parts.find(part => part.type === type)?.value || '--'; return `${get('year')}-${get('month')}-${get('day')} ${get('hour')}:${get('minute')}:${get('second')}` }
function formatUsd(value: number) { return `$${value.toFixed(value < 0.01 ? 4 : 2)}` }
function formatCatalystTimestamp(value: unknown) { const raw = String(value || '').trim(); if (!raw) return { date:'日期待定', time:'' }; const parsed = new Date(raw); if (Number.isNaN(parsed.getTime())) return { date:raw.slice(0,10), time:raw.length > 10 ? raw.slice(11,19) : '' }; const date = new Intl.DateTimeFormat('en-CA', { timeZone:'America/New_York', year:'numeric', month:'2-digit', day:'2-digit' }).format(parsed); const hasTime = /T\d{2}:\d{2}|\s\d{2}:\d{2}/.test(raw); const time = hasTime ? `${new Intl.DateTimeFormat('en-GB', { timeZone:'America/New_York', hour:'2-digit', minute:'2-digit', second:'2-digit', hour12:false }).format(parsed)} ET` : ''; return { date, time } }
function formatPricePointTime(value: string | undefined, interval: PriceInterval) { if (!value) return '—'; if (interval === '1d' || interval === '1w') return value.slice(0, 10); try { return new Intl.DateTimeFormat('zh-CN', { timeZone: 'America/New_York', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false }).format(new Date(value)).replace('/', '-') + ' ET' } catch { return value.slice(5, 16).replace('T', ' ') } }
function sessionLabel(session: MarketSession) { return ({ PRE_MARKET: '盘前', REGULAR: '盘中', AFTER_HOURS: '盘后', CLOSED: '已收盘' } as const)[session] }
function quoteChangePct(quote: PriceQuote) { return quote.session === 'PRE_MARKET' || quote.session === 'AFTER_HOURS' ? quote.extendedHoursChangePct : quote.dailyChangePct }
function adjustmentLabel(mode: string) { return mode === 'SPLIT_ADJUSTED' ? '拆股复权（价格口径）' : mode === 'TOTAL_RETURN_ADJUSTED_CLOSE' ? '含现金分红收益口径' : mode }
function barStatusLabel(bar: PricePoint) { return bar.status === 'STALE' || bar.isStale ? '数据待刷新' : bar.isFinal ? '已完成' : '形成中' }
function currentPricePrefix(session: MarketSession) { return session === 'AFTER_HOURS' ? 'AH' : session === 'PRE_MARKET' ? 'PM' : session === 'REGULAR' ? '现价' : '收盘' }
function formatActivityTimeET(value: string) { try { return `${new Intl.DateTimeFormat('en-CA', { timeZone: 'America/New_York', hour: '2-digit', minute: '2-digit', hour12: false }).format(new Date(value))} ET` } catch { return value.slice(11, 16) } }
function formatActivityMonthDayET(value: string) { try { const parts = new Intl.DateTimeFormat('en-US', { timeZone: 'America/New_York', month: '2-digit', day: '2-digit' }).formatToParts(new Date(value)); const month = parts.find(part => part.type === 'month')?.value || '--'; const day = parts.find(part => part.type === 'day')?.value || '--'; return `${month}-${day}` } catch { return value.slice(5, 10) } }
function htmlToText(html: string) { if (typeof window === 'undefined') return html.replace(/<[^>]+>/g, ' '); const doc = new DOMParser().parseFromString(html.replace(/<br\s*\/?>/gi, '\n'), 'text/html'); return (doc.body.textContent || '').replace(/\n\s+/g, '\n').trim() }
async function waitForChatJob(jobId: string): Promise<ChatApiResponse> { const deadline = Date.now() + 190_000; while (Date.now() < deadline) { await new Promise<void>(resolve => window.setTimeout(resolve, 1500)); const job = await apiJson<ChatApiResponse>(`/api/chat/jobs/${encodeURIComponent(jobId)}`); if (job.status === 'completed') return job; if (job.status === 'failed') throw new Error(job.error || '产业链研究失败') } throw new Error('产业链研究等待超过 190 秒') }
function agentSessionId(version: 'v2' | 'v3') { const key = `workbench:agent-${version}-session`; const stored = sessionStorage.getItem(key); if (stored) return stored; const id = `workbench-${globalThis.crypto?.randomUUID?.() || Date.now().toString(36)}`; sessionStorage.setItem(key, id); return id }
function waitWithSignal(ms: number, signal: AbortSignal) { return new Promise<void>((resolve, reject) => { const timer = window.setTimeout(resolve, ms); signal.addEventListener('abort', () => { window.clearTimeout(timer); reject(new DOMException('Request cancelled', 'AbortError')) }, { once: true }) }) }
async function waitForAgentJob(initial: AgentV2Job, signal: AbortSignal, onProgress: (job: AgentV2Job) => void, version: 'v2' | 'v3'): Promise<AgentV2Response> { let job = initial; const deadline = Date.now() + 15 * 60_000; while (Date.now() < deadline) { if (job.status === 'completed' && job.result) return job.result; if (job.status === 'failed') throw new Error(job.error || 'Agent 后台任务失败'); onProgress(job); await waitWithSignal(1000, signal); job = await getAgentV2Job(job.job_id, signal, version) } throw new Error('Agent 后台任务超过 15 分钟仍未完成') }
function uniqueEvidence(items: AgentV2Evidence[]) { const seen = new Set<string>(); return items.filter(item => { const key = item.id || `${item.source_url}:${item.claim}`; if (seen.has(key)) return false; seen.add(key); return true }) }

function AgentAnswer({ text, evidence = [], messageId }: { text: string; evidence?: AgentV2Evidence[]; messageId: number }) {
  const evidenceById = new Map(evidence.map((item, index) => [item.id, { item, number: index + 1 }]));
  const detailsId = `agent-evidence-${messageId}`;
  const revealEvidence = (number: number) => {
    const details = document.getElementById(detailsId) as HTMLDetailsElement | null;
    if (details) details.open = true;
    window.requestAnimationFrame(() => document.getElementById(`${detailsId}-${number}`)?.scrollIntoView({ block: 'nearest', behavior: 'smooth' }));
  };
  const inline = (value: string, keyPrefix: string): ReactNode[] => {
    const normalized = value.replace(/\\([*_])/g, '$1');
    const tokens = normalized.split(/(\*\*[^*]+\*\*|`[^`\n]+`|\[[A-Za-z0-9_.:-]+\])/g).filter(Boolean);
    return tokens.map((token, index) => {
      const key = `${keyPrefix}-${index}`;
      if (token.startsWith('**') && token.endsWith('**')) return <strong key={key}>{inline(token.slice(2, -2), `${key}-strong`)}</strong>;
      if (token.startsWith('`') && token.endsWith('`')) return <code key={key}>{token.slice(1, -1)}</code>;
      const citation = token.match(/^\[([A-Za-z0-9_.:-]+)\]$/)?.[1];
      const linked = citation ? evidenceById.get(citation) : undefined;
      if (linked) return <button key={key} type="button" className="evidence-ref" title={linked.item.claim} aria-label={`查看证据 ${linked.number}`} onClick={() => revealEvidence(linked.number)}>[{linked.number}]</button>;
      return token;
    });
  };
  const lines = text.replace(/\r\n/g, '\n').trim().split('\n');
  const blocks: ReactNode[] = [];
  const isRule = (line: string) => /^\s*(?:-{3,}|\*{3,}|_{3,})\s*$/.test(line);
  const isList = (line: string) => /^\s*(?:[-*+] |\d+\. )/.test(line);
  const isHeading = (line: string) => /^\s*#{1,6}\s+/.test(line);
  const isTable = (line: string) => /^\s*\|.*\|\s*$/.test(line);
  for (let index = 0; index < lines.length;) {
    const line = lines[index];
    if (!line.trim()) { index += 1; continue }
    if (isRule(line)) { blocks.push(<hr key={`rule-${index}`}/>); index += 1; continue }
    const heading = line.match(/^\s*(#{1,6})\s+(.+?)\s*#*\s*$/);
    if (heading) {
      const content = inline(heading[2], `heading-${index}`);
      blocks.push(heading[1].length <= 2 ? <h3 key={`heading-${index}`}>{content}</h3> : <h4 key={`heading-${index}`}>{content}</h4>);
      index += 1;
      continue;
    }
    if (isTable(line) && index + 1 < lines.length && /^\s*\|[\s:|-]+\|\s*$/.test(lines[index + 1])) {
      const rows: string[][] = [];
      while (index < lines.length && isTable(lines[index])) {
        if (!/^\s*\|[\s:|-]+\|\s*$/.test(lines[index])) rows.push(lines[index].trim().slice(1, -1).split('|').map(cell => cell.trim()));
        index += 1;
      }
      const [header, ...body] = rows;
      blocks.push(<div className="answer-table-wrap" key={`table-${index}`}><table><thead><tr>{header.map((cell, cellIndex) => <th key={cellIndex}>{inline(cell, `th-${index}-${cellIndex}`)}</th>)}</tr></thead><tbody>{body.map((row, rowIndex) => <tr key={rowIndex}>{row.map((cell, cellIndex) => <td key={cellIndex}>{inline(cell, `td-${index}-${rowIndex}-${cellIndex}`)}</td>)}</tr>)}</tbody></table></div>);
      continue;
    }
    if (isList(line)) {
      const ordered = /^\s*\d+\. /.test(line);
      const items: string[] = [];
      while (index < lines.length && isList(lines[index]) && /^\s*\d+\. /.test(lines[index]) === ordered) {
        items.push(lines[index].replace(/^\s*(?:[-*+] |\d+\. )/, ''));
        index += 1;
      }
      const children = items.map((item, itemIndex) => <li key={itemIndex}>{inline(item, `li-${index}-${itemIndex}`)}</li>);
      blocks.push(ordered ? <ol key={`list-${index}`}>{children}</ol> : <ul key={`list-${index}`}>{children}</ul>);
      continue;
    }
    const paragraph: string[] = [];
    while (index < lines.length && lines[index].trim() && !isRule(lines[index]) && !isHeading(lines[index]) && !isList(lines[index]) && !isTable(lines[index])) {
      paragraph.push(lines[index].trim());
      index += 1;
    }
    if (paragraph.length) blocks.push(<p key={`paragraph-${index}`}>{inline(paragraph.join(' '), `paragraph-${index}`)}</p>);
    else index += 1;
  }
  return <div className="message-markdown">{blocks}</div>;
}

function MiniEquityChart({ values }: { values: number[] }) {
  const source = values.length > 1 ? values : [97420, 98620, 98210, 99100, 100348];
  const min = Math.min(...source); const max = Math.max(...source); const span = Math.max(max - min, 1);
  const points = source.map((value, index) => `${(index / (source.length - 1)) * 760},${155 - ((value - min) / span) * 135}`).join(' ');
  const area = `0,170 ${points} 760,170`;
  return <svg className="equity-chart" viewBox="0 0 760 170" role="img" aria-label="组合净值曲线"><defs><linearGradient id="equityFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="currentColor" stopOpacity=".22"/><stop offset="1" stopColor="currentColor" stopOpacity=".02"/></linearGradient></defs><path className="chart-grid" d="M0 35H760M0 85H760M0 135H760"/><polygon className="chart-area" points={area}/><polyline className="chart-line" points={points}/></svg>;
}

function StockPriceChart({ symbol, rangeLabel, interval, points }: { symbol: string; rangeLabel: string; interval: PriceInterval; points: PricePoint[] }) {
  const closes = points.map(point => point.close).filter(value => Number.isFinite(value));
  if (closes.length < 2) return <div className="position-chart-empty">暂时没有可显示的历史价格</div>;
  const min = Math.min(...closes); const max = Math.max(...closes); const span = Math.max(max - min, 0.01);
  const change = closes[0] ? (closes.at(-1)! - closes[0]) / closes[0] : 0;
  const chartPoints = closes.map((value, index) => `${18 + (index / (closes.length - 1)) * 724},${235 - ((value - min) / span) * 190}`).join(' ');
  const area = `18,250 ${chartPoints} 742,250`;
  return <div className={`stock-price-chart ${change >= 0 ? 'chart-positive' : 'chart-negative'}`}><svg viewBox="0 0 760 265" role="img" aria-label={`${symbol} ${rangeLabel}价格波动曲线`}><defs><linearGradient id="stockPriceFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="currentColor" stopOpacity=".2"/><stop offset="1" stopColor="currentColor" stopOpacity=".02"/></linearGradient></defs><path className="stock-chart-grid" d="M18 45H742M18 140H742M18 235H742"/><polygon className="stock-chart-area" points={area}/><polyline className="stock-chart-line" points={chartPoints}/></svg><div className="stock-chart-range"><span>{formatPricePointTime(points[0]?.timestamp, interval)}</span><strong>{formatUnitPrice(closes.at(-1))}<em className={change >= 0 ? 'positive' : 'negative'}>{formatPct(change)}</em></strong><span>{formatPricePointTime(points.at(-1)?.timestamp, interval)}</span></div><div className="stock-chart-bounds"><span>区间低点 {formatUnitPrice(min)}</span><span>区间高点 {formatUnitPrice(max)}</span></div></div>;
}

function CandlestickChart({ symbol, rangeLabel, interval, points, analysis, quote, layers, activePreset, showExtendedHours, onLayerChange, onPreset }: { symbol: string; rangeLabel: string; interval: PriceInterval; points: PricePoint[]; analysis: TechnicalAnalysis; quote: PriceQuote; layers: ChartDisplaySettings; activePreset: ActiveChartPreset; showExtendedHours: boolean; onLayerChange: (key: ChartLayerKey, enabled: boolean) => void; onPreset: (preset: ChartPreset) => void }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const layerButtonRef = useRef<HTMLButtonElement>(null);
  const layerMenuRef = useRef<HTMLElement>(null);
  const [hoveredBar, setHoveredBar] = useState<PricePoint | null>(null);
  const [hoveredZone, setHoveredZone] = useState<PriceZone | null>(null);
  const [layerMenuOpen, setLayerMenuOpen] = useState(false);
  const bars = useMemo(() => points.filter(point => [point.open, point.high, point.low, point.close, point.volume].every(Number.isFinite)), [points]);
  useEffect(() => {
    if (!layerMenuOpen) return;
    const closeOnOutsidePointer = (event: PointerEvent) => {
      const target = event.target as Node;
      if (!layerMenuRef.current?.contains(target) && !layerButtonRef.current?.contains(target)) setLayerMenuOpen(false);
    };
    document.addEventListener('pointerdown', closeOnOutsidePointer);
    return () => document.removeEventListener('pointerdown', closeOnOutsidePointer);
  }, [layerMenuOpen]);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || bars.length < 2) return;
    const context = canvas.getContext('2d');
    if (!context) return;
    let visibleCount = bars.length;
    let startIndex = 0;
    let hoverIndex: number | null = null;
    let hoverY = 0;
    let dragging = false;
    let dragStartX = 0;
    let dragStartIndex = 0;
    const movingAverages = [{ key: 'sma20' as const, color: '#bb4bd3', enabled: layers.sma20 }, { key: 'sma50' as const, color: '#df8a3b', enabled: layers.sma50 }, { key: 'sma200' as const, color: '#95813b', enabled: layers.sma200 }].filter(series => series.enabled);
    const selectedLevels = [...(layers.sr1 ? [analysis.supports[0], analysis.resistances[0]] : []), ...(layers.sr2 ? [analysis.supports[1], analysis.resistances[1]] : [])].filter((level): level is PriceZone => Boolean(level));
    let hoverMinPrice = 0; let hoverMaxPrice = 1; let hoverPriceTop = 62; let hoverPriceBottom = 250;
    const clampStart = (candidate: number, count = visibleCount) => Math.max(0, Math.min(Math.round(candidate), bars.length - count));
    const draw = () => {
      const width = Math.max(canvas.clientWidth, 320);
      const height = Math.max(canvas.clientHeight, 300);
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) { canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio) }
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.clearRect(0, 0, width, height);
      context.fillStyle = '#181d27'; context.fillRect(0, 0, width, height);
      const left = 14; const right = width - 70; const top = 62; const axisBottom = height - 24; const hasVolumePane = layers.volume; const volumeTop = hasVolumePane ? height - 98 : axisBottom; const volumeBarsTop = volumeTop + 18; const priceBottom = hasVolumePane ? volumeTop - 12 : axisBottom;
      const plotWidth = Math.max(right - left, 10); const visible = bars.slice(startIndex, startIndex + visibleCount); const step = plotWidth / visible.length;
      const displayPrice = quote.session === 'CLOSED' && quote.regularClose != null ? quote.regularClose : quote.price; const candleLow = Math.min(...visible.map(point => point.low), layers.currentPrice ? displayPrice : Number.POSITIVE_INFINITY); const candleHigh = Math.max(...visible.map(point => point.high), layers.currentPrice ? displayPrice : Number.NEGATIVE_INFINITY); const visibleAtr = visible.at(-1)?.indicators.atr14 || candleHigh * .01; const levelDistanceLimit = Math.max(.05, Math.min(.1, visibleAtr / Math.max(displayPrice, .01) * 6)); const scaleLevels = selectedLevels.filter(level => !level.major && level.distancePct <= levelDistanceLimit); const rawLow = Math.min(candleLow, ...scaleLevels.map(level => level.lower)); const rawHigh = Math.max(candleHigh, ...scaleLevels.map(level => level.upper)); const padding = Math.max((rawHigh - rawLow) * .06, visibleAtr * 1.5, rawHigh * .002, .01); const minPrice = rawLow - padding; const maxPrice = rawHigh + padding; const priceSpan = Math.max(maxPrice - minPrice, .01); const maxVolume = Math.max(...visible.flatMap(point => [point.volume, layers.volumeMA20 ? point.indicators.volumeMA20 || 0 : 0]), 1); hoverMinPrice = minPrice; hoverMaxPrice = maxPrice; hoverPriceTop = top; hoverPriceBottom = priceBottom;
      const priceY = (value: number) => top + ((maxPrice - value) / priceSpan) * (priceBottom - top);
      const xFor = (visibleIndex: number) => left + visibleIndex * step + step / 2;
      context.font = '10px Arial'; context.textBaseline = 'middle'; context.strokeStyle = '#2a313d'; context.fillStyle = '#8d97a8'; context.lineWidth = 1;
      context.setLineDash([3, 4]);
      for (let line = 0; line <= 5; line += 1) { const fraction = line / 5; const y = top + fraction * (priceBottom - top); if (layers.grid) { context.beginPath(); context.moveTo(left, y); context.lineTo(right, y); context.stroke() } const price = maxPrice - fraction * priceSpan; context.fillText(price.toFixed(price >= 100 ? 2 : 3), right + 8, y) }
      for (let line = 0; line <= 5; line += 1) { const x = left + (line / 5) * plotWidth; if (layers.grid) { context.beginPath(); context.moveTo(x, top); context.lineTo(x, axisBottom); context.stroke() } const visiblePosition = Math.min(visible.length - 1, Math.round((line / 5) * (visible.length - 1))); const label = formatPricePointTime(visible[visiblePosition]?.timestamp, interval).replace(' ET', ''); context.textAlign = line === 0 ? 'left' : line === 5 ? 'right' : 'center'; context.fillText(label, x, height - 10) }
      context.setLineDash([]); context.textAlign = 'left';
      const candleWidth = Math.max(1, Math.min(8, step * .62));
      context.save(); context.beginPath(); context.rect(left, top, plotWidth, Math.max(priceBottom - top, 1)); context.clip();
      visible.forEach((point, index) => { const x = xFor(index); const rising = point.close >= point.open; const color = rising ? '#16c784' : '#f45b69'; const openY = priceY(point.open); const closeY = priceY(point.close); context.globalAlpha = point.status === 'STALE' ? .38 : point.isFinal ? (point.session === 'REGULAR' ? 1 : .58) : .62; context.strokeStyle = color; context.fillStyle = color; context.lineWidth = 1; context.beginPath(); context.moveTo(x, priceY(point.high)); context.lineTo(x, priceY(point.low)); context.stroke(); context.fillRect(x - candleWidth / 2, Math.min(openY, closeY), candleWidth, Math.max(Math.abs(openY - closeY), 1.2)); context.globalAlpha = 1 });
      movingAverages.forEach(series => { context.strokeStyle = series.color; context.lineWidth = 1.35; context.beginPath(); let started = false; visible.forEach((point, index) => { const value = point.indicators[series.key]; if (value == null) return; const x = xFor(index); const y = priceY(value); if (!started) { context.moveTo(x, y); started = true } else context.lineTo(x, y) }); if (started) context.stroke() });
      selectedLevels.forEach(level => { const color = level.role === 'support' ? '#35a8e0' : '#d98bea'; if (level.midpoint < minPrice || level.midpoint > maxPrice) return; const upperY = priceY(level.upper); const lowerY = priceY(level.lower); context.globalAlpha = .075; context.fillStyle = color; context.fillRect(left, upperY, plotWidth, Math.max(lowerY - upperY, 2)); context.globalAlpha = 1; context.strokeStyle = color; context.lineWidth = 1; context.setLineDash([6, 4]); context.beginPath(); context.moveTo(left, priceY(level.midpoint)); context.lineTo(right, priceY(level.midpoint)); context.stroke(); context.setLineDash([]); if (layers.levelLabels) { context.fillStyle = color; context.font = 'bold 10px Arial'; context.fillText(level.id, right - 25, priceY(level.midpoint) - 8) } });
      if (layers.currentPrice && displayPrice >= minPrice && displayPrice <= maxPrice) { const y = priceY(displayPrice); const currentChange = quoteChangePct(quote); const pricePrefix = currentPricePrefix(quote.session); context.strokeStyle = '#c7cfdb'; context.lineWidth = 1; context.setLineDash([2, 4]); context.beginPath(); context.moveTo(left, y); context.lineTo(right, y); context.stroke(); context.setLineDash([]); context.fillStyle = '#536176'; context.fillRect(right + 3, y - 11, 66, 22); context.fillStyle = '#fff'; context.font = 'bold 8px Arial'; context.textAlign = 'center'; context.fillText(`${pricePrefix} ${displayPrice.toFixed(displayPrice >= 100 ? 2 : 3)}`, right + 36, y - 3); context.font = '8px Arial'; context.fillStyle = currentChange == null || currentChange >= 0 ? '#74dfb7' : '#ff8a95'; context.fillText(formatPct(currentChange), right + 36, y + 6); context.textAlign = 'left' }
      context.restore();
      if (layers.levelLabels) { let upperBadge = 0; let lowerBadge = 0; selectedLevels.filter(level => level.midpoint < minPrice || level.midpoint > maxPrice).forEach(level => { const color = level.role === 'support' ? '#35a8e0' : '#d98bea'; context.fillStyle = color; context.font = 'bold 9px Arial'; if (level.midpoint > maxPrice && upperBadge < 2) { context.fillText(`${level.id} ↑${level.midpoint.toFixed(2)}`, left + upperBadge * 82, top + 9); upperBadge += 1 } else if (level.midpoint < minPrice && lowerBadge < 2) { context.fillText(`${level.id} ↓${level.midpoint.toFixed(2)}`, left + lowerBadge * 82, priceBottom - 7); lowerBadge += 1 } }) }
      const latestVisible = visible.at(-1)!; movingAverages.forEach((series, index) => { const value = latestVisible.indicators[series.key]; if (value == null || (value >= minPrice && value <= maxPrice)) return; context.fillStyle = series.color; context.font = 'bold 9px Arial'; context.fillText(`${series.key.toUpperCase()} ${value.toFixed(2)} ${value > maxPrice ? '↑' : '↓'}`, left + index * 112, value > maxPrice ? top + 22 : priceBottom - 18) });
      if (hasVolumePane) { const volumePoint = hoverIndex != null ? bars[hoverIndex] : latestVisible; context.fillStyle = '#8893a4'; context.font = '9px Arial'; const volumeParts = [layers.volume ? 'Volume' : '', layers.volumeMA20 && volumePoint.indicators.volumeMA20 != null ? `MA20 ${(volumePoint.indicators.volumeMA20 / 1_000_000).toFixed(2)}M` : '', layers.volumeRatio && volumePoint.indicators.volumeRatio != null ? `Ratio ${volumePoint.indicators.volumeRatio.toFixed(2)}×` : '', layers.atr14 && volumePoint.indicators.atr14 != null ? `ATR14 ${volumePoint.indicators.atr14.toFixed(2)}` : ''].filter(Boolean); context.fillText(volumeParts.join('  |  '), left, volumeTop + 8); context.save(); context.beginPath(); context.rect(left, volumeBarsTop, plotWidth, Math.max(axisBottom - volumeBarsTop, 1)); context.clip(); if (layers.volume) visible.forEach((point, index) => { const x = xFor(index); const color = point.close >= point.open ? '#16c784' : '#f45b69'; const volumeHeight = (point.volume / maxVolume) * Math.max(axisBottom - volumeBarsTop - 2, 1); context.globalAlpha = .34; context.fillStyle = color; context.fillRect(x - candleWidth / 2, axisBottom - volumeHeight, candleWidth, volumeHeight); context.globalAlpha = 1 }); if (layers.volumeMA20) { context.strokeStyle = '#9ca8b9'; context.lineWidth = 1.1; context.beginPath(); let volumeStarted = false; visible.forEach((point, index) => { const value = point.indicators.volumeMA20; if (value == null) return; const x = xFor(index); const y = axisBottom - (value / maxVolume) * Math.max(axisBottom - volumeBarsTop - 2, 1); if (!volumeStarted) { context.moveTo(x, y); volumeStarted = true } else context.lineTo(x, y) }); if (volumeStarted) context.stroke() } context.restore() }
      if (hoverIndex != null && hoverIndex >= startIndex && hoverIndex < startIndex + visible.length) { const localIndex = hoverIndex - startIndex; const point = bars[hoverIndex]; const x = xFor(localIndex); const y = Math.max(top, Math.min(hoverY, priceBottom)); context.strokeStyle = '#98a4b7'; context.lineWidth = 1; context.setLineDash([4, 4]); context.beginPath(); context.moveTo(x, top); context.lineTo(x, axisBottom); context.moveTo(left, y); context.lineTo(right, y); context.stroke(); context.setLineDash([]); const cursorPrice = maxPrice - ((y - top) / (priceBottom - top)) * priceSpan; context.fillStyle = '#647086'; context.fillRect(right + 3, Math.min(y - 10, priceBottom - 20), 66, 20); context.fillStyle = '#fff'; context.font = '10px Arial'; context.textAlign = 'center'; context.fillText(cursorPrice.toFixed(cursorPrice >= 100 ? 2 : 3), right + 36, Math.min(y, priceBottom - 10)); context.textAlign = 'left'; setHoveredBar(current => current?.timestamp === point.timestamp ? current : point) }
    };
    const pointerPosition = (event: PointerEvent) => { const rect = canvas.getBoundingClientRect(); return { x: event.clientX - rect.left, y: event.clientY - rect.top, width: rect.width } };
    const onPointerMove = (event: PointerEvent) => { const position = pointerPosition(event); const left = 14; const right = position.width - 70; const step = Math.max((right - left) / visibleCount, .01); if (dragging) { startIndex = clampStart(dragStartIndex - (position.x - dragStartX) / step); hoverIndex = null; setHoveredBar(null); setHoveredZone(null); draw(); return } const local = Math.max(0, Math.min(visibleCount - 1, Math.floor((position.x - left) / step))); hoverIndex = startIndex + local; hoverY = position.y; if (position.y >= hoverPriceTop && position.y <= hoverPriceBottom) { const pointerPrice = hoverMaxPrice - ((position.y - hoverPriceTop) / Math.max(hoverPriceBottom - hoverPriceTop, 1)) * (hoverMaxPrice - hoverMinPrice); const zone = selectedLevels.find(level => pointerPrice >= level.lower && pointerPrice <= level.upper) || null; setHoveredZone(current => current?.id === zone?.id ? current : zone) } else setHoveredZone(null); draw() };
    const onPointerDown = (event: PointerEvent) => { dragging = true; dragStartX = pointerPosition(event).x; dragStartIndex = startIndex; canvas.setPointerCapture(event.pointerId) };
    const onPointerUp = (event: PointerEvent) => { dragging = false; if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId) };
    const onPointerLeave = () => { if (!dragging) { hoverIndex = null; setHoveredBar(null); setHoveredZone(null); draw() } };
    const onWheel = (event: WheelEvent) => { event.preventDefault(); const rect = canvas.getBoundingClientRect(); const left = 14; const right = rect.width - 70; const anchorRatio = Math.max(0, Math.min(1, (event.clientX - rect.left - left) / Math.max(right - left, 1))); const anchorIndex = startIndex + Math.floor(anchorRatio * visibleCount); const nextCount = Math.max(Math.min(20, bars.length), Math.min(bars.length, Math.round(visibleCount * (event.deltaY > 0 ? 1.18 : .84)))); startIndex = clampStart(anchorIndex - anchorRatio * nextCount, nextCount); visibleCount = nextCount; hoverIndex = null; setHoveredBar(null); draw() };
    const resizeObserver = new ResizeObserver(draw); resizeObserver.observe(canvas); canvas.addEventListener('pointermove', onPointerMove); canvas.addEventListener('pointerdown', onPointerDown); canvas.addEventListener('pointerup', onPointerUp); canvas.addEventListener('pointercancel', onPointerUp); canvas.addEventListener('pointerleave', onPointerLeave); canvas.addEventListener('wheel', onWheel, { passive: false }); draw();
    return () => { resizeObserver.disconnect(); canvas.removeEventListener('pointermove', onPointerMove); canvas.removeEventListener('pointerdown', onPointerDown); canvas.removeEventListener('pointerup', onPointerUp); canvas.removeEventListener('pointercancel', onPointerUp); canvas.removeEventListener('pointerleave', onPointerLeave); canvas.removeEventListener('wheel', onWheel) };
  }, [analysis, bars, symbol, rangeLabel, interval, quote, layers]);
  if (bars.length < 2) return <div className="position-chart-empty">暂时没有可显示的K线数据</div>;
  const activeBar = hoveredBar || bars.at(-1)!; const indicators = activeBar.indicators; const barReturn = activeBar.barReturn ?? (activeBar.open ? activeBar.close / activeBar.open - 1 : 0);
  const activePresetLabel = activePreset === 'custom' ? '自定义' : CHART_PRESET_META[activePreset].label;
  const hoverSmas = hoveredBar ? ([layers.sma20 ? `SMA20 ${hoveredBar.indicators.sma20?.toFixed(2) || '—'}` : '', layers.sma50 ? `SMA50 ${hoveredBar.indicators.sma50?.toFixed(2) || '—'}` : '', layers.sma200 ? `SMA200 ${hoveredBar.indicators.sma200?.toFixed(2) || '—'}` : ''].filter(Boolean)) : [];
  const hoverVolumeMetrics = hoveredBar ? ([layers.volume ? `Volume ${hoveredBar.volume.toLocaleString('en-US')}` : '', layers.volume && layers.volumeMA20 ? `Vol MA20 ${hoveredBar.indicators.volumeMA20?.toLocaleString('en-US', { maximumFractionDigits: 0 }) || '—'}` : '', layers.volume && layers.volumeRatio ? `Ratio ${hoveredBar.indicators.volumeRatio?.toFixed(2) || '—'}×` : '', layers.volume && layers.atr14 ? `ATR14 ${hoveredBar.indicators.atr14?.toFixed(2) || '—'}` : ''].filter(Boolean)) : [];
  return <div className="professional-chart"><div className="professional-chart-stage">
    <button ref={layerButtonRef} className="chart-layer-button" type="button" aria-expanded={layerMenuOpen} onClick={() => setLayerMenuOpen(value => !value)}><span aria-hidden="true">☷</span> 显示内容 <em>{activePresetLabel}</em></button>
    {layerMenuOpen && <aside ref={layerMenuRef} className="chart-layer-menu" aria-label="K线显示内容"><div className="chart-layer-menu-heading"><div><strong>显示模式</strong><span>点击后立即应用</span></div>{activePreset === 'custom' && <em>自定义</em>}</div><div className="chart-preset-row">{(Object.keys(CHART_PRESET_META) as ChartPreset[]).map(preset => <button type="button" key={preset} className={activePreset === preset ? 'active' : ''} aria-pressed={activePreset === preset} onClick={() => onPreset(preset)}><span>{CHART_PRESET_META[preset].label}</span>{preset === 'standard' && <small>推荐</small>}</button>)}</div><p className="chart-preset-description">{activePreset === 'custom' ? '已手动调整单项显示内容' : CHART_PRESET_META[activePreset].description}</p>{CHART_LAYER_GROUPS.map(group => <section key={group.title}><strong>{group.title}</strong><div>{group.items.map(item => <label key={item.key}><input type="checkbox" checked={layers[item.key]} onChange={event => onLayerChange(item.key, event.target.checked)}/><span>{item.label}</span></label>)}</div></section>)}<section><strong>行情数据</strong><label className="menu-readonly"><input type="checkbox" checked={showExtendedHours} readOnly disabled/><span>盘前盘后：{showExtendedHours ? '已开启' : '已关闭'}（外部切换）</span></label></section><button className="restore-standard-button" type="button" onClick={() => onPreset('standard')}>恢复标准设置</button><small>隐藏仅影响图表；后台指标计算与 AI 结构化分析始终保持完整。</small></aside>}
    <canvas ref={canvasRef} aria-label={`${symbol} ${rangeLabel}专业K线图，可使用滚轮缩放并拖动浏览`}/>
    <div className="professional-chart-legend"><div><strong>{symbol}</strong><span>{formatPricePointTime(activeBar.timestamp, interval)} · {sessionLabel(activeBar.session)} · {barStatusLabel(activeBar)}</span></div>{layers.ohlc && <div className="ohlcv"><span>开 <b>{activeBar.open.toFixed(2)}</b></span><span>高 <b>{activeBar.high.toFixed(2)}</b></span><span>低 <b>{activeBar.low.toFixed(2)}</b></span><span>收 <b className={activeBar.close >= activeBar.open ? 'up' : 'down'}>{activeBar.close.toFixed(2)}</b></span><span>本K <b className={barReturn >= 0 ? 'up' : 'down'}>{formatPct(barReturn)}</b></span></div>}{(layers.sma20 || layers.sma50 || layers.sma200) && <div className="moving-average-legend">{layers.sma20 && <span className="sma20" title={`最近20根 ${interval} K线的简单移动平均价`}>SMA20@{interval}&nbsp; {indicators.sma20?.toFixed(2) || '—'}</span>}{layers.sma50 && <span className="sma50" title={`最近50根 ${interval} K线的简单移动平均价`}>SMA50@{interval}&nbsp; {indicators.sma50?.toFixed(2) || '—'}</span>}{layers.sma200 && <span className="sma200" title={`最近200根 ${interval} K线的简单移动平均价`}>SMA200@{interval}&nbsp; {indicators.sma200?.toFixed(2) || '—'}</span>}</div>}</div>
    {hoveredBar && <div className="chart-hover-card"><strong>{formatPricePointTime(hoveredBar.timestamp, interval)} · {barStatusLabel(hoveredBar)}</strong><span>O {hoveredBar.open.toFixed(2)} · H {hoveredBar.high.toFixed(2)} · L {hoveredBar.low.toFixed(2)} · C {hoveredBar.close.toFixed(2)}</span><span>本K {formatPct(hoveredBar.barReturn)} · 当日 {formatPct(analysis.dayReturn)}</span>{hoverVolumeMetrics.length > 0 && <span>{hoverVolumeMetrics.join(' · ')}</span>}{hoverSmas.length > 0 && <span>{hoverSmas.join(' · ')}</span>}{hoveredZone && <span className={hoveredZone.role === 'support' ? 'zone-support' : 'zone-resistance'}>{hoveredZone.id} {formatUnitPrice(hoveredZone.lower)}–{formatUnitPrice(hoveredZone.upper)} · 强度 {hoveredZone.score.toFixed(1)} · 触碰 {hoveredZone.touchCount} 次 · {hoveredZone.timeframe}</span>}</div>}
  </div></div>;
}

function TickerTape({ items }: { items: TickerItem[] }) {
  const [selectedKeys, setSelectedKeys] = useState<string[]>(DEFAULT_TAPE_KEYS);
  const [managerOpen, setManagerOpen] = useState(false);
  const managerRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const restore = window.setTimeout(() => {
      try {
        const saved = JSON.parse(localStorage.getItem(TAPE_STORAGE_KEY) || 'null');
        if (Array.isArray(saved) && saved.length && saved.every(key => typeof key === 'string')) setSelectedKeys(saved);
      } catch {
        localStorage.removeItem(TAPE_STORAGE_KEY);
      }
    }, 0);
    return () => window.clearTimeout(restore);
  }, []);
  useEffect(() => {
    if (!managerOpen) return;
    const close = (event: MouseEvent) => {
      if (managerRef.current && !managerRef.current.contains(event.target as Node)) setManagerOpen(false);
    };
    document.addEventListener('mousedown', close);
    return () => document.removeEventListener('mousedown', close);
  }, [managerOpen]);
  const selectedItems = items.filter(item => selectedKeys.includes(item.key));
  const visibleItems = selectedItems.length ? selectedItems : items.slice(0, 1);
  const toggleItem = (key: string) => {
    setSelectedKeys(current => {
      const next = current.includes(key) ? (current.length === 1 ? current : current.filter(item => item !== key)) : [...current, key];
      localStorage.setItem(TAPE_STORAGE_KEY, JSON.stringify(next));
      return next;
    });
  };
  return <div className="ticker-tape-shell" ref={managerRef}><div className="ticker-tape" aria-label="主要市场行情"><div className="ticker-track">{[false, true].map(duplicate => <div className="ticker-group" aria-hidden={duplicate} key={duplicate ? 'ticker-copy' : 'ticker-primary'}>{visibleItems.map(item => <div className="ticker-item" key={`${duplicate ? 'copy' : 'primary'}-${item.key}`}><span>{item.label}</span><strong>{formatTickerValue(item.value)}{item.unit}</strong>{item.change_pct != null && Number.isFinite(item.change_pct) && <em className={item.change_pct >= 0 ? 'positive' : 'negative'}>{formatPct(item.change_pct)}</em>}</div>)}</div>)}</div></div><button className="ticker-manage-button" type="button" aria-label="管理行情滚动条" aria-expanded={managerOpen} onClick={() => setManagerOpen(open => !open)}>+</button>{managerOpen && <section className="ticker-manager" role="dialog" aria-label="管理行情滚动条"><div className="ticker-manager-header"><div><strong>行情滚动条</strong><span>选择要展示的真实市场指标</span></div><button type="button" aria-label="关闭" onClick={() => setManagerOpen(false)}>×</button></div><div className="ticker-options">{items.map(item => { const selected = selectedKeys.includes(item.key); return <button type="button" key={item.key} className={selected ? 'selected' : ''} aria-pressed={selected} onClick={() => toggleItem(item.key)}><span><strong>{item.label}</strong><small>{formatTickerValue(item.value)}{item.unit}</small></span><em>{selected ? '✓' : '+'}</em></button> })}</div><p>选择会保存在当前浏览器；数据暂时不可用时显示“—”。</p></section>}</div>;
}

export default function Home() {
  const refreshInFlight = useRef(false);
  const messageListRef = useRef<HTMLDivElement>(null);
  const chatRequestRef = useRef<AbortController | null>(null);
  const accessRef = useRef<AccessStatus | null>(null);
  const [section, setSection] = useState<MainSection>('core'); const [researchTool, setResearchTool] = useState<ResearchTool>('stock'); const [labTool, setLabTool] = useState<LabTool>('overview');
  const [portfolio, setPortfolio] = useState(demoPortfolio); const [risk, setRisk] = useState(demoRisk); const [tape, setTape] = useState(demoTape); const [isDemo, setIsDemo] = useState(true);
  const [history, setHistory] = useState<number[]>(demoPortfolio.positions.map((_, i) => 97420 + i * 900)); const [activity, setActivity] = useState<ActivityItem[]>([]); const [activityWarning, setActivityWarning] = useState(''); const [monitoringUniverse, setMonitoringUniverse] = useState<MonitoringUniverse>({ intraday: [], source: 'TECH_30', scan_interval_seconds: 60, price_pct_threshold: .03, volume_pace_threshold: 2.5 }); const [watchlist, setWatchlist] = useState<WatchlistItem[]>([]); const [priceAlerts, setPriceAlerts] = useState<PriceAlertItem[]>([]);
  const [dataError, setDataError] = useState(''); const [isRefreshing, setIsRefreshing] = useState(false); const [lastUpdated, setLastUpdated] = useState<Date | null>(null); const [chartSnapshotMessage, setChartSnapshotMessage] = useState(''); const [researchResult, setResearchResult] = useState<ResearchResultState | null>(null); const [researchBusy, setResearchBusy] = useState(false);
  const [access, setAccess] = useState<AccessStatus | null>(null); const [accessLoading, setAccessLoading] = useState(true); const [loginUsername, setLoginUsername] = useState('owner'); const [loginPassword, setLoginPassword] = useState(''); const [loginError, setLoginError] = useState(''); const [loginBusy, setLoginBusy] = useState(false);
  const [currentTime, setCurrentTime] = useState(''); const [easternTime, setEasternTime] = useState('');
  const [costReport, setCostReport] = useState<CostReport | null>(null); const [costLoading, setCostLoading] = useState(false); const [costError, setCostError] = useState('');
  const [alertFilter, setAlertFilter] = useState<'all' | 'positions' | 'p0'>('all'); const [researchTicker, setResearchTicker] = useState('NVDA'); const [chatInput, setChatInput] = useState(''); const [chatBusy, setChatBusy] = useState(false); const [chatContext, setChatContext] = useState('盯盘总览'); const [chatMode, setChatMode] = useState<ChatMode>('agent_v2'); const [allowAgentWeb, setAllowAgentWeb] = useState(true); const [chatProgress, setChatProgress] = useState('');
  const [pageSelection, setPageSelection] = useState<PageSelection | undefined>();
  const [messages, setMessages] = useState<ChatMessage[]>([{ id: 1, role: 'assistant', text: '我会结合左侧当前页面、持仓和市场数据回答。你可以直接在这里提问。', meta: '事实与推断会分开标注' }]);
  const isGuest = access?.role === 'guest';
  useEffect(() => { accessRef.current = access }, [access]);
  const refreshData = useCallback(async (refreshChartSnapshots = false) => {
    if (refreshInFlight.current) return;
    refreshInFlight.current = true;
    setIsRefreshing(true);
    setDataError('');
    const failures: string[] = [];
    let coreSuccesses = 0;
    let anySuccess = false;
    let refreshedPortfolio: PortfolioResponse | null = null;
    async function load<T>(label: string, request: Promise<T>, apply: (value: T) => void, core = false) {
      try {
        const value = await request;
        apply(value);
        anySuccess = true;
        if (core) coreSuccesses += 1;
      } catch {
        failures.push(label);
      }
    }
    try {
      await Promise.all([
        load('组合', apiJson<PortfolioResponse>('/api/portfolio'), value => { refreshedPortfolio = value; setPortfolio(value); if (value.history?.equity.length) setHistory(value.history.equity) }, true),
        load('风险', apiJson<RiskResponse>('/api/risk'), setRisk, true),
        load('市场行情', apiJson<{ items: TickerItem[] }>('/api/tickertape'), value => setTape(value.items), true),
        load('异常归档', apiJson<ActivityResponse>('/api/activity?days=2&realtime_only=true'), value => { setActivity(value.items); setActivityWarning(value.warning || '') }),
        load('监控股票池', apiJson<MonitoringUniverse>('/api/monitoring/universe'), setMonitoringUniverse),
        load('Watchlist', apiJson<{ items: WatchlistItem[] }>('/api/watchlist'), value => setWatchlist(value.items)),
        load('价格提醒', apiJson<{ items: PriceAlertItem[] }>('/api/price-alerts'), value => setPriceAlerts(value.items)),
      ]);
      if (refreshChartSnapshots && accessRef.current?.role === 'owner' && refreshedPortfolio) {
        const symbols = (refreshedPortfolio as PortfolioResponse).positions.map(position => position.symbol);
        if (symbols.length) {
          try {
            const queued = await apiJson<ChartSnapshotRefreshStatus>('/api/public/snapshot/price-history/refresh', { method: 'POST', body: JSON.stringify({ symbols, ranges: PRICE_RANGES.map(item => item.key) }) });
            setChartSnapshotMessage(queued.started === false ? '持仓图表快照已在后台更新' : `正在后台生成持仓图表快照（0/${queued.total}）`);
            void (async () => {
              for (let attempt = 0; attempt < 200; attempt += 1) {
                await new Promise<void>(resolve => window.setTimeout(resolve, 3000));
                try {
                  const state = await apiJson<ChartSnapshotRefreshStatus>('/api/public/snapshot/price-history/status');
                  if (state.status === 'running') {
                    setChartSnapshotMessage(`正在后台生成持仓图表快照（${state.published}/${state.total}）`);
                    continue;
                  }
                  setChartSnapshotMessage(state.failed ? `持仓图表快照已更新 ${state.published} 项，${state.failed} 项失败` : `持仓图表快照已更新（${state.published}/${state.total}）`);
                  break;
                } catch {
                  setChartSnapshotMessage('持仓图表快照已在后台启动');
                  break;
                }
              }
            })();
          } catch {
            setChartSnapshotMessage('页面数据已刷新，但持仓图表快照启动失败');
          }
        }
      }
      setIsDemo(coreSuccesses === 0);
      if (anySuccess) {
        const updatedAt = isGuest && accessRef.current?.snapshot_updated_at ? new Date(accessRef.current.snapshot_updated_at) : new Date();
        setLastUpdated(updatedAt);
      }
      if (coreSuccesses === 0) setDataError('核心数据刷新失败，继续显示上一次数据。请检查 Web API 或连接设置。');
      else if (failures.length) setDataError(`部分数据刷新失败：${failures.join('、')}。其他模块已更新。`);
    } finally {
      refreshInFlight.current = false;
      setIsRefreshing(false);
    }
  }, [isGuest]);
  useEffect(() => {
    let active = true;
    localStorage.removeItem('ownerToken');
    localStorage.removeItem('dashboard:owner_token');
    void authStatus().then(status => {
      if (!active) return;
      setApiAccessRole(status.role);
      setAccess(status);
      setAccessLoading(false);
    }).catch(error => {
      if (!active) return;
      setLoginError(error instanceof Error ? error.message : '无法连接登录服务');
      setApiAccessRole('anonymous');
      setAccess({ authenticated: false, role: 'anonymous', username: '', guest_enabled: false, auth_configured: true, snapshot_updated_at: null, session_expires_at: null });
      setAccessLoading(false);
    });
    return () => { active = false };
  }, []);
  useEffect(() => { if (access?.authenticated) void refreshData() }, [access?.authenticated, access?.role, refreshData]);
  useEffect(() => () => chatRequestRef.current?.abort(), []);
  useEffect(() => { const updateClock = () => { const now = new Date(); setCurrentTime(formatClock(now)); setEasternTime(formatClockInZone(now, 'America/New_York')) }; updateClock(); const timer = window.setInterval(updateClock, 1000); return () => window.clearInterval(timer) }, []);
  const refreshCosts = useCallback(async (recentFilter = 'all', offset = 0, syncPrices = false) => { setCostLoading(true); setCostError(''); try { const query = new URLSearchParams({ limit: '100', filter: recentFilter, offset: String(offset) }); setCostReport(await apiJson<CostReport>(`/api/costs${syncPrices ? '/refresh' : ''}?${query}`, syncPrices ? { method: 'POST' } : undefined)) } catch (error) { setCostError(error instanceof Error ? error.message : 'unknown') } finally { setCostLoading(false) } }, []);
  useEffect(() => {
    if (section !== 'cost') return;
    const frame = window.requestAnimationFrame(() => { void refreshCosts() });
    return () => window.cancelAnimationFrame(frame);
  }, [section, refreshCosts]);
  useEffect(() => {
    const ticker = researchTicker.trim().toUpperCase();
    if (section !== 'research' || !ticker || researchBusy || (researchResult?.ticker === ticker && (researchResult.engine || researchResult.progress))) return;
    let cancelled = false;
    void apiJson<StockResearchResult>(`/api/research/results/${encodeURIComponent(ticker)}`).then(cached => {
      if (!cancelled) setResearchResult({ tool: researchTool, ticker, result: { text: '' }, engine: cached, progress: cached.module_status });
    }).catch(() => undefined);
    return () => { cancelled = true };
  }, [section, researchTool, researchTicker, researchBusy, researchResult]);
  useEffect(() => { const panel = messageListRef.current; if (panel) panel.scrollTo({ top: panel.scrollHeight, behavior: 'smooth' }) }, [messages, chatBusy]);
  const sendMessage = useCallback(async (raw: string, contextOverride?: string, selectionOverride?: PageSelection): Promise<ToolResult | undefined> => {
    const text = raw.trim();
    if (!text || chatBusy || isGuest) return;
    const messageContext = contextOverride || chatContext;
    const requestMode = chatMode;
    const researchMatches = researchResult?.ticker === researchTicker && researchResult?.tool === researchTool;
    const pageContext: PageContext = {
      section, label: messageContext.slice(0, 200), tool: section === 'research' ? researchTool : section === 'lab' ? labTool : '',
      captured_at: new Date().toISOString(),
      data_status: section === 'core' ? (!isDemo && !dataError ? 'available' : 'unavailable') : section === 'research' && researchMatches && researchResult?.engine && !researchBusy ? 'available' : 'unavailable',
      selection: section === 'research' ? { kind: 'research', ticker: researchTicker.slice(0, 32), record_id: researchMatches ? researchResult?.engine?.run_id || '' : '', occurred_at: researchMatches ? researchResult?.engine?.generated_at || '' : '', excerpt: researchMatches && !researchBusy ? (researchResult?.engine?.investment_thesis || researchResult?.engine?.core_thesis || researchResult?.result.text || '').slice(0, 4000) : '' } : contextOverride !== undefined ? selectionOverride : pageSelection,
    };
    const now = Date.now();
    let progressId: number | null = null;
    const controller = new AbortController();
    chatRequestRef.current = controller;
    setMessages(items => [...items, { id: now, role: 'user', text, mode: requestMode }]);
    setChatInput('');
    setChatBusy(true);
    setChatProgress(`${CHAT_LABELS[requestMode]} 正在规划…`);
    try {
      const version = requestMode === 'agent_v3' ? 'v3' : 'v2';
      let response = await askAgentV2(text, agentSessionId(version), allowAgentWeb, controller.signal, version, pageContext);
      if ('job_id' in response) {
        const initialJob = response;
        progressId = now + 1;
        setMessages(items => [...items, { id: progressId as number, role: 'assistant', text: initialJob.progress || `${CHAT_LABELS[requestMode]} 任务已进入后台队列。`, meta: '执行进度 · 最长等待 15 分钟', mode: requestMode }]);
        response = await waitForAgentJob(initialJob, controller.signal, job => {
          const progress = job.progress || `${CHAT_LABELS[requestMode]}：${job.agent_status}`;
          setChatProgress(progress);
          setMessages(items => items.map(item => item.id === progressId ? { ...item, text: progress } : item));
        }, version);
      }
      const result: ToolResult = { text: response.answer || response.error || `${CHAT_LABELS[requestMode]} 没有返回可显示的答案。`, intent: response.route.kind };
      const finalMessage: ChatMessage = {
        id: progressId ?? now + 1,
        role: 'assistant',
        ...result,
        mode: requestMode,
        meta: `运行 ${response.run_id} · 上下文：${messageContext}${requestMode === 'agent_v3' ? '（已传递，事实由工具核验）' : ''}`,
        agent: {
          status: response.status,
          route: response.route.kind,
          answerMode: response.answer_mode,
          elapsedMs: response.elapsed_ms,
          verified: response.status !== 'failed' && response.verification.ok,
          capabilities: response.plan.tasks.map(task => task.capability),
          webRequested: response.policy.web_requested,
          webEnabled: response.policy.web_enabled,
          webAllowed: response.policy.web_allowed,
          warnings: [...response.verification.warnings, ...response.verification.unknown_citations.map(id => `未知证据引用：${id}`), ...response.verification.ungrounded_numbers.map(value => `未落地数字：${value}`), ...synthesisWarnings(response.synthesis)],
          synthesis: SYNTHESIS_LABELS[response.synthesis?.outcome || ''] || '',
          synthesisFallback: response.synthesis?.outcome === 'fallback',
          rewritten: response.request?.metadata?.rewritten ? String(response.request.metadata.resolution_note || response.request.text) : '',
          subAgents: response.sub_agents || [],
        },
        evidence: uniqueEvidence(response.evidence || []),
      };
      setMessages(items => progressId == null ? [...items, finalMessage] : items.map(item => item.id === progressId ? finalMessage : item));
      return result;
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') return;
      const detail = error instanceof Error ? error.message : 'unknown';
      const result = { text: `查询失败：${detail}。请检查 Web API、凭据或 owner token。` };
      const errorMessage: ChatMessage = { id: progressId ?? now + 1, role: 'assistant', ...result, mode: requestMode, meta: `连接错误 · 上下文：${messageContext}` };
      setMessages(items => progressId == null ? [...items, errorMessage] : items.map(item => item.id === progressId ? errorMessage : item));
      return result;
    } finally {
      if (chatRequestRef.current === controller) chatRequestRef.current = null;
      setChatBusy(false);
      setChatProgress('');
    }
  }, [allowAgentWeb, chatBusy, chatContext, chatMode, section, researchTool, labTool, researchTicker, researchResult, researchBusy, isDemo, dataError, pageSelection, isGuest]);
  const runResearch: ResearchRun = async (tool, ticker, retry) => {
    if (isGuest) return;
    const resultTicker = ticker.trim().toUpperCase();
    setResearchBusy(true);
    try {
      const modules = RESEARCH_TOOL_MODULES[tool];
      let job = await apiJson<ResearchJob>(retry ? `/api/research/runs/${encodeURIComponent(retry.runId)}/retry` : '/api/research/runs', { method: 'POST', body: JSON.stringify(retry ? { modules: retry.modules } : { ticker: resultTicker, modules, force_refresh: researchResult?.engine?.ticker === resultTicker }) });
      setResearchResult({ tool, ticker: resultTicker, result: { text: '' }, progress: job.module_status || {} });
      for (let attempt = 0; ['PENDING','RUNNING'].includes(job.status) && attempt < 300; attempt += 1) {
        await new Promise(resolve => window.setTimeout(resolve, 1000));
        job = await apiJson<ResearchJob>(`/api/research/runs/${encodeURIComponent(job.job_id)}`);
        setResearchResult({ tool, ticker: resultTicker, result: { text: '' }, progress: job.module_status || {} });
      }
      if (!job.result) throw new Error(job.error || (job.status === 'RUNNING' ? '研究任务仍在运行，请稍后重试' : '研究任务未返回结果'));
      setResearchResult({ tool, ticker: resultTicker, result: { text: '' }, engine: job.result, progress: job.module_status || {} });
    } catch (error) {
      const detail = error instanceof Error ? error.message : 'unknown';
      setResearchResult({ tool, ticker: resultTicker, result: { text: '' }, progress: { run: `FAILED: ${detail}` } });
    } finally {
      setResearchBusy(false);
    }
  };
  const addWatchlist = async (ticker: string) => { const response = await apiJson<{ items: WatchlistItem[] }>('/api/watchlist', { method: 'POST', body: JSON.stringify({ ticker }) }); setWatchlist(response.items) };
  const removeWatchlist = async (ticker: string) => { const response = await apiJson<{ items: WatchlistItem[] }>(`/api/watchlist/${encodeURIComponent(ticker)}`, { method: 'DELETE' }); setWatchlist(response.items) };
  const addMonitoringTicker = async (ticker: string) => { setMonitoringUniverse(await apiJson<MonitoringUniverse>('/api/monitoring/universe', { method: 'POST', body: JSON.stringify({ ticker }) })) };
  const removeMonitoringTicker = async (ticker: string) => { setMonitoringUniverse(await apiJson<MonitoringUniverse>(`/api/monitoring/universe/${encodeURIComponent(ticker)}`, { method: 'DELETE' })) };
  const addPriceAlert = async (ticker: string, direction: 'above' | 'below', target_price: number) => { const response = await apiJson<{ items: PriceAlertItem[] }>('/api/price-alerts', { method: 'POST', body: JSON.stringify({ ticker, direction, target_price }) }); setPriceAlerts(response.items) };
  const removePriceAlert = async (id: number) => { const response = await apiJson<{ items: PriceAlertItem[] }>(`/api/price-alerts/${id}`, { method: 'DELETE' }); setPriceAlerts(response.items) };
  const askWithContext = (prompt: string, context: string, selection?: PageSelection) => { setPageSelection(selection); setChatContext(context); setChatInput(prompt) };
  const askAndSend = (prompt: string, context: string, selection?: PageSelection) => { setPageSelection(selection); setChatContext(context); void sendMessage(prompt, context, selection) };
  const handleChat = (event: FormEvent) => { event.preventDefault(); void sendMessage(chatInput) };
  const applyAccess = (status: AccessStatus) => { setApiAccessRole(status.role); accessRef.current = status; setAccess(status); setLoginError(''); setSection('core'); setResearchTicker('NVDA') };
  const handleLogin = async (event: FormEvent) => {
    event.preventDefault();
    if (!loginUsername.trim() || !loginPassword) return;
    setLoginBusy(true); setLoginError('');
    try { applyAccess(await authLogin(loginUsername, loginPassword)); setLoginPassword('') }
    catch (error) { setLoginError(error instanceof Error ? error.message : '登录失败') }
    finally { setLoginBusy(false) }
  };
  const enterGuestMode = async () => {
    setLoginBusy(true); setLoginError('');
    try { applyAccess(await authGuest()) }
    catch (error) { setLoginError(error instanceof Error ? error.message : '无法进入访客模式') }
    finally { setLoginBusy(false) }
  };
  const leaveSession = async () => {
    chatRequestRef.current?.abort();
    try { await authLogout() } catch { /* clear the local UI even if the network is interrupted */ }
    const anonymous: AccessStatus = { authenticated: false, role: 'anonymous', username: '', guest_enabled: access?.guest_enabled ?? true, auth_configured: true, snapshot_updated_at: access?.snapshot_updated_at ?? null, session_expires_at: null };
    setApiAccessRole('anonymous'); accessRef.current = anonymous; setAccess(anonymous); setPortfolio(demoPortfolio); setRisk(demoRisk); setTape(demoTape); setResearchResult(null); setCostReport(null); setLastUpdated(null); setIsDemo(true);
  };
  const changeSection = (next: MainSection) => { setSection(next); setPageSelection(undefined); setChatContext(next === 'core' ? '盯盘总览' : next === 'research' ? '研究工作台' : next === 'lab' ? '实验室' : '数据查询花费') };
  if (accessLoading) return <main className="access-gate"><div className="access-card loading"><span className="brand-mark">HF</span><strong>正在建立安全会话…</strong></div></main>;
  if (!access?.authenticated) return <main className="access-gate"><section className="access-card"><div className="access-brand"><span className="brand-mark">HF</span><div><h1>Market Intelligence Workbench</h1><p>所有者可以使用完整功能；访客只能查看最后发布的只读快照。</p></div></div><form onSubmit={handleLogin}><label><span>用户名</span><input autoComplete="username" value={loginUsername} onChange={event => setLoginUsername(event.target.value)} disabled={loginBusy}/></label><label><span>密码</span><input type="password" autoComplete="current-password" value={loginPassword} onChange={event => setLoginPassword(event.target.value)} disabled={loginBusy}/></label>{loginError && <div className="access-error">{loginError}</div>}<button className="access-login" type="submit" disabled={loginBusy || !loginUsername.trim() || !loginPassword}>{loginBusy ? '正在验证…' : '所有者登录'}</button></form>{access?.guest_enabled && <><div className="access-divider"><span>或者</span></div><button className="access-guest" type="button" disabled={loginBusy} onClick={() => void enterGuestMode()}>以访客身份查看</button><small>访客不能刷新数据、运行 Agent、修改监控或启动研究与实验。</small></>}</section></main>;
  return <main className="app-shell"><header className="app-header"><div className="brand"><span className="brand-mark">HF</span><span>AI Hedge Fund</span></div><nav className="main-nav" aria-label="主导航">{([['core', '盯盘'], ['research', '研究'], ['lab', '实验室'], ['cost', '花费']] as const).map(([id, label]) => <button key={id} className={section === id ? 'active' : ''} onClick={() => changeSection(id)}>{label}</button>)}</nav><time className="header-clock eastern" aria-label="本机时间" suppressHydrationWarning><span>本机</span>{currentTime || '---- -- -- --:--:--'}</time><time className="header-clock eastern" aria-label="美东时间" suppressHydrationWarning><span>美东</span>{easternTime || '---- -- -- --:--:--'}</time><div className="header-status"><span className={`status-dot ${isGuest || isDemo ? 'demo' : ''}`}/><span>{isGuest ? '访客只读' : isDemo ? '数据连接异常' : '所有者模式'}</span><button className="header-logout" type="button" onClick={() => void leaveSession()}>退出</button></div></header>
    {isGuest && <div className="guest-banner"><strong>访客只读模式</strong><span>当前展示所有者最后发布的快照，所有付费调用、刷新、研究、实验和管理操作均已在服务器端禁用。</span><time>{access.snapshot_updated_at ? `最近发布 ${new Date(access.snapshot_updated_at).toLocaleString('zh-CN')}` : '尚未发布快照'}</time></div>}
    <div className={`product-layout ${isGuest ? 'guest-layout' : ''}`}><section className={`main-workspace section-${section} ${section === 'research' || section === 'lab' ? 'with-sidebar' : ''}`}>{(section === 'research' || section === 'lab') && <aside className="tool-sidebar"><div className="sidebar-label">{section === 'research' ? '研究工具' : '实验工具'}</div>{(section === 'research' ? researchMenu : labMenu).map(item => { const active = section === 'research' ? researchTool === item.id : labTool === item.id; return <button key={item.id} className={active ? 'active' : ''} onClick={() => section === 'research' ? setResearchTool(item.id as ResearchTool) : setLabTool(item.id as LabTool)}><span>{item.icon}</span>{item.label}</button> })}</aside>}<div className="workspace-content">{section === 'core' && <CorePage readOnly={isGuest} portfolio={portfolio} risk={risk} tape={tape} history={history} activity={activity} activityWarning={activityWarning} monitoringUniverse={monitoringUniverse} alertFilter={alertFilter} setAlertFilter={setAlertFilter} refresh={() => void refreshData(true)} refreshing={isRefreshing} lastUpdated={lastUpdated} dataError={dataError} chartSnapshotMessage={chartSnapshotMessage} watchlist={watchlist} priceAlerts={priceAlerts} addMonitoringTicker={addMonitoringTicker} removeMonitoringTicker={removeMonitoringTicker} addWatchlist={addWatchlist} removeWatchlist={removeWatchlist} addPriceAlert={addPriceAlert} removePriceAlert={removePriceAlert}/>} {section === 'research' && <ResearchPage readOnly={isGuest} tool={researchTool} ticker={researchTicker} setTicker={setResearchTicker} ask={askWithContext} run={runResearch} result={researchResult} busy={researchBusy}/>} {section === 'lab' && <LabPage readOnly={isGuest} tool={labTool} selectTool={setLabTool} ask={askWithContext} askNow={askAndSend}/>} {section === 'cost' && <fieldset className="guest-disabled-page" disabled={isGuest}><CostPage report={costReport} loading={costLoading} error={costError} refresh={refreshCosts}/></fieldset>}</div></section>
      <aside className="chat-panel" aria-label="AI 投资助理">
        <div className="chat-header"><div className="ai-avatar">AI</div><div><strong>{CHAT_LABELS[chatMode]}</strong><span>{isGuest ? '访客模式不执行模型或付费工具' : chatMode === 'agent_v3' ? '公开研究、原文引用与会话追问 · 只读' : '规划并调用研究、账户与量化工具'}</span></div><div className="fact-mode">{isGuest ? '已禁用' : '证据优先'}</div></div>
        <div className="chat-mode-bar">
          <div className="chat-mode-switch" aria-label="聊天模式"><button type="button" className={chatMode === 'agent_v2' ? 'active' : ''} aria-pressed={chatMode === 'agent_v2'} disabled={chatBusy || isGuest} onClick={() => setChatMode('agent_v2')}>Agent V2</button><button type="button" className={chatMode === 'agent_v3' ? 'active' : ''} aria-pressed={chatMode === 'agent_v3'} disabled={chatBusy || isGuest} onClick={() => setChatMode('agent_v3')}>Agent V3</button></div>
          <label className="web-consent" title="仅在内部工具证据不足时允许当前 Agent 尝试网页搜索；服务器也必须启用该能力"><input type="checkbox" checked={!isGuest && allowAgentWeb} disabled={chatBusy || isGuest} onChange={event => setAllowAgentWeb(event.target.checked)}/><span>允许网页兜底</span></label>
        </div>
        <div className="context-bar"><span>当前上下文</span><strong>{chatContext}</strong></div>
        <div className="message-list" ref={messageListRef} aria-live="polite">
          {isGuest && <div className="guest-chat-notice"><strong>AI 功能仅限所有者</strong><span>访客可以浏览已发布的数据快照，但请求不会发送给 Agent V2、Agent V3 或外部数据服务。</span></div>}
          {messages.map(message => <div key={message.id} className={`message ${message.role}`}>
            {message.role === 'user' && message.mode && <div className="message-mode">{CHAT_LABELS[message.mode]}</div>}
            {message.agent && <div className="agent-badges"><span>{message.agent.status}</span>{message.agent.status === 'partial' && <span className="warning">数据或结论仍有缺口</span>}<span>{message.agent.route}</span><span>{message.agent.answerMode}</span>{Boolean(message.evidence?.length) && <span title="仅表示引用和数字可追溯，不代表数据口径一致或原因已确认" className={message.agent.verified ? 'verified' : 'warning'}>{message.agent.verified ? '引用与数字可追溯' : '引用或数字校验有警告'}</span>}{message.agent.synthesis && <span className={message.agent.synthesisFallback ? 'warning' : ''}>{message.agent.synthesis}</span>}{message.agent.rewritten && <span title={message.agent.rewritten}>接上文</span>}{message.agent.webAllowed && <span className="web">Web 已授权</span>}{message.agent.webRequested && !message.agent.webEnabled && <span className="warning">服务端未启用 Web</span>}<span>{(message.agent.elapsedMs / 1000).toFixed(1)}s</span></div>}
            <div className="message-body">{message.role === 'assistant' ? <AgentAnswer text={message.text} evidence={message.evidence} messageId={message.id}/> : message.text}</div>
            {message.image && <Image src={message.image} width={900} height={600} unoptimized alt="AI 查询生成的分析图表"/>}
            {message.agent && message.agent.capabilities.length > 0 && <details className="agent-detail"><summary>执行工具（{message.agent.capabilities.length}）{message.agent.subAgents.length > 0 && ` · 子智能体 ${message.agent.subAgents.length}`}</summary><div className="capability-list">{message.agent.capabilities.map((capability, index) => <code key={`${capability}-${index}`}>{capability}</code>)}</div>{message.agent.subAgents.map((agent, index) => <div key={`${agent.capability}-${index}`} className="sub-agent"><SubAgentTrace label={`${agent.label} ${agent.subject}`} rounds={agent.rounds} elapsedMs={agent.elapsed_ms} stop={agent.stop_reason} calls={agent.calls} trace={agent.trace} intraday={agent.intraday} notes={agent.notes}/>{agent.nested.map((nested, nestedIndex) => <div key={nestedIndex} className="sub-agent nested"><SubAgentTrace label={nested.label} rounds={nested.rounds} elapsedMs={nested.elapsed_ms} stop={nested.stop_reason} calls={nested.calls} trace={nested.trace}/></div>)}</div>)}</details>}
            {message.evidence && message.evidence.length > 0 && <details id={`agent-evidence-${message.id}`} className="agent-detail"><summary>证据（{message.evidence.length}）</summary><ol className="evidence-list">{message.evidence.map((item, index) => <li id={`agent-evidence-${message.id}-${index + 1}`} key={item.id}><div><code>{item.id}</code>{item.entity && <strong>{item.entity}</strong>}</div><p>{item.claim}</p>{item.source_url ? <a href={item.source_url} target="_blank" rel="noreferrer">{item.source_title || item.source_id || '查看来源'}{item.as_of ? ` · ${item.as_of.slice(0, 10)}` : ''}</a> : <small>{item.source_title || item.source_id || '内部计算结果'}{item.as_of ? ` · ${item.as_of.slice(0, 10)}` : ''}</small>}</li>)}</ol></details>}
            {message.agent && message.agent.warnings.length > 0 && <details className="agent-detail warning-detail"><summary>校验警告（{message.agent.warnings.length}）</summary><ul>{message.agent.warnings.map((warning, index) => <li key={`${warning}-${index}`}>{warning}</li>)}</ul></details>}
            {message.meta && <div className="message-meta">{message.meta}</div>}
          </div>)}
          {chatBusy && <div className="message assistant progress-message"><div className="typing"><i/><i/><i/></div><span>{chatProgress || '分析中…'}</span></div>}
        </div>
        <div className="quick-actions">{(chatMode === 'agent_v3' ? ['什么是自由现金流？', '查询 NVDA 最近收盘价和来源', '查阅 NVIDIA 最新 10-K 的供应链风险原文'] : ['比较 NVDA 和 AMD 的风险', '分析 AAPL 的估值', '回测 NVDA 动量策略']).map((suggestion, index) => <button type="button" disabled={isGuest} key={suggestion} onClick={() => setChatInput(suggestion)}>{chatMode === 'agent_v3' ? ['知识问答', '行情来源', '财报原文'][index] : ['风险比较', '估值研究', '策略回测'][index]}</button>)}</div>
        <form className="chat-composer" onSubmit={handleChat}><textarea aria-label="向 AI 提问" value={chatInput} disabled={isGuest} onChange={e => setChatInput(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); void sendMessage(chatInput) } }} placeholder={isGuest ? '访客模式下 AI 与付费工具已禁用' : chatMode === 'agent_v3' ? '让 Agent V3 查询行情、研究或查阅原文…' : '让 Agent V2 研究、比较或运行实验…'} rows={2}/><button type="submit" disabled={isGuest || chatBusy || !chatInput.trim()}>发送</button></form>
      </aside></div>
  </main>;
}


function CorePage({ readOnly, portfolio, risk, tape, history, activity, activityWarning, monitoringUniverse, alertFilter, setAlertFilter, refresh, refreshing, lastUpdated, dataError, chartSnapshotMessage, watchlist, priceAlerts, addMonitoringTicker, removeMonitoringTicker, addWatchlist, removeWatchlist, addPriceAlert, removePriceAlert }: {
  readOnly: boolean; portfolio: PortfolioResponse; risk: RiskResponse; tape: TickerItem[]; history: number[]; activity: ActivityItem[]; activityWarning: string; monitoringUniverse: MonitoringUniverse; alertFilter: 'all' | 'positions' | 'p0'; setAlertFilter: (value: 'all' | 'positions' | 'p0') => void; refresh: () => void; refreshing: boolean; lastUpdated: Date | null; dataError: string; chartSnapshotMessage: string; watchlist: WatchlistItem[]; priceAlerts: PriceAlertItem[]; addMonitoringTicker: (ticker: string) => Promise<void>; removeMonitoringTicker: (ticker: string) => Promise<void>; addWatchlist: (ticker: string) => Promise<void>; removeWatchlist: (ticker: string) => Promise<void>; addPriceAlert: (ticker: string, direction: 'above' | 'below', price: number) => Promise<void>; removePriceAlert: (id: number) => Promise<void>;
}) {
  const [scanTicker, setScanTicker] = useState(''); const [watchTicker, setWatchTicker] = useState(''); const [alertTicker, setAlertTicker] = useState(''); const [alertDirection, setAlertDirection] = useState<'above' | 'below'>('above'); const [alertPrice, setAlertPrice] = useState(''); const [scanError, setScanError] = useState(''); const [watchError, setWatchError] = useState(''); const [alertError, setAlertError] = useState('');
  const [positionChart, setPositionChart] = useState<PositionChartState | null>(null);
  const [positionChartView, setPositionChartView] = useState<PriceChartView>('line');
  const [showExtendedHours, setShowExtendedHours] = useState(false);
  const [chartLayers, setChartLayers] = useState<ChartDisplaySettings>(CHART_PRESETS.standard);
  const [chartPreset, setChartPreset] = useState<ActiveChartPreset>('standard');
  useEffect(() => {
    const restore = window.setTimeout(() => {
      try {
        const saved = JSON.parse(localStorage.getItem(CHART_DISPLAY_STORAGE_KEY) || 'null') as Partial<ChartPreference> | Partial<ChartDisplaySettings> | null;
        if (saved) {
          const savedLayers = 'layers' in saved && saved.layers && typeof saved.layers === 'object' ? saved.layers : saved;
          const restoredLayers = { ...CHART_PRESETS.standard, ...Object.fromEntries(Object.entries(savedLayers).filter(([, value]) => typeof value === 'boolean')) } as ChartDisplaySettings;
          const savedPreset = 'preset' in saved && ['simple', 'standard', 'professional', 'custom'].includes(String(saved.preset)) ? saved.preset as ActiveChartPreset : 'custom';
          setChartLayers(restoredLayers);
          setChartPreset(savedPreset);
        }
      } catch {
        localStorage.removeItem(CHART_DISPLAY_STORAGE_KEY);
      }
    }, 0);
    return () => window.clearTimeout(restore);
  }, []);
  useEffect(() => {
    if (!positionChart) return;
    const close = (event: KeyboardEvent) => { if (event.key === 'Escape') setPositionChart(null) };
    document.addEventListener('keydown', close);
    return () => document.removeEventListener('keydown', close);
  }, [positionChart]);
  const positionSymbols = new Set(portfolio.positions.map(item => item.symbol));
  const toAlertRow = (item: ActivityItem) => { const plain = htmlToText(item.preview); const lines = plain.split('\n').map(line => line.trim()).filter(line => line && !/^━+$/.test(line)); const firstLine = lines[0] || ''; const titleTicker = (item.title || '').match(/\b[A-Z][A-Z0-9.-]{0,9}\b/)?.[0]; const ticker = titleTicker || (item.tickers || '').split(',')[0].trim() || item.agent.toUpperCase().slice(0, 6); const level = item.priority_tier || 'P1'; const isAnomaly = item.agent.toLowerCase() !== 'alert'; const label = firstLine.includes('盘中异动') || (item.title || '').includes('盘中异动') ? '盘中异动' : firstLine.includes('异动') || (item.title || '').includes('异动') ? '异动' : (item.title || item.msg_type).replace(new RegExp(`\\s*[·・:-]\\s*${ticker}\\b`, 'i'), ''); const embeddedTime = firstLine.match(/\b\d{1,2}:\d{2}\s*ET\b/i)?.[0].toUpperCase(); const embeddedDate = firstLine.match(/\b(?:20\d{2}-)?(\d{2})-(\d{2})\b/); const monthDay = embeddedDate ? `${embeddedDate[1]}-${embeddedDate[2]}` : formatActivityMonthDayET(item.ts); const time = embeddedTime || formatActivityTimeET(item.ts); const detailLines = lines.slice(1).filter(line => !line.startsWith('用 /') && !line.startsWith('用 /why') && !line.startsWith('或 /summary')); const reason = (detailLines.slice(0, 2).join(' · ') || (isAnomaly ? '暂无更多异动明细' : '暂无更多提醒明细')).slice(0, 190); return { id: item.id, level, ticker, title: `${isAnomaly ? label : '价格提醒'} · ${monthDay} ${time}`, reason, time, tone: level === 'P0' ? 'danger' : level === 'P1' ? 'warning' : 'neutral' } };
  const isPriceAlertActivity = (item: ActivityItem) => item.agent.toLowerCase() === 'alert' || item.msg_type === 'alert_fire' || (item.title || '').startsWith('价格提醒');
  const anomalyActivity = activity.filter(item => !isPriceAlertActivity(item));
  const triggeredPriceAlerts = activity.filter(isPriceAlertActivity).map(toAlertRow);
  const archiveAlerts = anomalyActivity.map(toAlertRow);
  const displayed = archiveAlerts.filter(item => alertFilter === 'p0' ? item.level === 'P0' : alertFilter === 'positions' ? positionSymbols.has(item.ticker) : true).slice(0, 8);
  const alertTickers = [...new Set(priceAlerts.map(item => item.ticker))].sort();
  const watchTickers = [...new Set(watchlist.map(item => item.ticker))].sort();
  const monitoredCount = new Set([...monitoringUniverse.intraday, ...alertTickers, ...watchTickers]).size;
  const monitoringGroups = [
    { key: 'intraday', label: '盘中异动扫描', note: `每 ${monitoringUniverse.scan_interval_seconds} 秒 · 涨跌幅 ≥ ${(monitoringUniverse.price_pct_threshold * 100).toFixed(0)}% 且量能 ≥ ${monitoringUniverse.volume_pace_threshold.toFixed(1)}×`, items: monitoringUniverse.intraday },
    { key: 'alerts', label: '价格提醒', note: '逐分钟检查尚未触发的价格条件', items: alertTickers },
    { key: 'watchlist', label: 'Watchlist', note: '用于财报、事件和研究任务，不等同于盘中异动池', items: watchTickers },
  ];
  const activityStatus = activityWarning === 'archive.db not found' ? '实时监控尚未启动；启动 streamer 后，真实异常会自动出现在这里。' : activityWarning;
  const firstEquity = history[0] || portfolio.account.portfolio_value; const lastEquity = history.at(-1) || portfolio.account.portfolio_value; const historyReturn = firstEquity ? (lastEquity - firstEquity) / firstEquity : 0;
  const submitScanTicker = async () => { try { setScanError(''); await addMonitoringTicker(scanTicker); setScanTicker('') } catch (error) { setScanError(error instanceof Error ? error.message : '添加失败') } };
  const removeScanTicker = async (ticker: string) => { try { setScanError(''); await removeMonitoringTicker(ticker) } catch (error) { setScanError(error instanceof Error ? error.message : '移除失败') } };
  const submitWatch = async () => { try { setWatchError(''); await addWatchlist(watchTicker); setWatchTicker('') } catch (error) { setWatchError(error instanceof Error ? error.message : '添加失败') } };
  const submitAlert = async () => { try { setAlertError(''); await addPriceAlert(alertTicker, alertDirection, Number(alertPrice)); setAlertTicker(''); setAlertPrice('') } catch (error) { setAlertError(error instanceof Error ? error.message : '提醒创建失败') } };
  const fetchPositionChart = async (symbol: string, portfolioPrice: number, range: PriceRange, extended = false) => {
    setPositionChart({ symbol, portfolioPrice, range, points: [], quote: null, analysis: null, source: '', adjustmentMode: '', warmupBars: 0, includesExtendedHours: false, loading: true, error: '' });
    try {
      const result = await apiJson<PriceHistoryResponse>(`/api/price-history/${encodeURIComponent(symbol)}?range=${range}&extended=${extended}`);
      setPositionChart(current => current?.symbol === symbol && current.range === range ? { ...current, points: result.bars, quote: result.quote, analysis: result.technicalAnalysis, source: result.source, adjustmentMode: result.adjustmentMode, warmupBars: result.warmupBars, includesExtendedHours: result.includesExtendedHours, loading: false } : current);
    } catch (error) {
      const detail = error instanceof Error ? error.message : '未知错误';
      setPositionChart(current => current?.symbol === symbol && current.range === range ? { ...current, loading: false, error: `价格曲线加载失败：${detail}` } : current);
    }
  };
  const openPositionChart = (position: Position) => { setPositionChartView('line'); setShowExtendedHours(false); void fetchPositionChart(position.symbol, position.current_price, '1M') };
  const changePositionChartRange = (range: PriceRange) => {
    if (!positionChart || positionChart.range === range) return;
    const extended = showExtendedHours && (range === '1W' || range === '1M');
    void fetchPositionChart(positionChart.symbol, positionChart.portfolioPrice, range, extended);
  };
  const toggleExtendedHours = () => {
    if (!positionChart || (positionChart.range !== '1W' && positionChart.range !== '1M')) return;
    const next = !showExtendedHours;
    setShowExtendedHours(next);
    void fetchPositionChart(positionChart.symbol, positionChart.portfolioPrice, positionChart.range, next);
  };
  const saveChartPreference = (preference: ChartPreference) => { setChartLayers(preference.layers); setChartPreset(preference.preset); localStorage.setItem(CHART_DISPLAY_STORAGE_KEY, JSON.stringify(preference)) };
  const changeChartLayer = (key: ChartLayerKey, enabled: boolean) => saveChartPreference({ preset: 'custom', layers: { ...chartLayers, [key]: enabled } });
  const applyChartPreset = (preset: ChartPreset) => saveChartPreference({ preset, layers: { ...CHART_PRESETS[preset] } });
  const selectedPriceRange = positionChart ? PRICE_RANGES.find(item => item.key === positionChart.range) : undefined;
  const selectedQuoteChange = positionChart?.quote ? quoteChangePct(positionChart.quote) : null;
  const selectedDisplayPrice = positionChart?.quote?.session === 'CLOSED' && positionChart.quote.regularClose != null ? positionChart.quote.regularClose : positionChart?.quote?.price ?? positionChart?.portfolioPrice;
  return <div className="page core-page"><TickerTape items={tape}/><div className="page-heading"><div><h1>盯盘总览</h1></div><div className="refresh-controls"><span aria-live="polite">{refreshing ? '正在同步各数据源' : chartSnapshotMessage || (lastUpdated ? `${readOnly ? '快照发布' : '最后更新'} ${lastUpdated.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', second: '2-digit' })}` : '尚未更新')}</span><button className="quiet-button" onClick={refresh} disabled={refreshing || readOnly} aria-busy={refreshing} title={readOnly ? '访客模式只读取最后发布的快照' : '同步页面数据，并在后台更新全部持仓的常用周期图表快照'}>{refreshing && <i className="refresh-spinner" aria-hidden="true"/>}{readOnly ? '只读快照' : refreshing ? '刷新中…' : '刷新数据'}</button></div></div>{dataError && <div className="data-banner" role="status">{dataError}</div>}<div className="metric-grid"><Metric label="组合净值" value={formatMoney(portfolio.account.portfolio_value)} note={portfolio.account.paper ? 'Paper account' : 'Live account'}/><Metric label="组合当日" value={formatPct(risk.pnl.daily_pnl_pct ?? portfolio.pnl.intraday_pl_pct)} tone={(risk.pnl.daily_pnl_pct ?? portfolio.pnl.intraday_pl_pct) >= 0 ? 'positive' : 'negative'} note="较昨日收盘"/><Metric label="当前回撤" value={formatPct(risk.drawdown.current_drawdown_pct == null ? null : -Math.abs(risk.drawdown.current_drawdown_pct))} tone="negative" note="距近期高点"/><Metric label="可用现金" value={formatMoney(portfolio.account.cash)} note={`${((portfolio.account.cash / portfolio.account.portfolio_value) * 100).toFixed(1)}% 现金仓位`}/></div>
    <div className="core-grid"><section className="surface equity-card"><div className="surface-header"><div><h2>组合净值</h2><span>近 1 个月 · {history.length} 个数据点</span></div><strong className={historyReturn >= 0 ? 'positive' : 'negative'}>{formatPct(historyReturn)}</strong></div><MiniEquityChart values={history}/><div className="chart-footer"><span>{formatMoney(firstEquity)}</span><span>{formatMoney(lastEquity)}</span></div></section>
      <section className="surface alerts-card"><div className="surface-header"><div><h2>实时异常</h2><span>{activityStatus ? '监控数据源未就绪' : `最近两个交易日 · ${anomalyActivity.length} 条真实异常`}</span></div><div className="segmented">{([['all','全部'],['positions','持仓'],['p0','P0']] as const).map(([id,label]) => <button key={id} className={alertFilter === id ? 'active' : ''} onClick={() => setAlertFilter(id)}>{label}</button>)}</div></div><div className="alert-list">{displayed.length ? displayed.map(item => <div className="alert-row" key={item.id}><span className={`priority ${item.tone}`}>{item.level}</span><div className="alert-symbol">{item.ticker}{positionSymbols.has(item.ticker) && <small className="holding-badge">持仓</small>}</div><div className="alert-copy"><strong>{item.title}</strong><span>{item.reason}</span></div></div>) : <div className="empty-row real-feed-empty"><strong>{activityStatus ? '等待实时监控数据' : alertFilter === 'all' ? '当前没有符合条件的异常' : '当前筛选下没有异常'}</strong><span>{activityStatus || '价格提醒已独立显示在右侧；点击“刷新数据”可获取最新异常。'}</span></div>}</div></section>
      <section className="surface price-alert-card"><div className="surface-header"><div><h2>价格提醒</h2><span>{activityStatus ? '监控数据源未就绪' : `最近两个交易日 · ${triggeredPriceAlerts.length} 条已触发`}</span></div></div><div className="price-alert-list">{triggeredPriceAlerts.length ? triggeredPriceAlerts.slice(0, 8).map(item => <div className="alert-row price-alert-row" key={item.id}><div className="alert-symbol">{item.ticker}</div><div className="alert-copy"><strong>{item.title}</strong><span>{item.reason}</span></div></div>) : <div className="empty-row real-feed-empty"><strong>{activityStatus ? '等待价格提醒数据' : '最近两个交易日没有触发价格提醒'}</strong><span>{activityStatus || '下方可以创建新的价格条件；触发后会显示在这里。'}</span></div>}</div></section>
      <section className="surface portfolio-risk"><div className="surface-header"><div><h2>组合风险</h2><span>实时持仓加权</span></div></div><div className="risk-items"><RiskItem label="最大持仓" value={`${portfolio.positions[0]?.symbol || '—'} ${(risk.concentration.top_1_pct * 100).toFixed(1)}%`} fill={risk.concentration.top_1_pct * 100}/><RiskItem label="前三持仓" value={`${(risk.concentration.top_3_pct * 100).toFixed(1)}%`} fill={risk.concentration.top_3_pct * 100}/><RiskItem label="行业暴露" value={`${risk.exposure.largest_sector} ${(risk.exposure.largest_sector_pct * 100).toFixed(1)}%`} fill={risk.exposure.largest_sector_pct * 100}/><div className="earnings-note"><span>未来 7 天财报</span><strong>{risk.earnings_risk.length} 个持仓</strong><small>{risk.earnings_risk.map(item => `${item.ticker} D-${item.days_until}`).join(' · ') || '暂无'}</small></div></div></section>
      <section className="surface positions-card"><div className="surface-header"><div><h2>重点持仓</h2><span>{portfolio.positions.length} 个仓位</span></div></div><div className="positions-table"><div className="position-table-row position-table-head"><span>代码</span><span>市值</span><span>浮动盈亏</span><span>收益率</span><span>最新价</span></div>{portfolio.positions.slice(0, 8).map(position => <div className="position-table-row" key={position.symbol}><button className="position-row-main" onClick={() => void openPositionChart(position)} aria-label={`查看 ${position.symbol} 价格波动曲线`}><strong>{position.symbol}</strong><span>{formatMoney(position.market_value)}</span><span className={position.unrealized_pl >= 0 ? 'positive' : 'negative'}>{formatMoney(position.unrealized_pl)}</span><span className={position.unrealized_pl_pct >= 0 ? 'positive' : 'negative'}>{formatPct(position.unrealized_pl_pct)}</span><span>{formatUnitPrice(position.current_price)}</span></button></div>)}</div></section>
      <section className="surface monitored-universe-card"><div className="surface-header"><div><h2>当前监控股票</h2><span>{monitoredCount} 只去重标的 · 随“刷新数据”更新</span></div><span className="universe-source">{monitoringUniverse.source}</span></div><div className="monitoring-groups">{monitoringGroups.map(group => <div className="monitoring-group" key={group.key}><div className="monitoring-group-heading"><div><strong>{group.label}</strong><span>{group.note}</span></div><em>{group.items.length}</em></div><div className="monitoring-tickers">{group.items.length ? group.items.map(ticker => <span key={ticker}>{ticker}</span>) : <small>暂无标的</small>}</div></div>)}</div></section>
      <section className="surface monitor-card"><div className="surface-header"><div><h2>监控管理</h2><span>{readOnly ? '访客模式下管理操作已禁用' : '管理盘中扫描、关注列表与价格提醒'}</span></div>{readOnly && <span className="read-only-badge">只读</span>}</div><fieldset className="monitor-columns guest-disabled-page" disabled={readOnly}><div><h3>盘中异动扫描</h3>{scanError && <div className="column-error">{scanError}</div>}<div className="inline-form"><input value={scanTicker} onChange={e => setScanTicker(e.target.value.toUpperCase().replace(/[^A-Z]/g, '').slice(0, 5))} placeholder="AAPL" aria-label="盘中异动扫描股票代码"/><button onClick={() => void submitScanTicker()} disabled={!scanTicker.trim()}>添加</button></div><div className="tag-list scan-tag-list">{monitoringUniverse.intraday.length ? monitoringUniverse.intraday.map(ticker => <span key={ticker}>{ticker}<button aria-label={`从盘中异动扫描移除 ${ticker}`} onClick={() => void removeScanTicker(ticker)}>×</button></span>) : <small>盘中异动扫描池为空</small>}</div></div><div><h3>关注列表</h3>{watchError && <div className="column-error">{watchError}</div>}<div className="inline-form"><input value={watchTicker} onChange={e => setWatchTicker(e.target.value.toUpperCase())} placeholder="AAPL" aria-label="关注列表股票代码"/><button onClick={() => void submitWatch()} disabled={!watchTicker.trim()}>添加</button></div><div className="tag-list">{watchlist.length ? watchlist.map(item => <span key={item.ticker}>{item.ticker}<button aria-label={`移除 ${item.ticker}`} onClick={() => void removeWatchlist(item.ticker)}>×</button></span>) : <small>关注列表为空</small>}</div></div><div><h3>创建价格提醒</h3>{alertError && <div className="column-error">{alertError}</div>}<div className="alert-form"><input value={alertTicker} onChange={e => setAlertTicker(e.target.value.toUpperCase())} placeholder="NVDA" aria-label="提醒股票代码"/><select value={alertDirection} onChange={e => setAlertDirection(e.target.value as 'above' | 'below')} aria-label="提醒方向"><option value="above">高于</option><option value="below">低于</option></select><input type="number" min="0.01" step="0.01" value={alertPrice} onChange={e => setAlertPrice(e.target.value)} placeholder="价格" aria-label="提醒价格"/><button onClick={() => void submitAlert()} disabled={!alertTicker || !Number(alertPrice)}>创建</button></div><div className="alert-tags">{priceAlerts.length ? priceAlerts.map(item => <span key={item.id}>{item.ticker} {item.direction === 'above' ? '≥' : '≤'} ${item.target_price}<button aria-label={`删除 ${item.ticker} 提醒`} onClick={() => void removePriceAlert(item.id)}>×</button></span>) : <small>暂无未触发提醒</small>}</div></div></fieldset></section>
    </div>{positionChart && <div className="position-chart-backdrop" role="presentation" onMouseDown={() => setPositionChart(null)}><section className="position-chart-modal" role="dialog" aria-modal="true" aria-labelledby="position-chart-title" onMouseDown={event => event.stopPropagation()}>
      <header><div className="price-chart-heading"><h2 id="position-chart-title">{positionChart.symbol} 价格波动</h2><div className="chart-current-price"><strong>{formatUnitPrice(selectedDisplayPrice)}</strong>{positionChart.quote && <><em className={`market-session ${positionChart.quote.session.toLowerCase()}`}>{sessionLabel(positionChart.quote.session)}</em>{selectedQuoteChange != null && <span className={selectedQuoteChange >= 0 ? 'positive' : 'negative'}>{formatPct(selectedQuoteChange)}</span>}</>}</div><div className="quote-metadata"><span>{positionChart.quote?.session === 'CLOSED' ? '正式收盘价' : '最新价'}</span>{positionChart.quote?.regularClose != null && positionChart.quote.session !== 'CLOSED' && <span>正式收盘 {formatUnitPrice(positionChart.quote.regularClose)}</span>}{positionChart.quote && (positionChart.quote.session === 'PRE_MARKET' || positionChart.quote.session === 'AFTER_HOURS') && <span>当日 {formatPct(positionChart.quote.dayReturn)}</span>}{positionChart.quote?.timestamp && <span>报价 {formatPricePointTime(positionChart.quote.timestamp, selectedPriceRange?.interval || '1d')}</span>}<span>{selectedPriceRange?.intervalLabel || '日线'}</span>{positionChart.source && <span>{positionChart.source.replaceAll('_', ' ')}</span>}{positionChart.quote?.isDelayed && <span>延迟行情</span>}</div></div><button type="button" aria-label="关闭价格图表" onClick={() => setPositionChart(null)}>×</button></header>
      <div className="position-chart-toolbar"><nav className="position-chart-ranges" aria-label="价格图表时间范围">{PRICE_RANGES.map(item => <button type="button" key={item.key} className={positionChart.range === item.key ? 'active' : ''} aria-pressed={positionChart.range === item.key} onClick={() => changePositionChartRange(item.key)}>{item.label}</button>)}</nav><div className="chart-toolbar-actions"><label className={`extended-hours-toggle ${readOnly || (positionChart.range !== '1W' && positionChart.range !== '1M') ? 'disabled' : ''}`} title={readOnly ? '访客图表使用最后发布的正式交易时段快照' : '美股盘前和盘后K线，仅适用于30分钟及1小时周期'}><input type="checkbox" checked={showExtendedHours} disabled={readOnly || positionChart.loading || (positionChart.range !== '1W' && positionChart.range !== '1M')} onChange={toggleExtendedHours}/><span>盘前盘后</span></label><div className="position-chart-view-toggle" aria-label="图表类型">{([['line', '走势'], ['candles', 'K线']] as const).map(([view, label]) => <button type="button" key={view} className={positionChartView === view ? 'active' : ''} aria-pressed={positionChartView === view} onClick={() => setPositionChartView(view)}>{label}</button>)}</div></div></div>
      <div className="position-chart-body">{positionChart.loading ? <div className="position-chart-loading"><i className="refresh-spinner" aria-hidden="true"/>正在加载并计算技术指标…</div> : positionChart.error ? <div className="position-chart-error">{positionChart.error}</div> : positionChartView === 'candles' && positionChart.analysis && positionChart.quote ? <CandlestickChart symbol={positionChart.symbol} rangeLabel={selectedPriceRange?.label || positionChart.range} interval={selectedPriceRange?.interval || '1d'} points={positionChart.points} analysis={positionChart.analysis} quote={positionChart.quote} layers={chartLayers} activePreset={chartPreset} showExtendedHours={showExtendedHours} onLayerChange={changeChartLayer} onPreset={applyChartPreset}/> : <StockPriceChart symbol={positionChart.symbol} rangeLabel={selectedPriceRange?.label || positionChart.range} interval={selectedPriceRange?.interval || '1d'} points={positionChart.points}/>}</div>
      <footer><span>{positionChart.adjustmentMode ? adjustmentLabel(positionChart.adjustmentMode) : '统一复权价格'} · {positionChart.warmupBars} 根预热K线 · SMA按 {selectedPriceRange?.interval || '1d'} 周期计算</span><span>{positionChartView === 'candles' ? '滚轮缩放 · 拖动浏览 · 悬停查看完整指标' : '点击窗口外或按 Esc 关闭'}</span></footer>
    </section></div>}</div>;
}
function Metric({ label, value, note, tone = '' }: { label: string; value: string; note: string; tone?: string }) { return <div className="metric-card"><span>{label}</span><strong className={tone}>{value}</strong><small>{note}</small></div> }
function RiskItem({ label, value, fill }: { label: string; value: string; fill: number }) { return <div className="risk-item"><div><span>{label}</span><strong>{value}</strong></div><div className="risk-track"><i style={{ width: `${Math.min(fill, 100)}%` }}/></div></div> }

function ResearchPage({ readOnly, tool, ticker, setTicker, ask, run, result, busy }: { readOnly: boolean; tool: ResearchTool; ticker: string; setTicker: (value: string) => void; ask: (prompt: string, context: string) => void; run: ResearchRun; result: ResearchResultState | null; busy: boolean }) {
  const labels: Record<ResearchTool, [string, string]> = { stock: ['个股工作台', '统一运行研究模块并汇总投资论点'], fundamentals: ['公司与基本面', '理解业务、增长、盈利质量与财务健康'], valuation: ['估值与同业', '比较当前估值、历史位置与同行候选'], earnings: ['财报与 SEC', '查看财报预期、公告摘要和监管文件'], expectations: ['预期与催化剂', '跟踪一致预期、盈利动量与未来事件'], institutional: ['机构与内部人', '追踪 13F、机构持仓和内部人变化'], moneyflow: ['资金流', '观察成交量、CMF、OBV 与资金累积派发'], macro: ['宏观环境', '观察利率、波动率和关键经济数据'], chain: ['产业链', '发现供应商、客户与竞争关系'], risk: ['风险雷达', '用证据归纳估值、财务、供应链与治理风险'] };
  const resultTicker = ticker.trim().toUpperCase();
  const engine = result?.ticker === resultTicker ? result.engine : null;
  const progress = result?.ticker === resultTicker ? result.progress || {} : {};
  return <div className="page research-page"><div className="page-heading"><div><div className="research-title"><h1>{labels[tool][0]}</h1>{!readOnly && <ResearchHelp key={tool} tool={tool}/>}</div><p>{labels[tool][1]}</p></div><span className="on-demand-tag">{readOnly ? 'NVDA 只读快照' : '研究引擎'}</span></div>{readOnly && <div className="readonly-page-note">访客仅可查看所有者最后发布的 NVDA 研究结果，搜索、刷新、重试和关系维护均已禁用。</div>}<fieldset className="guest-disabled-page" disabled={readOnly}><div className="research-search"><label><span>{tool === 'macro' ? '关联股票代码' : '股票代码'}</span><input value={readOnly ? 'NVDA' : ticker} onChange={e => setTicker(e.target.value.toUpperCase().replace(/[^A-Z0-9.-]/g, '').slice(0, 8))}/></label><button disabled={readOnly || busy || !ticker} onClick={() => void run(tool, ticker)}>{readOnly ? '访客只读' : busy ? '研究中…' : engine ? '刷新当前模块' : '开始研究'}</button></div><StockResearchView tool={tool} ticker={ticker} result={engine || null} progress={progress} busy={busy} run={run}/></fieldset></div>;
}
function ResearchResultView({ result }: { result: ToolResult }) { return <section className="surface research-result"><div className="surface-header"><div><h2>最新研究结果</h2><span>{result.intent ? `识别为 ${result.intent}` : '来自现有研究管线'}</span></div></div><pre>{result.text}</pre>{result.image && <Image src={result.image} width={1000} height={700} unoptimized alt="研究结果图表"/>}</section> }
function formatResearchValue(value: unknown, kind = 'number') { if (value == null || value === '') return 'N/A'; if (typeof value !== 'number') return String(value); if (!Number.isFinite(value)) return 'N/A'; if (kind === 'pct') return `${value >= 0 ? '+' : ''}${(value * 100).toFixed(1)}%`; if (kind === 'money') return Math.abs(value) >= 1e9 ? `$${(value / 1e9).toFixed(2)}B` : Math.abs(value) >= 1e6 ? `$${(value / 1e6).toFixed(1)}M` : `$${value.toLocaleString()}`; return value.toFixed(2) }
const MODULE_LABELS: Record<string,string> = { data: '数据采集', fundamental: '基本面', growth: '成长性', profitability: '盈利能力', financial_health: '财务健康', valuation: '估值', earnings: '财报', earnings_momentum: '盈利动量', expectations: '市场预期', institutional: '机构与内部人', fund_flow: '资金流', technical: '技术面', catalyst: '催化剂', sec: 'SEC', macro: '宏观', supply_chain: '产业链', risk: '风险' };
const RESEARCH_TERM_MAP: Record<string,string> = {
  'Revenue growth': '营收增长', 'Profitability and capital returns': '盈利能力与资本回报', 'Latest earnings surprise': '最新盈利表现',
  'Technical regime': '技术趋势', 'Absolute valuation': '绝对估值', 'Reported insider activity': '已披露内部人交易',
  'SEC risk-factor changes': 'SEC 风险因素变化', 'Net Charge Off Rate': '净核销率', 'Roe': 'ROE', 'Cet1': 'CET1',
  'VALUATION risk': '估值风险', 'FINANCIAL risk': '财务风险', 'SUPPLY_CHAIN risk': '供应链风险', 'EVENT risk': '事件风险',
  'MANAGEMENT risk': '管理层风险', 'CUSTOMER_CONCENTRATION risk': '客户集中风险', 'GOVERNANCE risk': '治理风险',
  'reported revenue growth and management guidance': '已披露营收增长与管理层指引', 'gross margin and ROIC': '毛利率与 ROIC',
  'earnings surprise and guidance': '盈利表现与指引', 'daily trend and key levels': '日线趋势与关键价位',
  'peer-relative multiples and realized growth': '同业相对估值与实际增长', 'the observable evidence supporting this driver': '支持该驱动因素的可观察证据',
  'Historical expectations are unavailable; no revision trend was inferred.': '缺少历史一致预期数据，因此未推断预期修正趋势。',
  'current consensus': '当前一致预期', 'earnings surprise history': '历史财报超预期表现', 'revision 30d': '30 日预期修正', 'revision 60d': '60 日预期修正', 'revision 90d': '90 日预期修正',
  'Company': '公司', 'Ticker': '股票代码', 'Sector': '板块', 'Industry': '行业', 'Market Cap': '总市值', 'Exchange': '交易所',
  'Revenue YoY': '营收同比', 'EPS Growth': 'EPS 增长', 'Gross Margin': '毛利率', 'Operating Margin': '营业利润率', 'Net Margin': '净利润率',
  'Balance Sheet': '资产负债表', 'Debt Risk': '债务风险', 'Cash Generation': '现金创造能力', 'Current Ratio': '流动比率', 'Interest Coverage': '利息保障倍数', 'Debt / Equity': '负债权益比',
  'Shares Outstanding': '流通股数', 'Share Count Change': '股本变化', 'Stock-Based Compensation': '股权激励', 'Buyback': '股票回购', 'Dividend': '股息',
  'EPS Consensus': 'EPS 一致预期', 'Revenue Consensus': '营收一致预期', 'Revision Trend': '预期修正趋势', 'Next Earnings': '下次财报', 'Release Window': '发布时间段',
  'Momentum Score': '动量评分', 'Estimate Revision': '盈利预期修正', 'Confidence': '置信度', 'Evidence': '证据',
  'Volume': '成交量', 'Relative Volume': '相对成交量', 'Accumulation / Distribution': '累积/派发', 'Institutional Flow': '机构资金流', 'ETF Flow': 'ETF 资金流',
  'Insufficient Evidence': '证据不足', 'GENERAL': '通用行业', 'TECHNOLOGY': '科技', 'FINANCIALS': '金融', 'ENERGY': '能源', 'AUTOMOTIVE': '汽车',
};
function translateResearchText(value: unknown) {
  if (value == null) return '';
  let text = String(value);
  const exact = RESEARCH_TERM_MAP[text];
  if (exact) return exact;
  Object.entries(RESEARCH_TERM_MAP).sort(([a],[b]) => b.length - a.length).forEach(([source,target]) => { text = text.replaceAll(source, target) });
  return text.replace(/\bversus\b/gi, '与').replace(/\bavailable\b/gi, '可用').replace(/\brisk\b/gi, '风险');
}
const STATUS_LABELS: Record<string,string> = { COMPLETED: '已完成', PARTIAL: '部分完成', PARTIAL_DATA: '部分数据', PARTIAL_ERROR: '部分异常', RUNNING: '运行中', PENDING: '等待中', FAILED: '失败', PASSED: '已通过', FULL: '完整支持', STANDARD: '标准支持', LIMITED: '有限支持', INSUFFICIENT: '证据不足', HEALTHY: '正常', CACHED: '已缓存', UNCACHED: '未缓存', UNAVAILABLE: '不可用', AVAILABLE: '可用', AVAILABLE_WHEN_PROVIDER_RETURNS: '数据源返回时可用' };
const LEVEL_LABELS: Record<string,string> = { LOW: '低', MEDIUM: '中等', HIGH: '高', CRITICAL: '关键', UNKNOWN: '未知' };
function statusLabel(value: string) { return STATUS_LABELS[value.toUpperCase()] || translateResearchText(value) }
function levelLabel(value: string | undefined) { return value ? LEVEL_LABELS[value.toUpperCase()] || translateResearchText(value) : '' }
function financialHealthLabel(value: unknown) { const labels: Record<string,string> = { STRONG: '强', LOW: '低', MODERATE: '中等', MEDIUM: '中等', HEALTHY: '健康', WEAK: '弱', HIGH: '高', UNKNOWN: '未知', INSUFFICIENT: '证据不足' }; const key = String(value ?? '').trim().toUpperCase(); return labels[key] || translateResearchText(value) || 'N/A' }
function chineseCompanyDescription(company: Record<string,unknown>, ticker: string) { const raw = String(company.description || '').trim(); if (/[㐀-鿿]/.test(raw)) return raw; const name = String(company.name || ticker); const sector = translateResearchText(company.sector || '未分类板块'); const industry = translateResearchText(company.industry || '未分类行业'); const exchange = String(company.exchange || '').trim(); return `${name}（${ticker}）是一家属于${sector}板块、${industry}行业的公司${exchange ? `，股票在 ${exchange} 交易` : ''}。业务详情以公司最新披露文件为准。` }
function chineseSecRiskTitle(value: unknown) { return translateResearchText(value) || '风险因素' }
function chineseSecRiskText(value: unknown, _title: unknown, _changeType: unknown) { return String(value || '暂无风险说明') }
function ResearchProgress({ progress }: { progress: Record<string,string> }) { return <section className="surface engine-progress"><div className="pipeline-spinner"/><h2>正在构建股票研究数据集</h2><div>{Object.entries(progress).map(([name,status]) => <span key={name} className={status.toLowerCase()}><i>{status === 'COMPLETED' ? '✓' : status === 'RUNNING' ? '→' : status === 'FAILED' ? '×' : '○'}</i>{MODULE_LABELS[name] || name}</span>)}</div></section> }
function EmptyResearch({ ticker }: { ticker: string }) { return <section className="surface research-engine-empty"><span>⌁</span><h2>开始研究 {ticker || '一只股票'}</h2><p>系统会共享同一份标准化数据，分别计算基本面、估值、预期、资金流、风险与投资论点。结果不会写入右侧对话。</p></section> }
function ScoreCards({ scores }: { scores: Record<string,number | null> }) { const order = ['fundamental','growth','profitability','financial_health','valuation','earnings_momentum','institutional','technical','catalyst']; return <div className="engine-score-grid">{order.map(key => <div className="surface" key={key}><span>{MODULE_LABELS[key] || translateResearchText(key.replaceAll('_',' '))}</span><strong>{scores[key] == null ? 'N/A' : `${scores[key]} / 100`}</strong><i style={{ width: `${scores[key] || 0}%` }}/></div>)}</div> }
function StatusPill({ value }: { value: string }) { return <span className={`engine-status ${value.toLowerCase()}`}>{statusLabel(value)}</span> }
function MetricRows({ rows }: { rows: [string, unknown, string?][] }) { const healthLabels = new Set(['资产负债表','债务风险','现金创造能力']); return <div className="engine-metric-rows">{rows.map(([label,value,kind]) => <div key={label}><span>{translateResearchText(label)}</span><strong>{healthLabels.has(label) ? financialHealthLabel(value) : translateResearchText(formatResearchValue(value,kind))}</strong></div>)}</div> }
function FinancialHealthRows({ details, metrics }: { details: Record<string,unknown>; metrics: Record<string,unknown> }) { return <div className="engine-metric-rows"><div><span>资产负债表</span><strong>{financialHealthLabel(details.balance_sheet)}</strong></div><div><span>债务风险</span><strong>{financialHealthLabel(details.debt_risk)}</strong></div><div><span>现金创造能力</span><strong>{financialHealthLabel(details.cash_generation)}</strong></div><div><span>流动比率</span><strong>{formatResearchValue(metrics.current_ratio)}</strong></div><div><span>利息保障倍数</span><strong>{formatResearchValue(metrics.interest_coverage)}</strong></div><div><span>负债权益比</span><strong>{formatResearchValue(metrics.debt_to_equity)}</strong></div></div> }
function SourceFooter({ result, ids }: { result: StockResearchResult; ids?: string[] }) { const sources = ids?.length ? result.sources.filter(source => ids.includes(source.id)) : result.sources; return <div className="engine-sources"><strong>来源</strong>{sources.length ? sources.slice(0,6).map(source => source.url ? <a key={source.id} href={source.url} target="_blank" rel="noreferrer">{source.provider} · {source.type}</a> : <span key={source.id}>{source.provider} · {source.type}</span>) : <span>暂无可追踪来源</span>}</div> }
function FundamentalView({ result }: { result: StockResearchResult }) { const module = result.modules.fundamental; const details = module.details as { company?: Record<string,unknown>; quarterly_trends?: Record<string,unknown>[]; growth_trend?: string; financial_health?: Record<string,unknown>; shareholder_value?: Record<string,unknown> }; const company = details.company || {}; const quarters = details.quarterly_trends || []; return <><div className="engine-two-col"><section className="surface engine-card"><div className="surface-header"><div><h2>公司概况</h2><span>Financial Datasets 公司事实数据</span></div><StatusPill value={module.status}/></div><MetricRows rows={[["公司",company.name],["股票代码",result.ticker],["板块",company.sector],["行业",company.industry],["总市值",module.metrics.market_cap,'money'],["交易所",company.exchange]]}/><p className="insufficient-note">{String(company.description || '暂无可靠公司简介、业务分部和营收构成数据，因此不生成占位图。')}</p></section><section className="surface engine-card"><div className="surface-header"><div><h2>成长与盈利能力</h2><span>{translateResearchText(details.growth_trend || 'Insufficient Evidence')}</span></div><strong>{module.score == null ? '暂无评分' : `${module.score}/100`}</strong></div><MetricRows rows={[["营收同比",module.metrics.revenue_growth,'pct'],["EPS 增长",module.metrics.eps_growth,'pct'],["毛利率",module.metrics.gross_margin,'pct'],["营业利润率",module.metrics.operating_margin,'pct'],["净利率",module.metrics.net_margin,'pct'],["ROIC",module.metrics.roic,'pct'],["ROE",module.metrics.roe,'pct'],["ROA",module.metrics.roa,'pct']]}/></section></div><section className="surface engine-card"><div className="surface-header"><div><h2>最近季度趋势</h2><span>最多 8 个季度；同季度申报去重</span></div></div>{quarters.length ? <div className="engine-table"><div className="head"><span>季度</span><span>营收</span><span>营收同比</span><span>EPS</span><span>净利润率</span><span>FCF</span></div>{quarters.map(row => <div key={String(row.period)}><strong>{String(row.period)}</strong><span>{formatResearchValue(row.revenue,'money')}</span><span>{formatResearchValue(row.revenue_yoy,'pct')}</span><span>{formatResearchValue(row.eps)}</span><span>{formatResearchValue(row.net_margin,'pct')}</span><span>{formatResearchValue(row.free_cash_flow,'money')}</span></div>)}</div> : <p className="insufficient-note">暂无可靠季度趋势数据</p>}</section><div className="engine-two-col"><section className="surface engine-card"><div className="surface-header"><div><h2>财务健康</h2><span>确定性规则评价</span></div></div><MetricRows rows={[["资产负债表",details.financial_health?.balance_sheet],["债务风险",details.financial_health?.debt_risk],["现金创造能力",details.financial_health?.cash_generation],["流动比率",module.metrics.current_ratio],["利息保障倍数",module.metrics.interest_coverage],["负债权益比",module.metrics.debt_to_equity]]}/></section><section className="surface engine-card"><div className="surface-header"><div><h2>股东价值</h2><span>缺失数据明确标注</span></div></div><MetricRows rows={[["流通股数",details.shareholder_value?.shares_outstanding],["股本变化",details.shareholder_value?.share_count_change,'pct'],["股权激励",details.shareholder_value?.stock_based_compensation],["股票回购",details.shareholder_value?.buyback],["股息",details.shareholder_value?.dividend]]}/></section></div><SourceFooter result={result} ids={module.sources}/></> }
function ValuationView({ result }: { result: StockResearchResult }) { const module = result.modules.valuation; const details = module.details as { history?: Record<string,{current?:unknown;median?:unknown;percentile?:unknown;available_periods?:number}>; peer_set?: string[]; peer_note?: string }; const labels: Record<string,string> = { pe_ttm:'滚动市盈率',forward_pe:'预期市盈率',peg:'PEG',price_to_sales:'市销率',ev_sales:'企业价值/销售额',ev_ebitda:'企业价值/EBITDA',fcf_yield:'自由现金流收益率' }; return <><section className="surface engine-card"><div className="surface-header"><div><h2>当前估值</h2><span>{translateResearchText(module.summary)}</span></div><strong>{module.score == null ? 'N/A' : `${module.score}/100`}</strong></div><div className="valuation-grid">{Object.entries(labels).map(([key,label]) => <div key={key}><span>{label}</span><strong>{formatResearchValue(module.metrics[key],key === 'fcf_yield' ? 'pct' : 'number')}</strong></div>)}</div></section><section className="surface engine-card"><div className="surface-header"><div><h2>历史估值</h2><span>仅按实际可用滚动记录计算，不冒充 5 年数据</span></div></div><div className="engine-table valuation-history"><div className="head"><span>指标</span><span>当前值</span><span>可用中位数</span><span>样本数</span><span>历史百分位</span></div>{Object.entries(details.history || {}).map(([key,row]) => <div key={key}><strong>{labels[key] || key}</strong><span>{formatResearchValue(row.current,key === 'fcf_yield' ? 'pct' : 'number')}</span><span>{formatResearchValue(row.median,key === 'fcf_yield' ? 'pct' : 'number')}</span><span>{row.available_periods || 0}</span><span>{formatResearchValue(row.percentile,'pct')}</span></div>)}</div></section><section className="surface engine-card"><div className="surface-header"><div><h2>同行比较</h2><span>自动同行候选；避免为展示而重复请求外部数据</span></div></div><div className="peer-chips"><strong>{result.ticker}</strong>{(details.peer_set || []).map(peer => <span key={peer}>{peer}</span>)}</div><p className="insufficient-note">{translateResearchText(details.peer_note || '暂无可靠同行比较数据')}</p></section><SourceFooter result={result} ids={module.sources}/></> }
function ExpectationsView({ result }: { result: StockResearchResult }) {
  const module = result.modules.expectations; const catalyst = result.modules.catalyst; const upcoming = (module.details.upcoming_earnings || null) as Record<string,unknown> | null; const timeline = (catalyst.details.timeline || []) as Record<string,unknown>[];
  return <><div className="engine-two-col"><section className="surface engine-card"><div className="surface-header"><div><h2>市场预期</h2><span>{translateResearchText(module.summary)}</span></div><strong>{module.score == null ? 'N/A' : `${module.score}/100`}</strong></div><MetricRows rows={[["EPS Consensus",module.metrics.forward_eps],["Revenue Consensus",module.metrics.revenue_consensus,'money'],["Revision Trend",module.metrics.revision_trend],["Next Earnings",upcoming?.release_date],["Release Window",upcoming?.when]]}/><p className="insufficient-note">缺少 30/60/90 日一致预期快照时，不推断上修或下修。</p></section><section className="surface engine-card"><div className="surface-header"><div><h2>盈利动量</h2><span>历史超预期表现与可用一致预期的规则评分</span></div><StatusPill value={module.status}/></div><MetricRows rows={[["Momentum Score",module.metrics.earnings_momentum],["Estimate Revision",module.metrics.revision_trend],["Confidence",module.confidence,'pct']]}/></section></div><section className="surface engine-card"><div className="surface-header"><div><h2>催化剂时间线</h2><span>事件事实与方向判断分离</span></div><StatusPill value={catalyst.status}/></div>{timeline.length ? <div className="catalyst-list">{timeline.map((item,index) => { const timestamp = formatCatalystTimestamp(item.event_date); return <article key={`${String(item.title)}-${index}`}><time><span>{timestamp.date}</span>{timestamp.time && <small>{timestamp.time}</small>}</time><div><strong>{translateResearchText(item.title)}</strong><p>{translateResearchText(item.description || '')}</p></div><em>{levelLabel(String(item.impact || '')) || '未评分'}</em></article> })}</div> : <p className="insufficient-note">暂无可靠催化剂数据</p>}</section><SourceFooter result={result} ids={[...module.sources,...catalyst.sources]}/></>
}
function RiskRadarView({ result }: { result: StockResearchResult }) { const module = result.modules.risk; const radar = (module.details.radar || []) as (EngineFinding & { recent_change?: string; updated_at?: string })[]; return <><section className="surface engine-card"><div className="surface-header"><div><h2>{result.ticker} 风险雷达</h2><span>{translateResearchText(module.summary)}</span></div><strong className={`risk-level ${result.risk_level.toLowerCase()}`}>{levelLabel(result.risk_level)}</strong></div><div className="risk-radar-list">{radar.map(risk => <details key={risk.name}><summary><span>{translateResearchText(risk.name)}</span><em className={(risk.level || 'unknown').toLowerCase()}>{levelLabel(risk.level)}</em></summary><p>{translateResearchText(risk.reason)}</p><small>置信度 {formatResearchValue(risk.confidence,'pct')} · 证据 {(risk.evidence || []).map(translateResearchText).join('、') || '证据不足'}</small></details>)}</div></section><SourceFooter result={result} ids={module.sources}/></> }
function FundFlowView({ result }: { result: StockResearchResult }) { const module = result.modules.fund_flow; return <><section className="surface engine-card"><div className="surface-header"><div><h2>{result.ticker} 资金流</h2><span>{translateResearchText(module.summary)}</span></div><strong>{module.score == null ? 'N/A' : `${module.score}/100`}</strong></div><MetricRows rows={[["Volume",module.metrics.volume],["Relative Volume",module.metrics.relative_volume],["CMF20",module.metrics.cmf20],["OBV",module.metrics.obv],["Accumulation / Distribution",module.metrics.accumulation_distribution],["Institutional Flow",module.metrics.institutional_flow],["ETF Flow",module.metrics.etf_flow]]}/><p className="insufficient-note">上述为价量代理指标，不代表真实资金净流入金额；机构与 ETF 资金流未接入时保持空值。</p></section><MoneyflowPanel ticker={result.ticker} analysis={module.details.flow_analysis as FlowAnalysis | undefined}/><SourceFooter result={result} ids={module.sources}/></> }

function ResearchStockChart({ symbol }: { symbol: string }) {
  const [chart, setChart] = useState<PositionChartState | null>(null);
  const [view, setView] = useState<PriceChartView>('line');
  const [extended, setExtended] = useState(false);
  const [layers, setLayers] = useState<ChartDisplaySettings>(CHART_PRESETS.standard);
  const [preset, setPreset] = useState<ActiveChartPreset>('standard');
  useEffect(() => { const frame = window.requestAnimationFrame(() => { try { const saved = JSON.parse(localStorage.getItem(CHART_DISPLAY_STORAGE_KEY) || 'null') as Partial<ChartPreference> | null; if (saved?.layers) { setLayers({ ...CHART_PRESETS.standard, ...saved.layers }); setPreset(saved.preset || 'custom') } } catch { /* 使用标准显示设置 */ } }); return () => window.cancelAnimationFrame(frame) }, []);
  useEffect(() => { if (!chart) return; const close = (event: KeyboardEvent) => { if (event.key === 'Escape') setChart(null) }; document.addEventListener('keydown', close); return () => document.removeEventListener('keydown', close) }, [chart]);
  const fetchChart = async (range: PriceRange, includeExtended = false) => {
    setChart({ symbol, portfolioPrice: 0, range, points: [], quote: null, analysis: null, source: '', adjustmentMode: '', warmupBars: 0, includesExtendedHours: false, loading: true, error: '' });
    try { const result = await apiJson<PriceHistoryResponse>(`/api/price-history/${encodeURIComponent(symbol)}?range=${range}&extended=${includeExtended}`); setChart(current => current?.symbol === symbol && current.range === range ? { ...current, points: result.bars, quote: result.quote, analysis: result.technicalAnalysis, source: result.source, adjustmentMode: result.adjustmentMode, warmupBars: result.warmupBars, includesExtendedHours: result.includesExtendedHours, loading: false } : current) }
    catch (error) { const detail = error instanceof Error ? error.message : '未知错误'; setChart(current => current ? { ...current, loading: false, error: `行情图表加载失败：${detail}` } : current) }
  };
  const open = (nextView: PriceChartView) => { setView(nextView); setExtended(false); void fetchChart('1M') };
  const changeRange = (range: PriceRange) => { if (!chart || chart.range === range) return; void fetchChart(range, extended && (range === '1W' || range === '1M')) };
  const toggleExtended = () => { if (!chart || !['1W','1M'].includes(chart.range)) return; const next = !extended; setExtended(next); void fetchChart(chart.range, next) };
  const savePreference = (preference: ChartPreference) => { setLayers(preference.layers); setPreset(preference.preset); localStorage.setItem(CHART_DISPLAY_STORAGE_KEY, JSON.stringify(preference)) };
  const selectedRange = chart ? PRICE_RANGES.find(item => item.key === chart.range) : undefined;
  const displayedPrice = chart?.quote?.session === 'CLOSED' && chart.quote.regularClose != null ? chart.quote.regularClose : chart?.quote?.price;
  const displayedChange = chart?.quote ? quoteChangePct(chart.quote) : null;
  return <><section className="surface research-market-chart-card"><div><strong>{symbol} 行情图表</strong><span>使用与盯盘页面一致的实时行情、复权口径和技术指标</span></div><div><button type="button" onClick={() => open('line')}>查看价格走势</button><button type="button" className="primary" onClick={() => open('candles')}>查看 K 线图</button></div></section>{chart && <div className="position-chart-backdrop" role="presentation" onMouseDown={() => setChart(null)}><section className="position-chart-modal" role="dialog" aria-modal="true" aria-labelledby="research-chart-title" onMouseDown={event => event.stopPropagation()}><header><div className="price-chart-heading"><h2 id="research-chart-title">{symbol} 价格波动</h2><div className="chart-current-price"><strong>{formatUnitPrice(displayedPrice)}</strong>{chart.quote && <><em className={`market-session ${chart.quote.session.toLowerCase()}`}>{sessionLabel(chart.quote.session)}</em>{displayedChange != null && <span className={displayedChange >= 0 ? 'positive' : 'negative'}>{formatPct(displayedChange)}</span>}</>}</div><div className="quote-metadata"><span>{chart.quote?.session === 'CLOSED' ? '正式收盘价' : '最新价'}</span>{chart.quote?.regularClose != null && chart.quote.session !== 'CLOSED' && <span>正式收盘 {formatUnitPrice(chart.quote.regularClose)}</span>}{chart.quote?.timestamp && <span>报价 {formatPricePointTime(chart.quote.timestamp, selectedRange?.interval || '1d')}</span>}<span>{selectedRange?.intervalLabel || '日线'}</span>{chart.source && <span>{chart.source.replaceAll('_', ' ')}</span>}{chart.quote?.isDelayed && <span>延迟行情</span>}</div></div><button type="button" aria-label="关闭价格图表" onClick={() => setChart(null)}>×</button></header><div className="position-chart-toolbar"><nav className="position-chart-ranges" aria-label="价格图表时间范围">{PRICE_RANGES.map(item => <button type="button" key={item.key} className={chart.range === item.key ? 'active' : ''} aria-pressed={chart.range === item.key} onClick={() => changeRange(item.key)}>{item.label}</button>)}</nav><div className="chart-toolbar-actions"><label className={`extended-hours-toggle ${!['1W','1M'].includes(chart.range) ? 'disabled' : ''}`}><input type="checkbox" checked={extended} disabled={chart.loading || !['1W','1M'].includes(chart.range)} onChange={toggleExtended}/><span>盘前盘后</span></label><div className="position-chart-view-toggle" aria-label="图表类型">{([['line','走势'],['candles','K线']] as const).map(([item,label]) => <button type="button" key={item} className={view === item ? 'active' : ''} onClick={() => setView(item)}>{label}</button>)}</div></div></div><div className="position-chart-body">{chart.loading ? <div className="position-chart-loading"><i className="refresh-spinner" aria-hidden="true"/>正在加载并计算技术指标…</div> : chart.error ? <div className="position-chart-error">{chart.error}</div> : view === 'candles' && chart.analysis && chart.quote ? <CandlestickChart symbol={symbol} rangeLabel={selectedRange?.label || chart.range} interval={selectedRange?.interval || '1d'} points={chart.points} analysis={chart.analysis} quote={chart.quote} layers={layers} activePreset={preset} showExtendedHours={extended} onLayerChange={(key,enabled) => savePreference({preset:'custom',layers:{...layers,[key]:enabled}})} onPreset={next => savePreference({preset:next,layers:{...CHART_PRESETS[next]}})}/> : <StockPriceChart symbol={symbol} rangeLabel={selectedRange?.label || chart.range} interval={selectedRange?.interval || '1d'} points={chart.points}/>}</div><footer><span>{chart.adjustmentMode ? adjustmentLabel(chart.adjustmentMode) : '统一复权价格'} · {chart.warmupBars} 根预热K线 · SMA 按 {selectedRange?.interval || '1d'} 周期计算</span><span>点击窗口外或按 Esc 关闭</span></footer></section></div>}</>;
}

function StockDashboard({ result, run }: { result: StockResearchResult; run: ResearchRun }) { const invalidations = result.thesis_invalidation || []; return <><section className="surface stock-research-hero"><div><span className="stock-name">{result.ticker} <small>股票研究工作台</small></span><p>{result.from_cache ? '读取本地研究缓存' : '本次重新生成'} · {new Date(result.generated_at).toLocaleString()}</p></div><div><StatusPill value={result.status}/><strong className={`risk-level ${result.risk_level.toLowerCase()}`}>风险 {levelLabel(result.risk_level)}</strong></div></section><ScoreCards scores={result.scores}/><ResearchStockChart symbol={result.ticker}/><ResearchIntelligencePanel result={result}/><section className="surface thesis-card"><div className="surface-header"><div><h2>投资论点</h2><span>基于可追踪证据的决策支持，不输出买入或卖出建议</span></div></div><p>{translateResearchText(result.core_thesis || result.investment_thesis)}</p>{result.why_now && <p className="why-now"><strong>当前关注原因</strong> · {translateResearchText(result.why_now)}</p>}<div className="scenario-grid"><article><strong>乐观情景</strong><p>{translateResearchText(result.bull_case)}</p></article><article><strong>基准情景</strong><p>{translateResearchText(result.base_case)}</p></article><article><strong>悲观情景</strong><p>{translateResearchText(result.bear_case)}</p></article></div></section><div className="engine-two-col"><section className="surface thesis-list"><div className="surface-header"><div><h2>主要风险</h2><span>只展示有支持证据的风险</span></div></div>{result.key_risks.length ? result.key_risks.map((risk,index) => <div key={`${risk.name || risk.title}-${index}`}><strong>{translateResearchText(risk.name || risk.title)}</strong><span>{levelLabel(risk.level)}</span><p>{translateResearchText(risk.reason || risk.description)}</p></div>) : <p className="insufficient-note">证据不足</p>}</section><section className="surface thesis-list"><div className="surface-header"><div><h2>论点失效条件</h2><span>驱动因素 → 可观察指标 → 失效条件</span></div></div>{invalidations.map((item,index) => { const value = typeof item === 'string' ? {driver:'研究假设',monitor:'后续披露',condition:item} : item; return <div key={`${value.driver}-${index}`}><strong>{translateResearchText(value.driver)}</strong><p>监控：{translateResearchText(value.monitor)}</p><p>{translateResearchText(value.condition)}</p></div> })}</section></div><ResearchModuleOverview result={result} run={run}/><SourceFooter result={result}/></> }

function ResearchIntelligencePanel({ result }: { result: StockResearchResult }) {
  const confidence = result.research_confidence; const gate = result.quality_gate; const drivers = result.key_drivers; const conflicts = result.conflicts || []; const debate = result.investment_debate; const industry = result.industry_metrics; const supportTier = result.research_support_tier;
  if (!confidence || !gate || !drivers) return null;
  return <section className="surface intelligence-card"><div className="surface-header"><div><h2>研究情报</h2><span>{translateResearchText(result.industry_context?.profile || 'GENERAL')} · {translateResearchText(result.industry_context?.description || '跨模块证据聚合')}</span></div><div className="intelligence-confidence"><strong>{confidence.score}/100</strong><StatusPill value={supportTier || (gate.status === 'PASSED' ? confidence.band : 'LIMITED')}/></div></div>{industry && <div className="industry-coverage"><strong>行业数据覆盖</strong><span>{translateResearchText(industry.profile)} · 必需 {industry.available_count} / {industry.required_count} · 可选 {industry.optional_available_count ?? 0} / {industry.optional_count ?? 0}</span><i><b style={{width:`${(industry.required_coverage ?? industry.completeness)*100}%`}}/></i></div>}{result.main_tension && <div className="main-tension"><strong>核心矛盾</strong><p>{translateResearchText(result.main_tension.statement)}</p></div>}{gate.warnings.length > 0 && <div className="quality-warning"><strong>研究覆盖率 {gate.coverage}%</strong><span>{translateResearchText(gate.warnings.join(' '))}</span></div>}{debate && <div className="debate-grid"><div><h3>多头逻辑</h3>{debate.what_bulls_believe.slice(0,5).map((item,index)=><p key={`bull-${index}`}>{translateResearchText(item.claim)}</p>)}</div><div><h3>空头逻辑</h3>{debate.what_bears_believe.slice(0,5).map((item,index)=><p key={`bear-${index}`}>{translateResearchText(item.claim)}</p>)}</div><div><h3>最重要的观察点</h3>{debate.what_matters_most.slice(0,5).map((item,index)=><p key={`matter-${index}`}>{translateResearchText(item.claim)}</p>)}</div></div>}<div className="driver-grid"><div><h3>正面驱动因素</h3>{drivers.positive.length ? drivers.positive.map(item => <article key={item.id}><strong>{translateResearchText(item.title)}</strong><span>{levelLabel(item.importance)} · {Math.round(item.confidence*100)}%</span></article>) : <p>没有足够证据</p>}</div><div><h3>负面驱动因素</h3>{drivers.negative.length ? drivers.negative.map(item => <article key={item.id}><strong>{translateResearchText(item.title)}</strong><span>{levelLabel(item.importance)} · {Math.round(item.confidence*100)}%</span></article>) : <p>没有足够证据</p>}</div></div>{(result.confidence_contributors?.length || result.confidence_limitations?.length) && <div className="confidence-explanation"><div><strong>置信度支持因素</strong>{result.confidence_contributors?.map((item,index)=><span key={`strength-${index}`}>{translateResearchText(item)}</span>)}</div><div><strong>研究限制</strong>{result.confidence_limitations?.map((item,index)=><span key={`limit-${index}`}>{translateResearchText(item)}</span>)}</div></div>}{conflicts.length > 0 && <div className="conflict-list"><strong>相互矛盾的证据</strong>{conflicts.map(item => <span key={item.id}>{translateResearchText(item.description)} · {levelLabel(item.severity)}</span>)}</div>}</section>
}
function EarningsSecView({ result }: { result: StockResearchResult }) {
  const earnings = result.modules.earnings; const expectations = result.modules.expectations; const sec = result.modules.sec;
  const history = (earnings.details.history || []) as Record<string,unknown>[]; const filings = (sec.details.filings || []) as Record<string,unknown>[]; const upcoming = (expectations.details.upcoming_earnings || null) as Record<string,unknown> | null;
  return <><div className="engine-two-col"><section className="surface engine-card"><div className="surface-header"><div><h2>财报摘要</h2><span>{translateResearchText(earnings.summary)}</span></div><StatusPill value={earnings.status}/></div><MetricRows rows={[["下次财报",upcoming?.release_date],["发布时段",upcoming?.when],["最近 EPS 超预期幅度",earnings.metrics.latest_eps_surprise,'pct'],["最近营收超预期幅度",earnings.metrics.latest_revenue_surprise,'pct'],["历史超预期比例",earnings.metrics.beat_rate,'pct'],["盈利动量",expectations.metrics.earnings_momentum]]}/></section><section className="surface engine-card"><div className="surface-header"><div><h2>SEC 文件</h2><span>{translateResearchText(sec.summary)}</span></div><StatusPill value={sec.status}/></div><MetricRows rows={[["文件总数",sec.metrics.filing_count],["SEC 直接索引",sec.metrics.direct_sec_count],["风险条款变化",'尚未解析'],["数据置信度",sec.confidence,'pct']]}/><p className="insufficient-note">{translateResearchText(sec.details.status_note)}</p></section></div><section className="surface engine-card"><div className="surface-header"><div><h2>最近财报</h2><span>结构化实际值与预期值对比</span></div></div>{history.length ? <div className="engine-table research-five-col"><div className="head"><span>报告期</span><span>实际 EPS</span><span>预期 EPS</span><span>EPS 超预期</span><span>营收超预期</span></div>{history.slice(0,8).map((row,index) => <div key={`${String(row.period)}-${index}`}><strong>{String(row.period || '—')}</strong><span>{formatResearchValue(row.eps_actual)}</span><span>{formatResearchValue(row.eps_estimate)}</span><span>{formatResearchValue(row.eps_surprise,'pct')}</span><span>{formatResearchValue(row.revenue_surprise,'pct')}</span></div>)}</div> : <p className="insufficient-note">暂无可靠财报历史</p>}</section><section className="surface engine-card"><div className="surface-header"><div><h2>最近 SEC 文件</h2><span>研究引擎直接读取结构化元数据</span></div></div>{filings.length ? <div className="engine-table research-four-col"><div className="head"><span>表单类型</span><span>申报日期</span><span>报告期</span><span>来源</span></div>{filings.slice(0,12).map((row,index) => <div key={`${String(row.form)}-${String(row.filing_date)}-${index}`}><strong>{String(row.form || '—')}</strong><span>{String(row.filing_date || '—')}</span><span>{String(row.report_period || '—')}</span><span>{row.url ? <a href={String(row.url)} target="_blank" rel="noreferrer">{String(row.source || '查看文件')}</a> : String(row.source || '—')}</span></div>)}</div> : <p className="insufficient-note">暂无可追踪 SEC 文件</p>}</section><SourceFooter result={result} ids={[...earnings.sources,...expectations.sources,...sec.sources]}/></>;
}
function InstitutionalView({ result }: { result: StockResearchResult }) {
  const module = result.modules.institutional; const details = module.details as Record<string,unknown>;
  const holdings = (details.institutional_holdings || []) as Record<string,unknown>[]; const changes = (details['13f_changes'] || []) as Record<string,unknown>[]; const insiders = (details.insider_transactions || []) as Record<string,unknown>[]; const etfs = (details.etf_exposure || []) as Record<string,unknown>[];
  return <><section className="surface engine-card"><div className="surface-header"><div><h2>{result.ticker} 机构与内部人</h2><span>{translateResearchText(module.summary)}</span></div><StatusPill value={module.status}/></div><MetricRows rows={[["跟踪机构持仓",module.metrics.institutional_holders],["13F 变化",module.metrics['13f_changes']],["ETF 暴露",module.metrics.etf_exposure_count],["内部人买入(90d)",module.metrics.insider_buy_value_90d,'money'],["内部人卖出(90d)",module.metrics.insider_sell_value_90d,'money'],["置信度",module.confidence,'pct']]}/></section><div className="engine-two-col"><section className="surface engine-card"><div className="surface-header"><div><h2>机构持仓</h2><span>本地 SEC 13F 归档</span></div></div>{holdings.length ? <div className="compact-record-list">{holdings.map((item,index) => <div key={`${String(item.manager)}-${index}`}><strong>{String(item.manager)}</strong><span>{String(item.quarter || '')}</span><small>{formatResearchValue(item.market_value,'money')} · {formatResearchValue(item.portfolio_weight,'pct')}</small></div>)}</div> : <p className="insufficient-note">当前 edgar.db 没有该股票的跟踪机构持仓</p>}</section><section className="surface engine-card"><div className="surface-header"><div><h2>ETF 暴露</h2><span>最新本地 ETF 快照</span></div></div>{etfs.length ? <div className="compact-record-list">{etfs.map((item,index) => <div key={`${String(item.etf)}-${index}`}><strong>{String(item.etf)}</strong><span>{String(item.date || '')}</span><small>权重 {formatResearchValue(item.weight_pct)}% · {formatResearchValue(item.market_value,'money')}</small></div>)}</div> : <p className="insufficient-note">暂无该股票的 ETF 暴露快照</p>}</section></div><section className="surface engine-card"><div className="surface-header"><div><h2>13F 变化与内部人交易</h2><span>结构化变动，不依赖 /holders 文本解析</span></div></div><div className="engine-two-col embedded-columns"><div className="compact-record-list">{changes.slice(0,8).map((item,index) => <div key={`${String(item.manager)}-${index}`}><strong>{String(item.manager)}</strong><span>{translateResearchText(item.change_type)}</span><small>股数 {formatResearchValue(item.shares_change)} · 市值 {formatResearchValue(item.market_value_change,'money')}</small></div>)}{!changes.length && <p className="insufficient-note">暂无可比较的连续13F记录</p>}</div><div className="compact-record-list">{insiders.slice(0,8).map((item,index) => <div key={`${String(item.name)}-${index}`}><strong>{String(item.name || '内部人')}</strong><span>{translateResearchText(item.transaction_type)}</span><small>{String(item.transaction_date || item.filing_date || '')} · {formatResearchValue(item.value,'money')}</small></div>)}{!insiders.length && <p className="insufficient-note">近90日暂无内部人记录</p>}</div></div></section><SourceFooter result={result} ids={module.sources}/></>;
}
function MacroResearchView({ result }: { result: StockResearchResult }) {
  const module = result.modules.macro; const events = (module.details.upcoming_events || []) as Record<string,unknown>[];
  return <><section className="surface engine-card"><div className="surface-header"><div><h2>美国宏观环境</h2><span>{translateResearchText(module.summary)}</span></div><StatusPill value={module.status}/></div><div className="valuation-grid"><div><span>VIX</span><strong>{formatResearchValue(module.metrics.vix)}</strong></div><div><span>美元指数</span><strong>{formatResearchValue(module.metrics.dxy)}</strong></div><div><span>WTI</span><strong>{formatResearchValue(module.metrics.wti_crude)}</strong></div><div><span>黄金</span><strong>{formatResearchValue(module.metrics.gold)}</strong></div><div><span>联邦基金利率上限</span><strong>{formatResearchValue(module.metrics.fed_funds_upper)}</strong></div><div><span>2年期美债</span><strong>{formatResearchValue(module.metrics.dgs2)}</strong></div><div><span>10年期美债</span><strong>{formatResearchValue(module.metrics.dgs10)}</strong></div><div><span>10年-2年利差</span><strong>{formatResearchValue(module.metrics.t10y2y)}</strong></div></div></section><section className="surface engine-card"><div className="surface-header"><div><h2>宏观事件日历</h2><span>未来45天已公布日程</span></div></div>{events.length ? <div className="catalyst-list">{events.slice(0,16).map((item,index) => <article key={`${String(item.event_date)}-${index}`}><time>{String(item.event_date)}</time><div><strong>{translateResearchText(item.release_type || item.title)}</strong><p>{translateResearchText(item.title || '')}</p></div><em>{String(item.source || '')}</em></article>)}</div> : <p className="insufficient-note">暂无未来宏观事件日程</p>}</section><SourceFooter result={result} ids={module.sources}/></>;
}
function SupplyChainView({ result }: { result: StockResearchResult }) {
  const module = result.modules.supply_chain; const sourceRelations = (module.details.relationships || []) as Record<string,unknown>[]; const [relationUpdates,setRelationUpdates] = useState<Record<number,Record<string,unknown>>>({}); const [checking,setChecking] = useState<number | null>(null); const relations = sourceRelations.map(item => ({ ...item, ...(relationUpdates[Number(item.id)] || {}) }));
  const revalidate = async (id: number) => { setChecking(id); try { const updated = await apiJson<Record<string,unknown>>(`/api/research/supply-chain/relationships/${id}/revalidate`, { method: 'POST' }); setRelationUpdates(items => ({ ...items, [id]: { status: updated.status, evidence_status: updated.status, verified: false, source: ([...((updated.sources as Record<string,unknown>[] | undefined) || [])].reverse())[0]?.url, evidence_text: ([...((updated.sources as Record<string,unknown>[] | undefined) || [])].reverse())[0]?.evidence_text } })) } finally { setChecking(null) } };
  const category: Record<string,string> = { supplier: '供应商', customer: '客户', smaller_peer: '同业与竞争', beneficiary: '间接受益方', partner: '合作伙伴', platform: '平台', substitute: '替代技术' };
  return <><section className="surface engine-card"><div className="surface-header"><div><h2>{result.ticker} 产业关系</h2><span>{translateResearchText(module.summary)}</span></div><StatusPill value={module.status}/></div><MetricRows rows={[["关系总数",module.metrics.relationships],["Financial/Yahoo 调用",module.metrics.api_calls],["Tavily 验证",module.metrics.tavily_calls],["LLM Tokens",module.metrics.llm_tokens],["证据标准","共同提及 ≠ 关系证实"]]}/></section><RelationshipMap ticker={result.ticker} relations={relations}/><section className="surface engine-card"><div className="surface-header"><div><h2>公司产业关系</h2><span>公司身份已确认；以下关系描述为候选推断，不代表已核实。证据保留原文供核查。</span></div></div>{relations.length ? <div className="relationship-grid">{relations.map((item,index) => <article key={`${String(item.target_company)}-${String(item.relationship_type)}-${index}`}><div><strong>{String(item.target_company)}</strong><span>{category[String(item.relationship_type)] || translateResearchText(item.relationship_type)}</span></div><p>{'候选说明：' + String(item.description || '暂无关系说明')}</p>{Boolean(item.evidence_text) && <details><summary>查看搜索证据原文</summary><p>{String(item.evidence_text)}</p></details>}<footer><em className={item.verified ? 'verified' : 'unverified'}>{({ EVIDENCE_FOUND: '发现关系证据 · 待核原文', CO_MENTION: '仅搜索共同提及', LEGACY: '旧验证 · 待复核', VERIFIED: '旧验证 · 待复核', FETCH_FAILED: '核查失败', NOT_CONNECTED: '搜索未接入', NO_EVIDENCE: '未找到关系证据' } as Record<string,string>)[String(item.evidence_status || item.status)] || '关系待核查'}</em>{Boolean(item.source) ? <a href={String(item.source)} target="_blank" rel="noreferrer">查看来源</a> : <span>未保存可追溯来源</span>}{Boolean(item.id) && !item.verified && <button disabled={checking === Number(item.id)} onClick={() => void revalidate(Number(item.id))}>{checking === Number(item.id) ? '验证中…' : '重新验证'}</button>}</footer></article>)}</div> : <p className="insufficient-note">本次没有识别到可靠产业关系；系统不会用固定的 NVDA 模板填充其他股票。</p>}</section><SourceFooter result={result} ids={module.sources}/></>;
}
function ResearchModuleOverview({ result, run }: { result: StockResearchResult; run: ResearchRun }) {
  const names = [...new Set([...Object.keys(result.module_status), ...Object.keys(result.modules)])];
  return <section className="surface research-module-overview">
    <div className="surface-header"><div><h2>研究模块与数据质量</h2><span>运行状态、研究摘要与数据覆盖统一展示；展开查看来源、缺失字段和错误详情</span></div></div>
    {names.map(name => {
      const module = result.modules[name];
      const status = module?.status || result.module_status[name] || 'PENDING';
      const retryable = ['PARTIAL', 'PARTIAL_DATA', 'PARTIAL_ERROR', 'FAILED'].includes(status);
      const completeness = module?.completeness;
      const coverage = typeof completeness === 'number' && Number.isFinite(completeness) ? `${Math.round(Math.min(1, Math.max(0, completeness)) * 100)}%` : '待评估';
      return <article className="research-module-row" key={name}>
        <details>
          <summary>
            <span className="research-module-name">{MODULE_LABELS[name] || name}</span>
            <span className="research-module-badges"><StatusPill value={status}/><StatusPill value={module?.cache_hit === undefined ? '缓存待核查' : module.cache_hit ? 'CACHED' : 'UNCACHED'}/><span className="engine-status coverage">覆盖 {coverage}</span></span>
            <span className="research-module-cache">{module?.cache_expires_at ? `缓存至 ${new Date(module.cache_expires_at).toLocaleString()}` : '无缓存到期时间'}</span>
            <span className="research-module-toggle">详情</span>
            <span className="research-module-summary">{translateResearchText(module?.summary || '本次尚无模块摘要')}</span>
          </summary>
          <div className="research-module-diagnostics">
            <p>数据来源：{(module?.data_sources_used || module?.sources || []).join('、') || '暂无'}</p>
            <p>缺失字段：{(module?.missing_fields || []).join('、') || '无'}</p>
            <p>警告：{(module?.warnings || []).join('；') || '无'}</p>
            <p>错误：{(module?.provider_errors || []).map(item => `${item.provider}: ${item.message}`).join('；') || (module?.errors || []).join('；') || module?.error || '无'}</p>
          </div>
        </details>
        {retryable && <button className="research-module-retry" aria-label={`重试${MODULE_LABELS[name] || name}`} onClick={() => void run('stock', result.ticker, { runId: result.run_id, modules: [name] })}>重试</button>}
      </article>;
    })}
  </section>;
}

function ResearchDiagnostics({ result, tool, run }: { result: StockResearchResult; tool: ResearchTool; run: ResearchRun }) {
  const selected = RESEARCH_TOOL_MODULES[tool] || Object.keys(result.modules);
  return <section className="surface module-health"><div className="surface-header"><div><h2>模块数据质量</h2><span>数据来源、覆盖率与错误详情</span></div></div>{selected.map(name => { const module = result.modules[name]; if (!module) return null; const retryable = ['PARTIAL_DATA','PARTIAL_ERROR','FAILED'].includes(module.status); const coverage = Math.round((module.completeness ?? 0) * 100); return <details key={name}><summary><span>{MODULE_LABELS[name] || name}</span><StatusPill value={module.cache_hit ? 'CACHED' : 'UNCACHED'}/><span className="engine-status coverage">覆盖 {coverage}%</span><small>{module.cache_expires_at ? `缓存至 ${new Date(module.cache_expires_at).toLocaleString()}` : '无缓存到期时间'}</small>{retryable && <button onClick={event => { event.preventDefault(); void run(tool, result.ticker, { runId: result.run_id, modules: [name] }) }}>重试</button>}</summary><div className="research-diagnostic-detail"><p>数据来源：{(module.data_sources_used || module.sources || []).join('、') || '暂无'}</p><p>缺失字段：{(module.missing_fields || []).join('、') || '无'}</p><p>警告：{(module.warnings || []).join('；') || '无'}</p><p>错误：{(module.provider_errors || []).map(item => `${item.provider}: ${item.message}`).join('；') || (module.errors || []).join('；') || '无'}</p></div></details> })}</section>
}

function WhatChangedPanel({ result }: { result: StockResearchResult }) {
  const [comparison,setComparison] = useState<Record<string,unknown> | null>(null); const [open,setOpen] = useState(false);
  useEffect(() => { let active = true; apiJson<Record<string,unknown>>(`/api/research/history/${result.ticker}/compare`).then(data => { if (active) setComparison(data) }).catch(() => { if (active) setComparison(null) }); return () => { active = false } }, [result.run_id,result.ticker]);
  if (!comparison) return <section className="surface research-change-card"><strong>研究变化</strong><span>至少需要两次研究快照才能比较。</span></section>;
  const scoreChanges = comparison.score_changes as Record<string,{old?:unknown;new?:unknown}> || {}; const drivers = comparison.driver_changes as Record<string,unknown> || {}; const scenarios = comparison.scenario_changes as Record<string,unknown> || {};
  return <section className="surface research-change-card"><button onClick={() => setOpen(value => !value)}><span><strong>研究变化 v2</strong><small>论点、驱动因素、风险、催化剂与情景变化</small></span><em>{Object.keys(scoreChanges).length} 项评分变化</em></button>{open && <div><p>评分：{Object.entries(scoreChanges).map(([key,value]) => `${MODULE_LABELS[key] || translateResearchText(key)} ${String(value.old ?? 'N/A')} → ${String(value.new ?? 'N/A')}`).join('；') || '无变化'}</p><p>论点：{JSON.stringify(comparison.thesis_changes || {}).slice(0,500)}</p><p>驱动因素：{JSON.stringify(drivers).slice(0,700)}</p><p>风险：{JSON.stringify(comparison.risk_changes || {}).slice(0,500)}</p><p>催化剂：{JSON.stringify(comparison.catalyst_changes || {}).slice(0,500)}</p><p>情景：{JSON.stringify(scenarios).slice(0,700)}</p></div>}</section>;
}

function PeerManager({ result, run }: { result: StockResearchResult; run: ResearchRun }) {
  const module = result.modules.valuation; const details = module.details as Record<string,unknown>; const rows = (details.peer_comparison || []) as Record<string,unknown>[]; const [peers,setPeers] = useState<string[]>((details.peer_set || []) as string[]); const [input,setInput] = useState(''); const [busy,setBusy] = useState('');
  const add = async () => { const peer = input.trim().toUpperCase(); if (!peer) return; setBusy(peer); try { await apiJson(`/api/research/peers/${result.ticker}`, { method:'POST', body:JSON.stringify({ peer_ticker:peer }) }); setPeers(value => [...new Set([...value,peer])]); setInput('') } finally { setBusy('') } };
  const remove = async (peer:string) => { setBusy(peer); try { await apiJson(`/api/research/peers/${result.ticker}/${peer}`, { method:'DELETE' }); setPeers(value => value.filter(item => item !== peer)) } finally { setBusy('') } };
  return <section className="surface engine-card peer-manager"><div className="surface-header"><div><h2>已验证同行数据集</h2><span>{translateResearchText(details.peer_note || '同行验证结果')}</span></div><button onClick={() => void run('valuation', result.ticker)}>刷新比较</button></div><div className="peer-editor"><input value={input} onChange={event => setInput(event.target.value.toUpperCase().replace(/[^A-Z0-9.-]/g,'').slice(0,8))} placeholder="添加同行代码"/><button disabled={!input || !!busy} onClick={() => void add()}>添加</button></div><div className="peer-chips">{peers.map(peer => <span key={peer}>{peer}<button disabled={busy===peer} onClick={() => void remove(peer)}>×</button></span>)}</div>{rows.length ? <div className="engine-table peer-table"><div className="head"><span>同行</span><span>相关性</span><span>市值</span><span>P/E</span><span>营收增长</span><span>经营利润率</span><span>ROIC</span></div>{rows.map(row => <div key={String(row.ticker)}><strong>{String(row.ticker)}</strong><span>{translateResearchText(row.relevance_label || '待评估')} · {String(row.relevance_score ?? '—')}</span><span>{formatResearchValue(row.market_cap,'money')}</span><span>{formatResearchValue(row.pe_ttm)}</span><span>{formatResearchValue(row.revenue_growth,'pct')}</span><span>{formatResearchValue(row.operating_margin,'pct')}</span><span>{formatResearchValue(row.roic,'pct')}</span></div>)}</div> : <p className="insufficient-note">当前同行尚未通过公司与财务数据验证。修改后点击“刷新比较”。</p>}</section>;
}

function secSourceUrl(value: unknown) {
  if (typeof value !== 'string') return null;
  try { const url = new URL(value); return ['https:', 'http:'].includes(url.protocol) ? url.href : null } catch { return null }
}
function SecSourceTitle({ row, title }: { row: Record<string,unknown>; title: string }) {
  const url = secSourceUrl(row.section_url) || secSourceUrl(row.document_url) || secSourceUrl(row.source_url);
  return <b>{url ? <a className="sec-source-title" href={url} target="_blank" rel="noopener noreferrer" aria-label={`${title}：打开原始申报文件（新标签页）`}>{title} <span aria-hidden="true">↗</span></a> : title}</b>;
}
function SecSourceFooter({ row }: { row: Record<string,unknown> }) {
  const section = secSourceUrl(row.section_url); const document = secSourceUrl(row.document_url); const index = secSourceUrl(row.index_url) || secSourceUrl(row.source_url);
  return <footer className="sec-source-footer"><span>{[row.filing_type, row.filing_date].filter(Boolean).map(String).join(' · ')}</span>{section && <a href={section} target="_blank" rel="noopener noreferrer">定位相关章节 ↗</a>}{document && <a href={document} target="_blank" rel="noopener noreferrer">报告正文 ↗</a>}{index && index !== document && <a href={index} target="_blank" rel="noopener noreferrer">申报目录 ↗</a>}{!document && <span>{index ? '正文链接暂未解析' : '暂无出处链接'}</span>}</footer>;
}
function SecDepthPanel({ result }: { result: StockResearchResult }) {
  const details = result.modules.sec.details as Record<string,unknown>; const findings = (details.sec_findings || []) as Record<string,unknown>[]; const changes = (details.risk_factor_changes || []) as Record<string,unknown>[];
  return <section className="surface engine-card"><div className="surface-header"><div><h2>SEC 深度发现</h2><span>点击标题查看对应申报文件</span></div><StatusPill value={result.modules.sec.status}/></div><div className="sec-depth-grid"><div><strong>风险因素变化</strong>{changes.length ? changes.slice(0,10).map((row,index) => <article key={`${String(row.title)}-${index}`}><em>{translateResearchText(row.change_type) || '变化'}</em><SecSourceTitle row={row} title={chineseSecRiskTitle(row.title)}/><p>{chineseSecRiskText(row.text, row.title, row.change_type)}</p><SecSourceFooter row={row}/></article>) : <p className="insufficient-note">没有足够的连续文件形成可靠差异。</p>}</div><div><strong>结构化发现</strong>{findings.length ? findings.slice(0,10).map((row,index) => <article key={`${String(row.accession_number)}-${index}`}><em>{String(row.filing_type)}</em><SecSourceTitle row={row} title={translateResearchText(row.title)}/><p>{String(row.summary || '').slice(0,240)}</p><SecSourceFooter row={row}/></article>) : <p className="insufficient-note">正文暂未成功解析。</p>}</div></div></section>;
}

function ExpectationsDepthPanel({ result }: { result: StockResearchResult }) {
  const details = result.modules.expectations.details as Record<string,unknown>;
  const matrix = details.capability_matrix as Record<string,Record<string,unknown>> || {};
  const guidance = (details.guidance || []) as Record<string,unknown>[];
  const quality = details.guidance_quality as { filtered?: number; duplicates?: number } | undefined;
  const labels: Record<string,string> = { current_consensus: '当前一致预期', earnings_surprise_history: '历史业绩对照', revision_30d: '30 日预期修正', revision_60d: '60 日预期修正', revision_90d: '90 日预期修正' };
  const groups = [
    { id: 'guidance', title: '业绩指引候选', description: '包含量化预期或明确指引措辞；数值、单位和期间以原文为准。', empty: '本次材料未提取到可靠业绩指引。这不表示公司没有发布指引。' },
    { id: 'outlook', title: '经营展望', description: '管理层对需求、经营和业务发展的前瞻描述。', empty: '本次材料未提取到明确经营展望。' },
    { id: 'risk', title: '经营风险与约束', description: '可能影响业绩的条件和风险，不作为公司承诺或业绩指引。', empty: '本次材料未提取到相关经营风险说明。' },
  ];
  return <section className="surface engine-card expectations-evidence">
    <div className="surface-header"><div><h2>市场预期与管理层展望</h2><span>数据状态基于本次研究；公司表述与分析师一致预期分开展示</span></div></div>
    <div className="expectations-section-heading"><h3>本次预期数据</h3><span>研究时间：{new Date(result.generated_at).toLocaleString()}</span></div>
    <div className="capability-grid">{Object.entries(matrix).filter(([key]) => !key.startsWith('revision_')).map(([key,value]) => <div key={key}><strong>{labels[key] || translateResearchText(key.replaceAll('_',' '))}</strong><StatusPill value={String(({ NOT_CONNECTED: '尚未接入', FETCH_FAILED: '获取失败', NO_DATA: '本次未返回', UNKNOWN: '待重新核查' } as Record<string,string>)[String(value.status)] || value.status || '本次未返回')}/><small>{translateResearchText(value.reason || '本次尚无可用数据')}</small></div>)}</div>
    <p className="expectations-method">30 / 60 / 90 日预期修正：尚未接入历史快照，非查询故障。需要同一报告期在不同日期的一致预期，刷新不能补齐。旧研究需重新运行，才能使用新的预期获取与材料提取逻辑。</p>
    <p className="expectations-method">材料范围：10-K / 10-Q 管理层讨论及 8-K 业绩、经营披露正文。尚未完整覆盖财报附件及电话会，因此“本次未提取到”不等于公司未发布指引。</p>
    <p className="expectations-method">以下内容为申报材料中的证据摘录，已过滤通用免责声明并合并重复表述。“公司明确上调／下调／维持”只在原文明示时使用，未独立核验前后两期指引数值。{quality && (Number(quality.filtered || 0) + Number(quality.duplicates || 0) > 0) && <span> 本次整理缓存候选：过滤 {quality.filtered || 0} 条，合并重复 {quality.duplicates || 0} 条。</span>}</p>
    {groups.map(group => { const rows = guidance.filter(row => row.group === group.id); return <section className={`expectations-group group-${group.id}`} key={group.id}>
      <div className="expectations-section-heading"><h3>{group.title} <span>{rows.length}</span></h3><p>{group.description}</p></div>
      {rows.length ? <div className="guidance-list">{rows.map((row,index) => <article key={`${String(row.filing_date)}-${index}`}>
        <div className="guidance-evidence-heading"><strong>{String(row.metric_label || translateResearchText(row.metric))}</strong><span className="engine-status">{String(row.status_label || '未确认指引变动')}</span></div>
        <div className="guidance-evidence-period">适用期间：{String(row.period_label || '原文未明确期间')}</div>
        <p>{String(row.evidence_text || '')}</p>
        {Boolean(row.evidence_text_original) && <details className="guidance-original"><summary>查看英文证据</summary><p lang="en">{String(row.evidence_text_original)}</p></details>}
        <SecSourceFooter row={row}/>
      </article>)}</div> : <p className="insufficient-note">{group.empty}</p>}
    </section> })}
  </section>;
}

function StockResearchView({ tool, ticker, result, progress, busy, run }: { tool: ResearchTool; ticker: string; result: StockResearchResult | null; progress: Record<string,string>; busy: boolean; run: ResearchRun }) { if (busy) return <ResearchProgress progress={progress}/>; if (!result) return <EmptyResearch ticker={ticker}/>; const view = tool === 'stock' ? <><StockDashboard result={result} run={run}/><WhatChangedPanel result={result}/></> : tool === 'fundamentals' ? <FundamentalView result={result}/> : tool === 'valuation' ? <><ValuationView result={result}/><PeerManager result={result} run={run}/></> : tool === 'earnings' ? <><EarningsSecView result={result}/><SecDepthPanel result={result}/></> : tool === 'expectations' ? <><ExpectationsView result={result}/><ExpectationsDepthPanel result={result}/></> : tool === 'institutional' ? <InstitutionalView result={result}/> : tool === 'moneyflow' ? <FundFlowView result={result}/> : tool === 'macro' ? <MacroResearchView result={result}/> : tool === 'chain' ? <SupplyChainView key={`${result.ticker}:${result.run_id}:${result.generated_at}`} result={result}/> : <RiskRadarView result={result}/>; return <>{view}{tool !== 'stock' && <ResearchDiagnostics result={result} tool={tool} run={run}/>}</> }
