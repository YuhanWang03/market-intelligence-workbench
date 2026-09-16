'use client';

/** Signed-session and frozen-snapshot fetch helpers shared by the workbench. */

export type AccessRole = 'unknown' | 'anonymous' | 'owner' | 'guest';
export type AccessStatus = {
  authenticated: boolean;
  role: Exclude<AccessRole, 'unknown'>;
  username: string;
  guest_enabled: boolean;
  auth_configured: boolean;
  snapshot_updated_at: string | null;
  session_expires_at: number | null;
};

type SnapshotEnvelope<T> = { path: string; payload: T; published_at: string; schema_version: number };
let accessRole: AccessRole = 'unknown';

export function setApiAccessRole(role: AccessRole) { accessRole = role; }

/** Kept for older imports. Browser credentials now live only in HttpOnly cookies. */
export function authHeaders(): Record<string, string> { return {}; }

export class ApiError extends Error {
  status: number;
  detail: string;
  constructor(status: number, detail: string) {
    super(detail ? `${status}: ${detail}` : `${status}`);
    this.status = status;
    this.detail = detail;
  }
}

async function responseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = '';
    try { const body = await response.json() as { detail?: unknown }; detail = typeof body.detail === 'string' ? body.detail : body.detail ? JSON.stringify(body.detail) : '' } catch { /* non-JSON error body */ }
    throw new ApiError(response.status, detail);
  }
  return response.json() as Promise<T>;
}

async function rawJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, { cache: 'no-store', credentials: 'same-origin', ...init, headers: { 'Content-Type': 'application/json', ...((init?.headers as Record<string, string>) || {}) } });
  return responseJson<T>(response);
}

function canPublishSnapshot(path: string): boolean {
  const pathname = path.split('?', 1)[0];
  const exact = new Set(['/api/portfolio', '/api/risk', '/api/tickertape', '/api/activity', '/api/macro', '/api/history', '/api/flow_status', '/api/recommendations', '/api/monitoring/universe', '/api/watchlist', '/api/price-alerts', '/api/costs', '/api/lab/screening/criteria', '/api/lab/universes', '/api/lab/signals', '/api/lab/runs', '/api/lab/committee/personas', '/api/lab/committee/runs', '/api/lab/committee/scoreboard', '/api/lab/committee/pricing', '/api/research/results/NVDA', '/api/research/history/NVDA', '/api/research/history/NVDA/compare', '/api/research/peers/NVDA']);
  return exact.has(pathname) || pathname.startsWith('/api/price-history/') || pathname.startsWith('/api/lab/runs/') || pathname.startsWith('/api/lab/committee/runs/');
}

export async function authStatus(): Promise<AccessStatus> {
  return rawJson<AccessStatus>('/api/auth/status');
}

export async function authLogin(username: string, password: string): Promise<AccessStatus> {
  return rawJson<AccessStatus>('/api/auth/login', { method: 'POST', body: JSON.stringify({ username, password }) });
}

export async function authGuest(): Promise<AccessStatus> {
  return rawJson<AccessStatus>('/api/auth/guest', { method: 'POST', body: '{}' });
}

export async function authLogout(): Promise<void> {
  await rawJson('/api/auth/logout', { method: 'POST', body: '{}' });
}

export async function apiJson<T>(path: string, init?: RequestInit): Promise<T> {
  const method = String(init?.method || 'GET').toUpperCase();
  if (accessRole === 'guest') {
    if (method !== 'GET' || !canPublishSnapshot(path)) throw new ApiError(403, '访客模式为只读，不能执行此操作');
    const envelope = await rawJson<SnapshotEnvelope<T>>(`/api/public/snapshot?path=${encodeURIComponent(path)}`);
    return envelope.payload;
  }
  const payload = await rawJson<T>(path, init);
  if (accessRole === 'owner' && method === 'GET' && canPublishSnapshot(path)) {
    try {
      await rawJson('/api/public/snapshot', { method: 'POST', body: JSON.stringify({ path, payload }) });
    } catch { /* A snapshot failure must never break the owner's live request. */ }
  }
  return payload;
}

export type AgentV2Evidence = {
  id: string;
  entity: string;
  claim: string;
  metric?: string;
  value?: unknown;
  unit?: string;
  period?: string;
  as_of: string;
  source_id: string;
  source_title: string;
  source_url: string;
  confidence: number | null;
};

export type AgentV2TraceStep = { round: number; action: string; detail: string; ms: number };
export type AgentV2SubAgent = {
  capability: string;
  name: string;
  label: string;
  subject: string;
  rounds: number;
  llm_calls: number;
  elapsed_ms: number;
  seconds_allowed?: number | null;
  stop_reason: string;
  calls: Record<string, number | null | undefined>;
  intraday?: boolean;
  notes?: string[];
  stance?: string;
  trace: AgentV2TraceStep[];
  nested: { label: string; rounds: number; elapsed_ms: number; stop_reason: string; calls: Record<string, number | null | undefined>; trace: AgentV2TraceStep[] }[];
};

export type AgentV2Response = {
  run_id: string;
  status: string;
  answer: string;
  answer_mode: string;
  request?: { original_text: string; text: string; session_id: string; entities: string[]; metadata?: Record<string, unknown> };
  route: { kind: string; packs: string[]; reason: string; asynchronous: boolean };
  plan: {
    objective: string;
    tasks: { id: string; capability: string; purpose: string }[];
    assumptions: string[];
  };
  evidence: AgentV2Evidence[];
  sub_agents?: AgentV2SubAgent[];
  verification: {
    ok: boolean;
    warnings: string[];
    unknown_citations: string[];
    ungrounded_numbers: string[];
  };
  elapsed_ms: number;
  error: string;
  synthesis?: { outcome: string; draft: string; attempts: { stage: string; ok: boolean; warnings: string[]; unknown_citations: string[]; ungrounded_numbers: string[] }[] };
  pending_mutation?: AgentPendingMutation | null;
  interface: 'web';
  policy: { web_requested: boolean; web_enabled: boolean; web_allowed: boolean; mutations_enabled?: boolean };
};

export type AgentPendingMutation = { operation: string; payload: Record<string, unknown>; description: string };

export type AgentV2Job = {
  job_id: string;
  status: 'running' | 'completed' | 'failed';
  agent_status: string;
  progress: string;
  error?: string;
  result?: AgentV2Response;
};

export type PageSelection = {
  kind: 'position' | 'anomaly' | 'price_alert' | 'research';
  ticker?: string; record_id?: string; occurred_at?: string; excerpt?: string;
};
export type PageContext = {
  section: 'core' | 'research' | 'lab' | 'cost'; label: string; tool: string;
  captured_at: string; data_status: 'available' | 'unavailable'; selection?: PageSelection;
};

export function askAgentV2(text: string, sessionId: string, allowWeb: boolean, signal?: AbortSignal, version: 'v2' | 'v3' = 'v2', pageContext?: PageContext) {
  return apiJson<AgentV2Response | AgentV2Job>(`/api/agent-${version}/ask`, {
    method: 'POST',
    body: JSON.stringify({ text, session_id: sessionId, allow_web: allowWeb, ...(version === 'v3' && pageContext ? { page_context: pageContext } : {}) }),
    signal,
  });
}

/** Approve or reject the write an Agent V3 run stopped on; the graph resumes from its checkpoint. */
export function confirmAgentV3Run(runId: string, sessionId: string, approve: boolean, signal?: AbortSignal) {
  return apiJson<AgentV2Response>(`/api/agent-v3/runs/${encodeURIComponent(runId)}/confirm`, {
    method: 'POST',
    body: JSON.stringify({ session_id: sessionId, approve }),
    signal,
  });
}

export function getAgentV2Job(jobId: string, signal?: AbortSignal, version: 'v2' | 'v3' = 'v2') {
  return apiJson<AgentV2Job>(`/api/agent-${version}/jobs/${encodeURIComponent(jobId)}`, { signal });
}
