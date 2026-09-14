'use client';

/** Owner-token aware fetch helpers shared by the workbench pages. */

export function authHeaders(): Record<string, string> {
  const token = typeof window === 'undefined' ? '' : localStorage.getItem('ownerToken') || localStorage.getItem('dashboard:owner_token') || '';
  return token ? { 'X-Owner-Token': token } : {};
}

export class ApiError extends Error {
  status: number;
  detail: string;
  constructor(status: number, detail: string) {
    super(detail ? `${status}: ${detail}` : `${status}`);
    this.status = status;
    this.detail = detail;
  }
}

export async function apiJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, { cache: 'no-store', ...init, headers: { 'Content-Type': 'application/json', ...authHeaders(), ...((init?.headers as Record<string, string>) || {}) } });
  if (!response.ok) {
    let detail = '';
    try { const body = await response.json() as { detail?: unknown }; detail = typeof body.detail === 'string' ? body.detail : body.detail ? JSON.stringify(body.detail) : '' } catch { /* non-JSON error body */ }
    throw new ApiError(response.status, detail);
  }
  return response.json() as Promise<T>;
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
  interface: 'web';
  policy: { web_requested: boolean; web_enabled: boolean; web_allowed: boolean };
};

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

export function getAgentV2Job(jobId: string, signal?: AbortSignal, version: 'v2' | 'v3' = 'v2') {
  return apiJson<AgentV2Job>(`/api/agent-${version}/jobs/${encodeURIComponent(jobId)}`, { signal });
}
