'use client';

import { useEffect, useMemo, useRef, useState } from 'react';

type Relation = Record<string, unknown>;
const GROUPS = [
  { id: 'supplier', title: '供应商候选', link: '上游供给 ↓' },
  { id: 'smaller_peer', title: '同业与竞争候选', link: '竞争对标' },
  { id: 'beneficiary', title: '间接受益候选', link: '需求外溢' },
  { id: 'customer', title: '客户候选', link: '产品交付 ↓' },
];
const TYPES: Record<string, string> = { competitor: 'smaller_peer', supplier: 'supplier', customer: 'customer', beneficiary: 'beneficiary' };
const STATES: Record<string, string> = { EVIDENCE_FOUND: '发现关系证据 · 待核原文', CO_MENTION: '仅搜索共同提及', LEGACY: '旧验证 · 待复核', VERIFIED: '旧验证 · 待复核', FETCH_FAILED: '核查失败', NOT_CONNECTED: '搜索未接入', NO_EVIDENCE: '未找到关系证据' };
function sourceUrl(value: unknown) {
  try { const url = new URL(String(value || '')); return ['http:', 'https:'].includes(url.protocol) ? url.href : null; } catch { return null; }
}
function stateLabel(row: Relation) { return STATES[String(row.evidence_status || row.status)] || '关系待核查'; }

export function RelationshipMap({ ticker, relations }: { ticker: string; relations: Relation[] }) {
  const rows = useMemo(() => relations.filter(row => row.target_company).map((row, index) => {
    const raw = String(row.relationship_type || '').toLowerCase();
    return { row, group: TYPES[raw] || raw, key: `${ticker}:${row.id ?? index}:${row.target_company}:${raw}` };
  }), [ticker, relations]);
  const [selection, setSelection] = useState<string | null>(null);
  const selected = rows.find(row => row.key === selection);
  const mapRef = useRef<HTMLDivElement>(null);
  const [drawing, setDrawing] = useState<{ width: number; height: number; lines: { id: string; x1: number; y1: number; x2: number; y2: number; label: string }[] }>({ width: 1, height: 1, lines: [] });
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const draw = () => {
      const box = map.getBoundingClientRect();
      const center = map.querySelector('[data-map-center]')!.getBoundingClientRect();
      const lines = GROUPS.flatMap(group => {
        const node = map.querySelector(`[data-map-group="${group.id}"]`);
        if (!node || !rows.some(row => row.group === group.id)) return [];
        if (box.width <= 640 && ['smaller_peer', 'beneficiary'].includes(group.id)) return [];
        const rect = node.getBoundingClientRect();
        let x1 = center.left + center.width / 2 - box.left, y1 = center.top + center.height / 2 - box.top;
        let x2 = rect.left + rect.width / 2 - box.left, y2 = rect.top + rect.height / 2 - box.top;
        if (group.id === 'supplier') { y1 = rect.bottom - box.top; y2 = center.top - box.top; }
        if (group.id === 'customer') { y1 = center.bottom - box.top; y2 = rect.top - box.top; }
        if (group.id === 'smaller_peer') { x1 = rect.right - box.left; x2 = center.left - box.left; }
        if (group.id === 'beneficiary') { x1 = center.right - box.left; x2 = rect.left - box.left; }
        return [{ id: group.id, x1, y1, x2, y2, label: group.link }];
      });
      setDrawing({ width: box.width, height: box.height, lines });
    };
    const observer = new ResizeObserver(draw);
    observer.observe(map);
    map.querySelectorAll('[data-map-group], [data-map-center]').forEach(node => observer.observe(node));
    draw();
    return () => observer.disconnect();
  }, [rows]);
  const node = (entry: typeof rows[number]) => <button type="button" key={entry.key} className="relation-map-node" aria-pressed={entry.key === selected?.key} onClick={() => setSelection(entry.key)}>
    <strong>{String(entry.row.target_company)}</strong><small>{String(entry.row.description || '暂无候选说明')}</small>
  </button>;
  const extra = rows.filter(row => !GROUPS.some(group => group.id === row.group));
  const url = selected ? sourceUrl(selected.row.source) : null;
  return <section className="surface engine-card relationship-map">
    <div className="surface-header"><div><h2>公司产业关系图</h2><span>{ticker} · 根据本次研究返回的 {rows.length} 条关系生成；候选关联不等于已确认事实</span></div></div>
    {rows.length ? <>
      <div className="relation-map-layout" ref={mapRef}>
        <svg className="relation-map-wires" viewBox={`0 0 ${drawing.width || 1} ${drawing.height || 1}`} aria-hidden="true">
          {drawing.lines.map(line => <g key={line.id} className={selected?.group === line.id ? 'selected' : ''}>
            <path d={`M ${line.x1} ${line.y1} L ${line.x2} ${line.y2}`}/>
            <text x={(line.x1 + line.x2) / 2} y={(line.y1 + line.y2) / 2} textAnchor="middle" dominantBaseline="middle">{line.label}</text>
          </g>)}
        </svg>
        {GROUPS.map(group => { const items = rows.filter(row => row.group === group.id); return <section key={group.id} className={`relation-map-group group-${group.id}`} data-map-group={group.id}>
          <h3>{group.title}<small>{items.length} 条</small></h3><span className="relation-map-mobile-label">{group.link}</span>
          {items.length ? <div className="relation-map-nodes">{items.map(node)}</div> : <p className="relation-map-empty">本次暂无候选</p>}
        </section>; })}
        <div className="relation-map-center" data-map-center><strong>{ticker}</strong><span>当前研究公司</span></div>
      </div>
      {extra.length > 0 && <div className="relation-map-extra"><h3>其他关系（不归入四类）</h3><div className="relation-map-nodes">{extra.map(node)}</div></div>}
      <p className="relation-map-legend">虚线表示候选关联；文字表示关系类别，不代表已确认的直接交易。点击公司查看说明和来源。</p>
      <div className="relation-map-detail" aria-live="polite">
        {selected ? <><div className="relation-map-detail-heading"><strong>{ticker} · {String(selected.row.target_company)}</strong><span>{stateLabel(selected.row)}</span>{url ? <a href={url} target="_blank" rel="noreferrer">查看来源 ↗</a> : <span>未保存可追溯来源</span>}</div>
          <p>候选说明：{String(selected.row.description || '暂无说明')}</p>
          {Boolean(selected.row.evidence_text) && <details><summary>查看搜索证据原文</summary><p>{String(selected.row.evidence_text)}</p></details>}
        </> : <p>请选择一个公司节点，查看对应关系及证据状态。</p>}
      </div>
    </> : <p className="insufficient-note">本次未返回可绘制的公司关系，请先运行当前模块。不会填充示例公司。</p>}
  </section>;
}
