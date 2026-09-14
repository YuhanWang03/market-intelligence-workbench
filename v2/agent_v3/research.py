"""Preserve comparison provenance without inventing dates or provider links."""
from dataclasses import replace


def comparison_signature(item):
    """Missing basis is unknown, never an implicit match."""
    fields=(item.period,item.metadata.get('period_start'),item.metadata.get('period_end'),item.unit,item.metadata.get('accounting_basis'),item.metadata.get('measurement_window'))
    return fields if all(fields) and item.source_url else None


def attach_comparison_sources(envelope):
    # Compare only original financial rows with identical actual date windows.
    from v2.agent_v2.models import EvidenceItem
    import hashlib
    groups = {}
    for item in envelope.evidence:
        if item.metadata.get('original_financial_fact'):
            groups.setdefault((item.metric, comparison_signature(item)), {})[item.entity] = item
    rows = []
    for (metric, signature), entities in groups.items():
        if signature and len(entities) >= 2:
            components = list(entities.values())
            key = hashlib.sha256('|'.join(row.id for row in components).encode()).hexdigest()[:16]
            rows.append(EvidenceItem('sec-compare-' + key, envelope.subject,
                '相同财务期间指标对照：' + '；'.join(row.claim for row in components),
                metadata={'comparison': True, 'from': [row.id for row in components]}))
    envelope = replace(envelope, evidence=[*envelope.evidence, *rows])
    by_id = {item.id: item for item in envelope.evidence}
    enriched = []
    for item in envelope.evidence:
        if not item.metadata.get("comparison"):
            enriched.append(item)
            continue
        sources = []
        for key in item.metadata.get("from", []):
            origin = by_id.get(key)
            if origin:
                sources.append({"evidence_id": key, "entity": origin.entity,
                                "period": origin.period, "as_of": origin.as_of,
                                "source_title": origin.source_title, "source_url": origin.source_url})
        components=[by_id[key] for key in item.metadata.get('from',[]) if key in by_id]
        signatures=[comparison_signature(row) for row in components]
        allowed=len(components)>=2 and len(components)==len(item.metadata.get('from',[])) and all(signatures) and len(set(signatures))==1
        enriched.append(replace(item, as_of="", source_title="Research Engine 指标对照", metadata={**item.metadata, "generated_at":envelope.as_of, "source_components": sources,
                                                "periods_verified_equal": allowed,"financial_comparison_allowed":allowed}))
    enriched=[replace(item,metadata={**item.metadata,'exclude_from_comparison':True}) if (item.metadata.get('comparison') and not item.metadata.get('financial_comparison_allowed')) or (item.metric and not comparison_signature(item)) or item.metadata.get('citation_kind')=='metrics' else item for item in enriched]
    return replace(envelope, evidence=enriched)


def audit_research_sources(envelope):
    from v2.agent_v2.models import ResultStatus
    gaps=[]
    for item in envelope.evidence:
        if item.metadata.get('comparison') and not item.metadata.get('financial_comparison_allowed'):
            gaps.append(f'{item.id}: comparison period/unit/accounting basis not aligned')
        if item.metric and not item.metadata.get("comparison"):
            if not item.period:
                gaps.append(f"{item.id}: financial period unavailable")
            if not item.source_url:
                gaps.append(f"{item.id}: original source URL unavailable")
    if not gaps:
        return envelope
    return replace(envelope,status=ResultStatus.PARTIAL_DATA,limitations=[*envelope.limitations,"部分指标缺少财务期间或原始来源，不应视为已核验同期数据。"],metadata={**envelope.metadata,"provenance_gaps":gaps,"answer_constraints":[*envelope.metadata.get('answer_constraints',[]),{'forbid_claim':'根据缺少期间、单位或会计口径核对的财务指标，判断某公司更便宜、盈利更强、基本面接近或更值得买。','warning':'未对齐财务口径不能生成优劣结论；请保留已核实的行情对比及数据缺口。'}]})


def register_research_capabilities(registry):
    from v2.agent_v3.financial_facts import enrich_financials
    from v2.agent_v2.adapters.research import register_research_capabilities as shared
    shared(registry)
    stock = registry.handlers["research.stock"]
    registry.register("research.stock", lambda args, context: audit_research_sources(enrich_financials(stock(args,context),context)))
    compare = registry.handlers["research.compare"]
    registry.register("research.compare", lambda args, context: audit_research_sources(attach_comparison_sources(enrich_financials(compare(args, context),context))))
