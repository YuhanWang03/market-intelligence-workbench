"""Strict SEC boundary: a provider failure must not become a no-filings fact."""
from dataclasses import replace
from v2.agent_v2.agents.filing_reader import EdgarFilingSource, Section


class ItemFilingSource(EdgarFilingSource):
    """Use the SDK's statutory 10-K Item index, rather than visual headings."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._reports = {}

    def outline(self, ref):
        if ref.form not in {"10-K", "10-K/A"}:
            return super().outline(ref)
        if ref.accession not in self._reports:
            self._reports[ref.accession] = ref.raw.obj()
        report = self._reports[ref.accession]
        return [Section(str(item),"Risk Factors" if item == "Item 1A" else str(item),0) for item in report.items]

    def read(self, ref, section_id):
        if ref.form not in {"10-K", "10-K/A"}:
            return super().read(ref,section_id)
        self.outline(ref)
        value = self._reports[ref.accession][section_id]
        if not value:
            raise LookupError(f"SEC Item unavailable: {section_id}")
        return str(value)


def fetch_filings(ticker, form, since, until):
    from v2.sec.client import _ensure_identity, _throttle
    from edgar import Company
    _ensure_identity()
    _throttle()
    rows = Company(ticker).get_filings(form=form, filing_date=f"{since}:{until}")
    return [] if rows is None else list(rows)


def filing_source():
    return ItemFilingSource(fetch=fetch_filings)


def register_history_capabilities(registry, *, fetch=fetch_filings):
    from v2.agent_v2.adapters.history import register_history_capabilities as shared
    shared(registry,filings_fetch=fetch)
    reader = registry.handlers["filings.recent"]
    def strict(arguments, context):
        result = reader(arguments,context)
        if result.errors:
            evidence = [item for item in result.evidence if item.metadata.get("evidence_scope") == "filing"]
            return replace(result,evidence=evidence,summary="SEC 查询未完整完成；不能据此断言没有申报。",metadata={},
                           limitations=[*result.limitations,"Provider errors prevent a complete filing inventory."])
        return result
    registry.register("filings.recent",strict)
