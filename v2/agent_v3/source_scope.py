"""Enforce validated date and Item constraints without interpreting user text."""
from datetime import date


class ScopedFilings:
    def __init__(self, source, window=None, items=(), forms=()):
        self.source,self.window,self.items=source,window,items
        self.forms=forms

    def list_filings(self,ticker,since,until,forms=None):
        if self.window and self.window["basis"] == "filing":
            # Explicit user bounds override the toolbox's default lookback.
            since,until=self.window["start"],self.window["end"]
        rows=self.source.list_filings(ticker,since,until,self.forms or forms)
        return [ref for ref in rows if since <= ref.filing_date <= until]

    def outline(self,ref):
        sections=self.source.outline(ref)
        return sorted(sections,key=lambda section: section.id not in self.items)

    def read(self,ref,section_id):
        if self.items and section_id not in self.items:
            raise ValueError(f"Read requested SEC Items first: {list(self.items)}")
        return self.source.read(ref,section_id)


def scoped_search(search,window):
    if search is None or not window or window["basis"] not in {"event","publication"}:
        return search
    def run(query,*,days,max_results):
        lookback=max(1,(date.today()-date.fromisoformat(window["start"])).days+1)
        return search(f"{query} {window['basis']} date {window['start']} through {window['end']}",days=lookback,max_results=max_results)
    return run


def scope_rejection(finding, window, items):
    if finding.date:
        try:
            date.fromisoformat(finding.date)
        except ValueError:
            return "invalid_event_date"
    if window and window["basis"] in {"event","publication"}:
        if not finding.date:
            return "missing_event_date"
        if not window["start"] <= finding.date <= window["end"]:
            return "event_date_out_of_scope"
    if items and ":" in finding.source:
        return "" if finding.source.split(":",1)[1] in items else "wrong_sec_item"
    return "wrong_sec_item" if items else ""


def finding_in_scope(finding,window,items):
    return not scope_rejection(finding,window,items)
