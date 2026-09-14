"""V3 provenance boundary around the shared structured market calculation."""
from dataclasses import replace
from urllib.parse import quote


def source_descriptor(source, ticker):
    from v2.data.price_source import YFinancePriceSource, FDPriceSource
    if type(source) is YFinancePriceSource:
        symbol = quote(source.yfinance_symbol(ticker), safe="")
        return {"provider":"Yahoo Finance via yfinance", "url":f"https://finance.yahoo.com/quote/{symbol}/history/"}
    if type(source) is FDPriceSource:
        return {"provider":"Financial Datasets", "url":"https://api.financialdatasets.ai/prices/"}
    return {"provider":"", "url":""}


def register_market_capabilities(registry, *, price_source_factory=None, now_factory=None):
    from v2.agent_v2.adapters.market import register_market_capabilities as register_shared, _performance_envelope, _now_et
    from v2.data.price_source import default_price_source
    factory = price_source_factory or default_price_source
    clock = now_factory or _now_et
    register_shared(registry, price_source_factory=factory, now_factory=clock)

    def performance(arguments, context):
        ticker = arguments["ticker"].upper()
        source = factory()
        current = clock()
        if arguments.get("_as_of"):
            from datetime import date
            selected = date.fromisoformat(arguments["_as_of"])
            current = current.replace(year=selected.year, month=selected.month, day=selected.day, hour=23, minute=59)
        observed={}
        class RecordingSource:
            upstream=source
            def get_prices(self,symbol,*args,**kwargs):
                rows=source.get_prices(symbol,*args,**kwargs)
                observed[symbol]=rows
                return rows
            def __getattr__(self,name):return getattr(source,name)
        result = _performance_envelope(ticker, context, RecordingSource(), now=current)
        from v2.agent_v2.adapters.market import _usable
        bars=_usable(observed.get(ticker,[]))
        windows={key:{'start':str(bars[-1-count].time)[:10],'end':str(bars[-1].time)[:10],'sessions':count} for key,count in [('1d',1),('5d',5),('1m',21),('3m',63),('1y',252)] if len(bars)>count}
        evidence = []
        for item in result.evidence:
            descriptors = [source_descriptor(source, ticker)]
            if benchmark := item.metadata.get("benchmark"):
                descriptors.append(source_descriptor(source, benchmark))
            main = descriptors[0]
            window_note=' '.join(f"{key}收益起止{window['start']}至{window['end']}。" for key,window in windows.items()) if item.metadata.get('returns') else ''
            evidence.append(replace(item, claim=item.claim+window_note, source_title=main["provider"] or "行情供应商未提供", source_url=main["url"],
                metadata={**item.metadata,"return_windows":windows,"sources":descriptors,"provenance_known":bool(main["provider"])}))
        return replace(result, evidence=evidence)

    registry.register("market.performance", performance)
