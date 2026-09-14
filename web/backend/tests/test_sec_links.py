from v2.research import sec_links as links

INDEX = 'https://www.sec.gov/Archives/edgar/data/1045810/0001045810-26-000075-index.html'
DOC = 'https://www.sec.gov/Archives/edgar/data/1045810/000104581026000075/nvda-20260726.htm'
INDEX_HTML = '''<table class="tableFile"><tr><td>1</td><td>10-Q</td><td><a href="/ix?doc=/Archives/edgar/data/1045810/000104581026000075/nvda-20260726.htm">report</a></td><td>10-Q</td></tr></table>'''
DOC_HTML = '''<a href="#risk">Item 1A. Risk Factors</a><a href="#missing">Business</a><a href="#mda">Management’s Discussion and Analysis</a><div id="risk">Risk Factors</div><div id="mda">Discussion</div>'''


def test_exact_form_and_existing_anchors():
    assert links.primary_document(INDEX_HTML, INDEX, '10-Q') == DOC
    assert links.primary_document(INDEX_HTML, INDEX, '10-K') is None
    assert links.section_anchors(DOC_HTML) == {'RISK': 'risk', 'MD&A': 'mda'}
    assert links.archive_url('https://example.com/report.htm') is None
    assert links.archive_url('javascript:alert(1)') is None


def test_table_of_contents_separate_label_and_page_link():
    html = '<table><tr><td><a href="#r">Item 1A.</a></td><td>Risk Factors</td><td><a href="#r">34</a></td></tr></table><div id="r">Risk Factors</div>'
    assert links.section_anchors(html) == {'RISK': 'r'}


def test_cached_snapshot_links_resolve_once(tmp_path, monkeypatch):
    monkeypatch.setattr(links, 'DB_PATH', tmp_path / 'links.db')
    calls = []
    def download(url):
        calls.append(url)
        return INDEX_HTML if url == INDEX else DOC_HTML
    monkeypatch.setattr(links, '_download', download)
    row = {'source_url': INDEX, 'filing_type': '10-Q'}
    data = {'modules': {'sec': {'details': {'risk_factor_changes': [row]}}}}
    links.enrich_sec_links(data)
    assert row['source_url'] == INDEX
    assert row['index_url'] == INDEX
    assert row['document_url'] == DOC
    assert row['section_url'] == DOC + '#risk'
    links.enrich_sec_links(data)
    assert calls == [INDEX, DOC]


def test_failure_preserves_source(tmp_path, monkeypatch):
    monkeypatch.setattr(links, 'DB_PATH', tmp_path / 'links.db')
    monkeypatch.setattr(links, '_download', lambda _: '<html>Unavailable</html>')
    row = {'source_url': INDEX, 'filing_type': '10-Q'}
    links.enrich_sec_links({'modules': {'sec': {'details': {'sec_findings': [row]}}}})
    assert row['index_url'] == INDEX
    assert 'document_url' not in row
