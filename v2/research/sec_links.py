"""Resolve SEC filing index URLs to verified primary documents and TOC anchors."""
from v2.usage_context import ContextExecutor as ThreadPoolExecutor
import json
from pathlib import Path
import re
import sqlite3
from urllib.parse import urljoin, urlparse, parse_qs, unquote, quote

from bs4 import BeautifulSoup

DB_PATH = Path(__file__).resolve().parents[2] / 'data' / 'sec_document_links.db'


def archive_url(value, base='https://www.sec.gov'):
    if not isinstance(value, str) or not value:
        return None
    url = urljoin(base, value)
    parsed = urlparse(url)
    if parsed.scheme != 'https' or parsed.hostname != 'www.sec.gov' or parsed.port not in (None, 443):
        return None
    if parsed.path in ('/ixviewer/doc/action', '/ixviewer/doc/action/', '/ixviewer/ix.html', '/ix'):
        return archive_url(parse_qs(parsed.query).get('doc', [''])[0])
    return url if parsed.path.startswith('/Archives/edgar/data/') else None


def primary_document(html, index_url, form):
    soup = BeautifulSoup(html, 'html.parser')
    for row in soup.select('table.tableFile tr'):
        cells = row.find_all('td', recursive=False)
        if len(cells) < 4 or cells[3].get_text(strip=True).upper() != form.upper():
            continue
        for link in cells[2].find_all('a', href=True):
            url = archive_url(link['href'], index_url)
            if url and urlparse(url).path.lower().endswith(('.htm', '.html')):
                return url
    return None


def section_anchors(html):
    soup = BeautifulSoup(html, 'html.parser')
    ids = {str(tag.get('id') or tag.get('name')) for tag in soup.find_all(attrs={'id': True}) + soup.find_all('a', attrs={'name': True})}
    anchors = {}
    for link in soup.find_all('a', href=True):
        href = link['href']
        if not href.startswith('#') or unquote(href[1:]) not in ids:
            continue
        text = ' '.join(link.stripped_strings)
        row = link.find_parent('tr')
        if row is not None:
            row_text = ' '.join(row.stripped_strings)
            if len(row_text) < 350:
                text = row_text
        # Match explicit TOC labels only; never guess an anchor from a heading.
        category = ('RISK' if re.search(r'\brisk\s+factors\b', text, re.I) else
                    'MD&A' if re.search(r"management.{0,12}discussion", text, re.I) else
                    'BUSINESS' if re.fullmatch(r'(?:Item\s+1[.\s]*)?Business(?:\s+\d+)?', text.strip(), re.I) else None)
        if category and category not in anchors:
            anchors[category] = quote(unquote(href[1:]), safe='-_.:')
        item = re.match(r'Item\s+(\d+\.\d+)', text, re.I)
        if item: anchors.setdefault(item[1], quote(unquote(href[1:]), safe='-_.:'))
    return anchors


def _download(url):
    from v2.sec.client import _ensure_identity, _throttle
    from edgar.httprequests import download_text
    _ensure_identity()
    _throttle()
    return download_text(url)


def resolve_document(index_url, form):
    url = archive_url(index_url)
    if not url: return {}
    key = f'v2:{url}:{form}'
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH, timeout=10) as conn:
        conn.execute('CREATE TABLE IF NOT EXISTS links (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        cached = conn.execute('SELECT value FROM links WHERE key=?', (key,)).fetchone()
        if cached: return json.loads(cached[0])
    try:
        is_index = bool(re.search(r'-index\.(?:html?|htm)$', urlparse(url).path, re.I))
        html = _download(url)
        document = primary_document(html, url, form) if is_index else url
        if not document: return {}
        result = {'document_url': document, 'anchors': {}}
        try:
            result['anchors'] = section_anchors(_download(document) if is_index else html)
        except Exception:
            # A verified primary document remains useful if chapter resolution fails.
            return result
        with sqlite3.connect(DB_PATH, timeout=10) as conn:
            conn.execute('INSERT OR REPLACE INTO links VALUES (?,?)', (key, json.dumps(result)))
        return result
    except Exception:
        return {}


def enrich_sec_links(result):
    """Add presentation links without changing cached evidence or source_url."""
    sec = result.get('modules', {}).get('sec', {}).get('details', {})
    risks = sec.get('risk_factor_changes', [])[:10]
    findings = sec.get('sec_findings', [])[:10]
    rows = [(row, 'RISK') for row in risks] + [(row, row.get('category', '')) for row in findings]
    guidance = result.get('modules', {}).get('expectations', {}).get('details', {}).get('guidance', [])
    for row in guidance:
        if not row.get('filing_type'):
            match = next((r for r in findings if r.get('source_url') == row.get('source_url')), None)
            if match: row['filing_type'] = match.get('filing_type', '')
        rows.append((row, row.get('source_section', 'MD&A')))
    keys = list(dict.fromkeys((r.get('source_url'), r.get('filing_type', '')) for r, _ in rows if archive_url(r.get('source_url'))))
    with ThreadPoolExecutor(max_workers=3) as pool:
        resolved = dict(zip(keys, pool.map(lambda key: resolve_document(*key), keys)))
    for row, category in rows:
        original = archive_url(row.get('source_url'))
        if not original: continue
        row['index_url'] = original if '-index.' in original else None
        data = resolved.get((row.get('source_url'), row.get('filing_type', '')), {})
        if not data.get('document_url'): continue
        row['document_url'] = data['document_url']
        anchor = data.get('anchors', {}).get(category)
        if anchor: row['section_url'] = data['document_url'] + '#' + anchor
    return result
