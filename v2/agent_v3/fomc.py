"""Read published statements from the Fed calendar, without sentiment heuristics."""
from datetime import date
import re
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from v2.agent_v2.models import EvidenceItem, ToolEnvelope, ResultStatus

CALENDAR = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
# Stable publisher URL grammar; never used to classify user language.
STATEMENT = re.compile(r"/newsevents/pressreleases/monetary(\d{8})a\.htm$")


def read_statements(*, get=None, today=None):
    today = today or date.today()
    if get is None:
        import requests
        def get(url):
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            return response.text
    calendar = BeautifulSoup(get(CALENDAR), "html.parser")
    links = {}
    for link in calendar.select("a[href]"):
        url = urljoin(CALENDAR, link["href"])
        parsed = urlparse(url)
        match = STATEMENT.fullmatch(parsed.path)
        if parsed.hostname != "www.federalreserve.gov" or not match:
            continue
        stamp = date.fromisoformat(match[1])
        if stamp <= today:
            links[stamp.isoformat()] = url
    evidence, limits = [], []
    for stamp in sorted(links, reverse=True)[:2]:
        try:
            document = BeautifulSoup(get(links[stamp]), "html.parser")
            article = document.select_one("#article")
            if article is None:
                raise ValueError("Statement article unavailable")
            paragraphs = [p.get_text(" ", strip=True) for p in article.select("p")]
            text = "\n\n".join(p for p in paragraphs if p)
            if len(text) < 300:
                raise ValueError("Statement body incomplete")
            evidence.append(EvidenceItem(f"fomc-{stamp}", "FOMC", text, as_of=stamp,
                source_id=links[stamp], source_url=links[stamp], source_title="Federal Reserve FOMC statement",
                metadata={"date_basis":"publication", "document_type":"policy_statement", "full_text":True}))
        except Exception as exc:
            limits.append(f"FOMC {stamp}: {type(exc).__name__}")
    if not evidence:
        limits.append("未取得美联储声明正文，不能用利率序列替代声明内容。")
    return ToolEnvelope("macro.release", ResultStatus.PARTIAL_DATA if limits else ResultStatus.COMPLETED,
        subject="FOMC", evidence=evidence, limitations=limits,
        metadata={"statement_count":len(evidence),"source_calendar":CALENDAR})
