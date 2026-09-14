"""Opt-in live specialist evaluation with synthetic, local source ports."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from v2.agent_v2.agents.filing_reader import FilingRef, Section
from v2.agent_v2.models import ExecutionPlan, NormalizedRequest, PlanTask, RouteKind
from v2.agent_v3.context import RunContext
from v2.agent_v3.contracts import plain
from v2.agent_v3.specialists import register_specialists
from v2.agent_v3.tools import Registry

NEWS = "Synthetic fixture. On September 10, 2026, EXMP announced a pilot supply agreement. The company did not disclose financial terms. This announcement does not establish why the stock price moved."
FILING = "Synthetic filing. On September 9, 2026, EXMP entered into a pilot supply agreement. The agreement may be terminated with thirty days of written notice. No termination has occurred."


class FixtureFilings:
    def list_filings(self, ticker, since, until, forms=None):
        return [FilingRef("EXMP","8-K","2026-09-11","synthetic",url="https://example.org/filing")]

    def outline(self, ref):
        return [Section("s1","Material agreement",len(FILING))]

    def read(self, ref, section_id):
        return FILING if section_id == "s1" else ""


def search(query, *, days, max_results):
    return [{"url":"https://example.org/news","title":"Synthetic company announcement","content":"A pilot agreement was announced.","raw_content":NEWS,"published_date":"2026-09-11"}]


CASES = (
    ("news", "web.research", {"query":"EXMP pilot agreement","topic":"news"}, "核查合成样例 EXMP 的协议新闻，读取原文后给出事件日期和原文引文。"),
    ("filing", "filings.read_events", {"ticker":"EXMP","since":"2026-09-01"}, "读取合成样例 EXMP 的 8-K，说明协议何时签署以及解除条款；不要把可解除说成已解除。"),
    ("attribution", "market.explain_move", {"ticker":"EXMP"}, "调查合成样例 EXMP 的协议新闻是否能证明股价异动原因，不要把时间相关说成因果。"),
)


def evaluate(model):
    rows = []
    for name, capability, arguments, question in CASES:
        registry = Registry()
        register_specialists(registry,model,search=search,filing_source=FixtureFilings())
        run = RunContext("fixture-"+name,time.monotonic()+90)
        result = registry.execute(PlanTask(name,capability,arguments),ExecutionPlan(question,RouteKind.RESEARCH),
                                  NormalizedRequest(question,question,allow_web=True),run)
        rows.append({"case":name,"result":plain(result),"usage":run.usage})
    return {"conditions":"Live model; all search and filing sources synthetic; direct specialist boundary, not full routing or live SEC/search availability.","runs":rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live",action="store_true",required=True)
    parser.add_argument("--env-file")
    args = parser.parse_args()
    if args.env_file:
        from dotenv import load_dotenv
        load_dotenv(args.env_file,override=False)
    from v2.agent_v3.runtime import build_model
    model = build_model()
    report = evaluate(model)
    report["model"] = model.model_name
    root = Path("data/agent_v3/specialists")
    root.mkdir(parents=True,exist_ok=True)
    path = root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+".json")
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    for row in report["runs"]:
        print(row["case"],row["result"]["status"],len(row["result"]["evidence"]),flush=True)
    print(path,flush=True)


if __name__ == "__main__":
    main()
