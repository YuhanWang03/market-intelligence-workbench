"""Opt-in real-model evaluation against fixed, explicitly synthetic evidence."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from v2.agent_v2.models import EvidenceItem, ToolEnvelope, ResultStatus
from v2.agent_v3.brain import ModelBrain
from v2.agent_v3.graph import AgentV3, AgentV3Config
from v2.agent_v3.tools import Registry


CASES = (
    {"id":"quote", "question":"仅用一句话告诉我 NVDA 在给定数据中的收盘价、日期和来源。", "style":"brief", "max_chars":350},
    {"id":"paraphrase", "question":"我只想知道 NVDA 收盘多少钱，哪天的数据、从哪里来的，一句话就好。", "style":"brief", "max_chars":350},
    {"id":"restate", "question":"把刚才的结果浓缩复述一下，不必再查。", "style":"brief", "max_chars":350},
)


def fixture():
    return ToolEnvelope("market.performance",ResultStatus.COMPLETED,subject="NVDA",as_of="2026-09-11",
        evidence=[EvidenceItem(id="fixture-close",entity="NVDA",claim="合成评测数据：NVDA 在 2026-09-11 的收盘价为 120 美元。",metric="close",value=120,unit="USD",as_of="2026-09-11",source_id="synthetic",source_title="合成评测数据（非真实行情）",source_url="https://example.org/fixture"),
                  EvidenceItem(id="fixture-volume",entity="NVDA",claim="合成评测数据：NVDA 在 2026-09-11 的成交量为 1000000 股。",metric="volume",value=1000000,as_of="2026-09-11",source_id="synthetic",source_title="合成评测数据（非真实行情）")])


def evaluate(model):
    registry = Registry()
    calls = []
    def read(arguments, context):
        calls.append(dict(arguments))
        return fixture()
    registry.register("market.performance",read)
    agent = AgentV3(registry=registry,brain=ModelBrain(model),config=AgentV3Config(debate=False,max_seconds=60))
    rows = []
    try:
        for case in CASES:
            before = len(calls)
            session = "quote" if case["id"] == "quote" else "paraphrase-and-restate"
            result = agent.run(case["question"],session_id=session)
            intent = agent.store.get(session)["turns"][-1]["intent"]
            rows.append({"case":case,"intent":intent,"result":result.to_dict(),"observations":{
                "characters":len(result.answer),"within_length_budget":len(result.answer)<=case["max_chars"],
                "style_matches":intent.get("response_style")==case["style"],"new_tool_calls":len(calls)-before}})
    finally:
        agent.store.close()
    return {"conditions":"Synthetic fixed evidence, live model; length/style checks are not semantic quality judgments. Human review required for relevance and source fidelity.","runs":rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live",action="store_true",required=True,help="Explicitly allow paid model calls; no market requests")
    parser.add_argument("--env-file")
    parser.add_argument("--output-dir",default="data/agent_v3/quality")
    args = parser.parse_args()
    if args.env_file:
        from dotenv import load_dotenv
        load_dotenv(args.env_file,override=False)
    from v2.agent_v3.runtime import build_model
    model = build_model()
    report = evaluate(model)
    report.update(model=model.model_name,extra_body=model.extra_body)
    root = Path(args.output_dir)
    root.mkdir(parents=True,exist_ok=True)
    output = root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+".json")
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    for row in report["runs"]:
        print(row["case"]["id"],row["result"]["status"],row["observations"],flush=True)
    print(output,flush=True)


if __name__ == "__main__":
    main()
