"""Opt-in public news/SEC end-to-end smoke; model and business ports are live."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

CASES = {
    "news": "核查 NVIDIA 最近一周的一条公司新闻，读取原文，用三句话说明事件、日期和来源；证据不足请说明。",
    "sec": "查阅 NVIDIA 在 2026 年提交的 10-K 原文，找到供应链相关的一项风险，引用原文并用三句话解释，不要把风险说成已发生的事实。",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live",action="store_true",required=True)
    parser.add_argument("--env-file")
    parser.add_argument("--case",choices=["all",*CASES],default="all")
    args = parser.parse_args()
    if args.env_file:
        from dotenv import load_dotenv
        load_dotenv(args.env_file,override=False)
    from v2.agent_v3.runtime import build_workspace_agent
    from v2.agent_v3.graph import AgentV3Config
    root = Path("data/agent_v3/live_sources") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    root.mkdir(parents=True,exist_ok=False)
    from v2.data import cost_ledger
    cost_ledger._DB_PATH = root / "usage.sqlite"
    agent = build_workspace_agent(config=AgentV3Config(enable_web=True,debate=False,max_seconds=180),data_dir=root)
    # Exclude all private account/state and experiment handlers from this smoke.
    for name in list(agent.registry.handlers):
        if name.split(".",1)[0] in {"account","state","lab"}:
            del agent.registry.handlers[name]
    rows=[]
    try:
        for name,question in CASES.items():
            if args.case not in {"all",name}:
                continue
            print("START",name,flush=True)
            result = agent.run(question,session_id=name,allow_web=True,
                on_progress=lambda event: print("NODE",event.message,flush=True))
            rows.append({"case":name,"result":result.to_dict()})
            (root/"report.json").write_text(json.dumps({"conditions":"Live model and public sources; private handlers excluded; debate disabled; isolated sessions.","runs":rows},ensure_ascii=False,indent=2),encoding="utf-8")
            print("END",name,result.status.value,"evidence",len(result.evidence),"ms",result.elapsed_ms,flush=True)
    finally:
        agent.store.close()
        agent.checkpoint_connection.close()
    print(root/"report.json",flush=True)


if __name__ == "__main__":
    main()
