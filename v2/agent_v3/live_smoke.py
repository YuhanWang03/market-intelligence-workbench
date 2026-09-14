"""Opt-in, public read-only smoke comparison; never part of offline CI.

Run with explicit V3 model configuration and --env-file if needed. Outputs are
observations, not a quality score. V2 retains its own model protocol defaults.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


QUESTIONS = (
    "市盈率是什么意思？请简短解释，不需要实时数据。",
    "查询 NVDA 最近一个交易日的收盘价，注明日期和来源。",
    "把刚才 NVDA 的结果简短复述一下，不要重新查询。",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file")
    parser.add_argument("--output-dir", default="data/agent_v3/eval")
    parser.add_argument("--max-seconds", type=float, default=90)
    args = parser.parse_args()
    if args.env_file:
        from dotenv import load_dotenv
        load_dotenv(args.env_file, override=False)
    from v2.agent_v3.runtime import build_model
    from v2.agent_v3.brain import ModelBrain
    from v2.agent_v3.graph import AgentV3, AgentV3Config
    from v2.agent_v3.tools import Registry
    from v2.agent_v2.adapters.market import register_market_capabilities
    from v2.agent_v2.catalog import default_catalog
    from v2.agent_v2.execution import CapabilityRegistry
    from v2.agent_v2.intent import IntentClassifier
    from v2.agent_v2.llm import StructuredLLMPlanner, LLMEvidenceSynthesizer
    from v2.agent_v2.orchestrator import AgentV2, AgentV2Config
    from v2.agent_v2.session import ShortTermSession
    from v2.agent_common.llm import OpenAICompatLLM

    model = build_model()
    root = Path(args.output_dir) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    root.mkdir(parents=True, exist_ok=False)
    from v2.data import cost_ledger
    cost_ledger._DB_PATH = root / "query_costs.sqlite"
    # Register only public market reads. No account, state, Lab or web handlers.
    registry3 = Registry()
    from v2.agent_v3.market import register_market_capabilities as register_v3_market
    register_v3_market(registry3)
    v3 = AgentV3(registry=registry3, brain=ModelBrain(model), config=AgentV3Config(max_seconds=args.max_seconds, debate=False))
    catalog = default_catalog()
    registry2 = CapabilityRegistry(catalog)
    register_market_capabilities(registry2)
    llm = OpenAICompatLLM(model=model.model_name, base_url=str(model.openai_api_base), api_key=model.openai_api_key.get_secret_value(), timeout=45, max_retries=1)
    v2 = AgentV2(catalog=catalog, registry=registry2, planner=StructuredLLMPlanner(llm,catalog),
                 synthesizer=LLMEvidenceSynthesizer(llm,catalog=catalog), classifier=IntentClassifier(llm),
                 session=ShortTermSession(), config=AgentV2Config(max_seconds=args.max_seconds, debate=False,
                 record_intents=False, record_sub_agents=False, record_capabilities=False))
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "model": model.model_name,
              "v3_extra_body": model.extra_body, "conditions": "Public market handlers only; web/debate disabled; V2 provider defaults; sequential runs may share market caches; not a controlled quality benchmark.", "runs": []}
    try:
        for name, agent in (("v3", v3), ("v2", v2)):
            for index, question in enumerate(QUESTIONS):
                print(f"START {name} case={index+1}", flush=True)
                result = agent.run(question, session_id=f"smoke-{name}")
                report["runs"].append({"version":name,"case":index+1,"question":question,"result":result.to_dict()})
                (root / "report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
                print(f"END {name} case={index+1} status={result.status.value} elapsed_ms={result.elapsed_ms} evidence={len(result.evidence)}",flush=True)
    finally:
        v3.store.close()
    print(f"REPORT {root / 'report.json'}",flush=True)


if __name__ == "__main__":
    main()
