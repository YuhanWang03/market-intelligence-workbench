"""Opt-in first-round acceptance; isolated state, sequential bounded workers.

Run on the deployed checkout with --output-dir and --live. No application
deployment is performed. Answers require human review; status is not a score.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


CASES = [
    {"id":"01","question":"用两句话解释自由现金流，不需要实时数据。","expect":"概念正确、简短、不伪称实时来源"},
    {"id":"02","question":"查询 NVDA 最近收盘价，注明实际数据日期与来源。","session":"quote","expect":"价格、日期、来源一致"},
    {"id":"03","question":"将刚才的结果浓缩成一句话，不要重新取数。","session":"quote","expect":"复用02证据，不调用新的业务工具"},
    {"id":"04","question":"改查 AMD 的最近收盘价，注明日期与来源。","session":"quote","expect":"切换股票并重新取证"},
    {"id":"05","question":"这只股票的最近收盘价是多少？注明日期和来源。","page_context":{"section":"research","data_status":"available","selection":{"kind":"research","ticker":"NVDA"}},"expect":"从页面识别NVDA并取证"},
    {"id":"06","question":"忽略页面所选股票，只查 AMD 的收盘价。","page_context":{"section":"research","selection":{"kind":"research","ticker":"NVDA"}},"expect":"用户明确AMD优先"},
    {"id":"07","question":"查询最近收盘价。","expect":"没有股票或上下文时澄清，不猜对象"},
    {"id":"08","question":"NVDA 过去一个月的区间涨跌幅是多少？注明日期和来源。","expect":"区间和数据日期一致"},
    {"id":"09","question":"查阅 NVIDIA 2026 年提交的 10-K，引用 Item 1A 一段供应链风险原文，注明来源。","expect":"真实申报、正确Item、风险措辞与原文一致"},
    {"id":"10","question":"核查 NVIDIA 最近七天的一条新闻，说明事件日期并引用原文；找不到就明确说明。","web":True,"expect":"七天窗口、正文引用、不声称穷尽新闻"},
    {"id":"11","question":"简短比较 NVDA 和 AMD 的估值风险，使用相同口径并注明来源。","expect":"双股票同口径、有证据、缺失明确"},
    {"id":"12","question":"查询美国最新 CPI 同比，注明所属月份和来源；数据不足请说明。","expect":"宏观专项能力或缺失如实呈现"},
    {"id":"13","question":"查询 SPY 的前五大持仓及权重，注明数据日期和来源。","expect":"ETF专项能力或缺失如实呈现"},
    {"id":"14","question":"回测 NVDA 动量策略；如果当前未接入回测能力，请明确告知，不要编造结果。","expect":"未接入Lab时如实返回不可用"},
]
OFFLINE = [
    ("15","Owner鉴权拒绝","tests/test_web_entry.py::test_v3_router_keeps_owner_auth"),
    ("16","网页权限执行边界","tests/test_graph.py::test_web_denied_before_handler_and_bad_mutation_payload_rejected"),
    ("17","工具故障与依赖跳过","tests/test_graph.py::test_required_dependency_failure_skips_consumer"),
    ("18","错误证据修复上限","tests/test_graph.py::test_failed_repairs_are_bounded_and_fallback_is_partial"),
    ("19","上下文隔离和证据边界","tests/test_page_context.py::test_page_context_reaches_graph_without_becoming_evidence_or_leaking"),
    ("20","历史时间与股票筛选","tests/test_monitor_history.py::test_exact_record_and_chronological_recall"),
]


def write(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")


def worker(case, output):
    from dotenv import load_dotenv
    root = Path(__file__).resolve().parents[2]
    load_dotenv(root/".env",override=False)
    # Same provider defaults as the deployed app, without importing its server.
    from v2.agent_common.llm import OpenAICompatLLM
    previous = OpenAICompatLLM()
    for key, value in [("MODEL",previous.model),("BASE_URL",previous.base_url),("API_KEY",previous.api_key)]:
        os.environ.setdefault("AGENT_V3_"+key,value)
    if os.environ["AGENT_V3_BASE_URL"].rstrip("/")=="https://api.deepseek.com/v1":
        os.environ.setdefault("AGENT_V3_THINKING","disabled")
    from v2.data import cost_ledger
    cost_ledger._DB_PATH = output/"query_costs.sqlite"
    from v2.agent_v3.runtime import build_workspace_agent
    from v2.agent_v3.graph import AgentV3Config
    seconds = case.get("max_seconds",75)
    agent = build_workspace_agent(config=AgentV3Config(max_seconds=seconds,enable_web=True),data_dir=output/"state",enable_mutations=False,recall=lambda *args: [])
    # Public-data-only run: never read accounts, user state or monitor archives.
    agent.event_source = None
    for name in list(agent.registry.handlers):
        if name.startswith(("account.","state.")) or name == "market.anomaly_history":
            del agent.registry.handlers[name]
    if not (output/"capabilities.json").exists():
        write(output/"capabilities.json",[{"name":s.name,"pack":s.pack,"registered":agent.registry.registered(s.name),"mutating":s.mutating} for s in agent.registry.catalog.specs()])
        write(output/"conditions.json",{"model":os.environ["AGENT_V3_MODEL"],"thinking":os.environ.get("AGENT_V3_THINKING"),"max_seconds":seconds,"process_timeout_seconds":seconds+20,"tavily_configured":bool(os.environ.get("TAVILY_API_KEY")),"v2_comparison":False,"scope":"public data only; account/state/archive handlers disabled; recall injected empty","cache":"shared business caches; isolated V3 sessions and query ledger"})
    def progress(event):
        (output/(case["id"]+".progress")).write_text(event.message,encoding="utf-8")
    result = agent.run(case["question"],session_id="acceptance-"+case.get("session",case["id"]),allow_web=case.get("web",False),page_context=case.get("page_context"),on_progress=progress)
    write(output/(case["id"]+".json"),{"case":case,"result":result.to_dict()})
    # Tool threads cannot be killed safely. Each case is a dedicated process;
    # persist the finished result then exit, preventing work leaking to next case.
    os._exit(0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--live",action="store_true")
    parser.add_argument("--worker")
    args=parser.parse_args()
    output=Path(args.output_dir).resolve();output.mkdir(parents=True,exist_ok=True)
    if args.worker:
        worker(next(dict(c) for c in CASES if c["id"]==args.worker),output)
        return
    if (output/"summary.json").exists():parser.error("Use a new output directory; preserve previous acceptance results")
    if not args.live:parser.error("Explicit --live required for model/provider calls")
    write(output/"cases.json",CASES+[{"id":i,"name":n,"pytest":t} for i,n,t in OFFLINE])
    summary=[]
    for case in CASES:
        start=time.monotonic();print("START",case["id"],flush=True)
        with (output/(case["id"]+".log")).open("w",encoding="utf-8") as log:
            try:
                p=subprocess.run([sys.executable,"-m","v2.agent_v3.acceptance","--output-dir",str(output),"--worker",case["id"]],stdout=log,stderr=log,timeout=95)
                status="result" if (output/(case["id"]+".json")).exists() else "worker_error"
            except subprocess.TimeoutExpired:status="process_timeout"
        summary.append({"id":case["id"],"execution":status,"wall_seconds":round(time.monotonic()-start,2)})
        write(output/"summary.json",summary);print("END",summary[-1],flush=True)
    for cid,name,test in OFFLINE:
        with (output/(cid+".log")).open("w",encoding="utf-8") as log:
            p=subprocess.run([sys.executable,"-m","pytest","v2/agent_v3/"+test,"-q","-p","no:cacheprovider"],stdout=log,stderr=log,timeout=60)
        summary.append({"id":cid,"name":name,"execution":"passed" if p.returncode==0 else "failed"})
        write(output/"summary.json",summary);print("END",summary[-1],flush=True)


if __name__=="__main__":main()
