"""Opt-in public paraphrases for acceptance fixes; same isolated worker."""
import argparse
from pathlib import Path
import subprocess
import sys
from v2.agent_v3.acceptance import worker, write

CASES = [
    {"id":"V1","question":"帮我看下收盘报多少。","expect":"没有对象时澄清，不猜账户"},
    {"id":"V2","question":"最新一份美国 PPI 数据是多少？","expect":"稳定发布标识规范化，未接入时明确告知"},
    {"id":"V3","question":"列出 VOO 权重最高的五只成分股。","expect":"识别ETF成分数据，不执行普通股票研究"},
    {"id":"V4","question":"只告诉我 NVDA 的收盘价和日期，注明来源。","session":"paraphrase","expect":"为改写题建立事实范围"},
    {"id":"V5","question":"刚才那段换种说法写短点，别加新内容，也别重新查。","session":"paraphrase","expect":"只改写上一答，不扩展证据"},
]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--live",action="store_true")
    parser.add_argument("--worker")
    args=parser.parse_args();output=Path(args.output_dir).resolve();output.mkdir(parents=True,exist_ok=True)
    if args.worker:
        worker(next(dict(c) for c in CASES if c["id"]==args.worker),output)
        return
    if not args.live or (output/"summary.json").exists():parser.error("Use --live and a fresh output directory")
    write(output/"cases.json",CASES);summary=[]
    for case in CASES:
        with (output/(case["id"]+".log")).open("w",encoding="utf-8") as log:
            try:
                proc=subprocess.run([sys.executable,"-m","v2.agent_v3.repair_variants","--output-dir",str(output),"--worker",case["id"]],stdout=log,stderr=log,timeout=95)
                status="result" if (output/(case["id"]+".json")).exists() else "worker_error"
            except subprocess.TimeoutExpired:status="process_timeout"
        summary.append({"id":case["id"],"execution":status});write(output/"summary.json",summary);print(summary[-1],flush=True)


if __name__=="__main__":main()
