"""Public-only live conversation regression. Status is not a quality score.

python -m v2.agent_v3.batch_acceptance --live --output-dir data/agent_v3/batch-01
Each worker shares only its named synthetic session; no private account/archive data.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from v2.agent_v3.acceptance import worker, write


def case(cid, question, session=None, targets=(), web=True, page_context=None):
    return dict(id=cid, question=question, session=session or cid, targets=list(targets),
                web=web, page_context=page_context,
                expect="检查对象、相关性、事实支持、时间表达及能力边界；完成状态不等于通过")


CASES = [
    case("36", "用yfinance筛选NVDA和AMD，只要求市值大于100亿美元，不获取财报预期。", targets=["NVDA","AMD"],web=False),
    case("37", "对NVDA做财报事件研究，用最近2次财报、yfinance行情，n_bootstrap=100，按surprise分组。", targets=["NVDA"],web=False),
    case("38", "对MU运行投资人委员会，source=tickers，lean=true，只分析这一只。", targets=["MU"],web=False),
    case("31", "查询美国最新CPI同比和环比，给出所属月份与来源。", web=False),
    case("32", "查询SPY前五大持仓与原始权重，说明持仓日期。", targets=["SPY"],web=False),
    case("33", "用yfinance对NVDA做动量策略回测，历史730天，回看60天，跳过5天，持有21天，top_n=1。", targets=["NVDA"],web=False),
    case("34", "查询最新非农就业增量和失业率，标注月份。", web=False),
    case("35", "用yfinance对NVDA和AMD做动量参数扫描：历史730天，回看60天，跳过5天，top_ns=[1]，holding_days_list=[5,21]，near_high_pcts=[null]。", targets=["NVDA","AMD"],web=False),
    case("25", "AMD最近有哪些新闻？", "user-feedback", ["AMD"]),
    case("26", "分析一下MU", "user-feedback", ["MU"]),
    case("27", "AAPL最近为什么涨？", "user-feedback", ["AAPL"]),
    case("28", "帮我整体看看美光这家公司。", targets=["MU"]),
    case("29", "AMD这两周发生了什么值得关注的事？", targets=["AMD"]),
    case("30", "苹果这阵子上涨有什么驱动因素？", targets=["AAPL"]),
    case("01", "最近NVDA有什么新闻？", "news-switch", ["NVDA"], False),
    case("02", "最近NVDA有哪些新闻？", "news-switch", ["NVDA"]),
    case("03", "ORCL为什么下跌？", "news-switch", ["ORCL"]),
    case("04", "只用两句话概括刚才的原因，不要重新查。", "news-switch", ["ORCL"]),
    case("05", "换成AMD，最近有什么新闻？", "news-switch", ["AMD"]),
    case("06", "ORCL为什么下跌？", targets=["ORCL"]),
    case("07", "查询 NVDA 最近收盘价和日期。", "quote-switch", ["NVDA"]),
    case("08", "改查 ORCL 的最近收盘价。", "quote-switch", ["ORCL"]),
    case("09", "把刚才的结果压缩成一句话，不要增加信息。", "quote-switch", ["ORCL"]),
    case("10", "AMD呢？只查最近收盘价。", "quote-switch", ["AMD"]),
    case("11", "ORCL最近有哪些新闻？", targets=["ORCL"], page_context={"section":"research", "selection":{"kind":"research", "ticker":"NVDA"}}),
    case("12", "这只股票最近收盘价是多少？", targets=["AMD"], page_context={"section":"research", "selection":{"kind":"research", "ticker":"AMD"}}),
    case("13", "最近收盘价是多少？", web=False),
    case("14", "为什么跌了？", web=False),
    case("15", "找一条 NVIDIA 最近七天的新闻，给出日期和原文来源。", targets=["NVDA"]),
    case("16", "列出 NVDA 最近七天最重要的三条新闻；不足三条不要凑数。", targets=["NVDA"]),
    case("17", "最近 ORCL 有什么新闻？不要联网。", targets=["ORCL"], web=False),
    case("18", "现在允许联网，重新查 ORCL 最近的新闻。", targets=["ORCL"]),
    case("19", "比较 NVDA 和 AMD 的估值，简短说明数据口径。", targets=["NVDA","AMD"]),
    case("20", "查询最新美国 CPI 同比和所属月份。"),
    case("21", "SPY前五大持仓及权重是多少？", targets=["SPY"]),
    case("22", "回测 ORCL 最近一年的动量策略，给出真实收益率。", targets=["ORCL"]),
    case("23", "只解释自由现金流，不需要实时数据。", web=False),
    case("24", "ORCL 最近一个交易日跌幅是多少？若未下跌请纠正前提。", targets=["ORCL"]),
]


def report(output):
    rows = []
    markdown = ["# 批量测试原始回答与检查", "", "自动检查只核对结构化对象，不判定语义质量。需人工复核全文和证据。", ""]
    for c in CASES:
        path = output / (c["id"] + ".json")
        if not path.exists():
            continue
        r = json.loads(path.read_text(encoding="utf-8"))["result"]
        entities = r.get("request", {}).get("entities", [])
        expected = set(c["targets"])
        flags = []
        if expected and set(entities) != expected:
            flags.append("request_entity_mismatch")
        rows.append(dict(id=c["id"], status=r.get("status"), entities=entities, flags=flags))
        markdown += [f"## {c['id']} · {c['question']}", "",
                     f"session={c['session']} web={c['web']} status={r.get('status')} entities={entities} flags={flags}", "",
                     r.get("answer", ""), ""]
    write(output / "checks.json", rows)
    (output / "answers.md").write_text("\n".join(markdown), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--worker")
    parser.add_argument("--cases", help="Comma-separated case IDs, preserving suite order and session dependencies")
    parser.add_argument("--max-seconds",type=int,default=75,choices=range(30,181),metavar="30..180")
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not args.live:
        parser.error("Explicit --live required")
    if args.worker:
        worker({**next(c for c in CASES if c["id"] == args.worker),"max_seconds":args.max_seconds}, output)
        return
    if (output / "cases.json").exists():
        parser.error("Use a fresh output directory")
    selected = set(args.cases.split(",")) if args.cases else {c["id"] for c in CASES}
    if selected - {c["id"] for c in CASES}:
        parser.error("Unknown case ID")
    cases = [c for c in CASES if c["id"] in selected]
    write(output / "cases.json", cases)
    summary = []
    for c in cases:
        start = time.monotonic()
        with (output / (c["id"] + ".log")).open("w", encoding="utf-8") as log:
            try:
                subprocess.run([sys.executable, "-m", "v2.agent_v3.batch_acceptance", "--live",
                                "--output-dir", str(output), "--worker", c["id"],"--max-seconds",str(args.max_seconds)],
                               stdout=log, stderr=log, timeout=args.max_seconds+20)
                status = "result" if (output / (c["id"] + ".json")).exists() else "worker_error"
            except subprocess.TimeoutExpired:
                status = "process_timeout"
        summary.append(dict(id=c["id"], execution=status, seconds=round(time.monotonic()-start, 2)))
        write(output / "summary.json", summary)
        report(output)
        print(summary[-1], flush=True)


if __name__ == "__main__":
    main()
