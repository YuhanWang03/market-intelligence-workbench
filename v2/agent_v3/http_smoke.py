"""Explicit public-question HTTP smoke; no account/page context is sent."""
import argparse
import json
import os
from pathlib import Path
import time
import uuid
import httpx
from dotenv import dotenv_values


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--verify-restart",action="store_true")
    parser.add_argument("--question",default="用一句话解释夏普比率，不查询实时数据。")
    parser.add_argument("--allow-web",action="store_true")
    args=parser.parse_args()
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    token=os.environ.get("WEB_OWNER_TOKEN") or dotenv_values("/etc/hedge-fund/web.env").get("WEB_OWNER_TOKEN")
    if not token:
        raise RuntimeError("Owner token unavailable")
    base="http://127.0.0.1:8104/api/agent-v3"
    with httpx.Client(headers={"X-Owner-Token":token},timeout=15) as client:
        if args.verify_restart:
            initial=json.loads((output/"job.json").read_text())
            result=client.get(base+"/jobs/"+initial["job_id"]);result.raise_for_status()
            assert result.json()["result"]["run_id"]==initial["result"]["run_id"]
            print("Completed job survived service restart")
            return
        response=client.post(base+"/ask",json={"text":args.question,"session_id":"http-smoke-"+uuid.uuid4().hex,"allow_web":args.allow_web,"background":True})
        response.raise_for_status();job=response.json()
        for _ in range(400):
            response=client.get(base+"/jobs/"+job["job_id"]);response.raise_for_status();job=response.json()
            if job["status"]!="running":break
            time.sleep(.5)
        assert job["status"]=="completed", job.get("error")
        assert job["result"]["synthesis"]["framework"]=="langgraph"
        (output/"job.json").write_text(json.dumps(job,ensure_ascii=False,indent=2),encoding="utf-8")
        print("HTTP authenticated job completed; V3 graph confirmed")


if __name__=="__main__":main()
