"""Public real-provider checks for comparison, dates and broad-market routing."""
import argparse
import json
from pathlib import Path
from v2.agent_v3.acceptance import worker

CASES=[
    {'id':'market','question':'美股大盘最近一周表现如何？','max_seconds':90},
    {'id':'compare','question':'MU和SNDK哪个更值得购买？','max_seconds':180},
    {'id':'orcl','question':'ORCL这周五为什么跌？','max_seconds':120,'web':True},
    {'id':'aapl','question':'苹果最近为什么涨？','max_seconds':120,'web':True},
]

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=[row['id'] for row in CASES],required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--live',action='store_true')
    args=parser.parse_args()
    if not args.live:parser.error('--live required')
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    if (output/(args.case+'.json')).exists():parser.error('Do not overwrite earlier results')
    worker(next(row for row in CASES if row['id']==args.case),output)
