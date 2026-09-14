"""Isolated remote V3 app. Uses existing owner auth; never run the local preview."""
import os
from pathlib import Path
import sys

ROOT=Path("/root/market-intelligence-workbench")
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"web/backend"))
from dotenv import load_dotenv
load_dotenv(ROOT/".env",override=False)
from v2.agent_common.llm import OpenAICompatLLM
previous=OpenAICompatLLM()
os.environ.setdefault("AGENT_V3_MODEL",previous.model)
os.environ.setdefault("AGENT_V3_BASE_URL",previous.base_url)
os.environ.setdefault("AGENT_V3_API_KEY",previous.api_key)
if os.environ["AGENT_V3_BASE_URL"].rstrip("/")=="https://api.deepseek.com/v1":
    os.environ.setdefault("AGENT_V3_THINKING","disabled")
from app.config import SETTINGS
if not SETTINGS.owner_token:
    raise RuntimeError("V3 remote service requires existing WEB_OWNER_TOKEN")
from fastapi import FastAPI
from app.routers.agent_v3 import router
app=FastAPI(title="Agent V3")
app.include_router(router)

@app.get("/health")
def health():
    return {"status":"ok","agent":"v3"}
