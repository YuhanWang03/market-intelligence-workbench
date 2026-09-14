"""Bounded browser context for reference resolution, never trusted evidence."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class PageSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["position", "anomaly", "price_alert", "research"]
    ticker: str = Field(default="", max_length=32)
    record_id: str = Field(default="", max_length=100)
    occurred_at: str = Field(default="", max_length=80)
    excerpt: str = Field(default="", max_length=4000)


class PageContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    section: Literal["core", "research", "lab", "cost"]
    label: str = Field(default="", max_length=200)
    tool: str = Field(default="", max_length=80)
    captured_at: str = Field(default="", max_length=80)
    data_status: Literal["available", "unavailable"] = "unavailable"
    selection: PageSelection | None = None
