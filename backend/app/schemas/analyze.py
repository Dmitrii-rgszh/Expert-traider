from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=5)
    telegram_id: Optional[str] = None
    source_url: Optional[str] = None
    title: Optional[str] = None
    init_data: Optional[str] = None


class AnalyzeResponse(BaseModel):
    news_id: int
    analysis_id: int
    trigger_score: float
    direction: str
    event_type: Optional[str] = None
    horizon: Optional[str] = None
    assets: List[str] = Field(default_factory=list)
    summary: Optional[str] = None


class HistoryItem(BaseModel):
    news_id: int
    analysis_id: int
    created_at: datetime
    title: Optional[str] = None
    text: Optional[str] = None
    trigger_score: float
    direction: str
    event_type: Optional[str] = None
    horizon: Optional[str] = None
    assets: List[str] = Field(default_factory=list)
    summary: Optional[str] = None
