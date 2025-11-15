from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class FeedbackVerdict(str, Enum):
    agree = "agree"
    disagree = "disagree"


class FeedbackRequest(BaseModel):
    analysis_id: int
    verdict: FeedbackVerdict
    comment: Optional[str] = Field(default=None, max_length=500)
    telegram_id: Optional[str] = None
    init_data: Optional[str] = None


class FeedbackResponse(BaseModel):
    analysis_id: int
    verdict: FeedbackVerdict
    comment: Optional[str] = None
    created_at: datetime
    updated_at: datetime
