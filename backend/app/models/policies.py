from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column

from ..db.session import Base


class PolicyRun(Base):
    __tablename__ = "policy_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    policy_name: Mapped[str] = mapped_column(String(64), nullable=False)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    start_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    config_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="running")
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)


class PolicyFeedback(Base):
    __tablename__ = "policy_feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    policy_run_id: Mapped[int] = mapped_column(ForeignKey("policy_runs.id"), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"), nullable=True)
    secid: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    context_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    chosen_action: Mapped[str] = mapped_column(String(64), nullable=False)
    reward: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    reward_components_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
