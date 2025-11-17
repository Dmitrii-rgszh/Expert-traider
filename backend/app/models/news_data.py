from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, JSON, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from ..db.session import Base


class NewsSource(Base):
    __tablename__ = "news_sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    provider: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    geo_scope: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    reliability_score: Mapped[Optional[float]] = mapped_column(Numeric(5, 2), nullable=True)
    latency_profile: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    ingest_params: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )


class NewsEvent(Base):
    __tablename__ = "news_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_id: Mapped[Optional[int]] = mapped_column(ForeignKey("news_sources.id"), nullable=True)
    external_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    secid: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    title: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    tags: Mapped[Optional[list[str]]] = mapped_column(JSON, nullable=True)
    raw_payload: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    __table_args__ = (
        Index("ix_news_events_published_at", "published_at"),
        Index("ix_news_events_secid_published_at", "secid", "published_at"),
    )


class GlobalRiskEvent(Base):
    __tablename__ = "global_risk_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    geo_region: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    severity: Mapped[int] = mapped_column(Integer, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source_event_id: Mapped[Optional[int]] = mapped_column(ForeignKey("news_events.id"), nullable=True)
    affected_sectors: Mapped[Optional[list[str]]] = mapped_column(JSON, nullable=True)
    expected_impact_json: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)


class RiskAlert(Base):
    __tablename__ = "risk_alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    secid: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    sector_code: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    risk_level: Mapped[str] = mapped_column(String(32), nullable=False)
    trigger_reason: Mapped[str] = mapped_column(String(128), nullable=False)
    linked_global_event_id: Mapped[Optional[int]] = mapped_column(ForeignKey("global_risk_events.id"), nullable=True)
    decay_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_risk_alerts_timestamp", "timestamp"),
        Index("ix_risk_alerts_secid_timestamp", "secid", "timestamp"),
    )


class SanctionEntity(Base):
    __tablename__ = "sanction_entities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    entity_name: Mapped[str] = mapped_column(String(256), nullable=False)
    list_code: Mapped[str] = mapped_column(String(64), nullable=False)
    listed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")
    source: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)


class SanctionLink(Base):
    __tablename__ = "sanction_links"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    entity_id: Mapped[int] = mapped_column(ForeignKey("sanction_entities.id"), nullable=False)
    secid: Mapped[str] = mapped_column(String(32), nullable=False)
    confidence: Mapped[Optional[float]] = mapped_column(Numeric(5, 2), nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    linked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    __table_args__ = (
        {"sqlite_autoincrement": True},
    )
