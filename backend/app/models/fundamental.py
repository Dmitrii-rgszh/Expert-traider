from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Index, Integer, JSON, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from ..db.session import Base


class FundamentalMetric(Base):
    """Stores point-in-time fundamental metrics (earnings, leverage, etc.)."""

    __tablename__ = "fundamental_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    secid: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    metric_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    metric_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    period_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    currency: Mapped[str | None] = mapped_column(String(8), nullable=True)
    source: Mapped[str | None] = mapped_column(String(64), nullable=True)
    reliability_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint("secid", "metric_type", "metric_date", name="uq_fundamental_metric_snapshot"),
        Index("ix_fundamental_metrics_metric_date", "metric_date"),
        Index("ix_fundamental_metrics_metric_date_secid", "metric_date", "secid"),
    )
