from datetime import datetime, timezone
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import String, Integer, DateTime, Text, ForeignKey, Float

from ..db.session import Base


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    news_id: Mapped[int] = mapped_column(ForeignKey("news.id"), index=True)
    trigger_score: Mapped[float] = mapped_column(Float)
    direction: Mapped[str] = mapped_column(String(16))  # bullish, bearish, neutral
    event_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    horizon: Mapped[str | None] = mapped_column(String(32), nullable=True)
    assets_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    news: Mapped[object] = relationship("News", back_populates="analysis_results")

    def __repr__(self) -> str:
        return f"AnalysisResult(id={self.id}, news_id={self.news_id}, score={self.trigger_score})"
