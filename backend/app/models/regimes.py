from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, Numeric, String, JSON
from sqlalchemy.orm import Mapped, mapped_column

from ..db.session import Base


class MarketRegime(Base):
    __tablename__ = "market_regimes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, nullable=False)
    scope: Mapped[str] = mapped_column(String(32), nullable=False, default="market")
    scope_value: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    value: Mapped[str] = mapped_column(String(64), nullable=False)
    probabilities_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    features_snapshot: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)


class MarketRegimeDetail(Base):
    __tablename__ = "market_regime_details"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, nullable=False)
    scope: Mapped[str] = mapped_column(String(32), nullable=False)
    scope_value: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    base_regime_id: Mapped[int] = mapped_column(ForeignKey("market_regimes.id"), nullable=False)
    liquidity_regime: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    spread_regime: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    news_burst_level: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    derived_from: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    features_snapshot: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)


class StrategyRegimePolicy(Base):
    __tablename__ = "strategy_regime_policies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy_code: Mapped[str] = mapped_column(String(64), nullable=False)
    regime_pattern: Mapped[str] = mapped_column(String(128), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    weight_adjustment: Mapped[Optional[float]] = mapped_column(Numeric(10, 4), nullable=True)
    rules_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
