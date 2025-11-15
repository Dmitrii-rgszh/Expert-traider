from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
    Index,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..db.session import Base

NUMERIC_18_6 = Numeric(18, 6)
NUMERIC_20_2 = Numeric(20, 2)


class Candle(Base):
    __tablename__ = "candles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    secid: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    board: Mapped[str] = mapped_column(String(16), nullable=False, default="TQBR")
    timeframe: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    open: Mapped[float] = mapped_column(NUMERIC_18_6, nullable=False)
    high: Mapped[float] = mapped_column(NUMERIC_18_6, nullable=False)
    low: Mapped[float] = mapped_column(NUMERIC_18_6, nullable=False)
    close: Mapped[float] = mapped_column(NUMERIC_18_6, nullable=False)
    volume: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    value: Mapped[float | None] = mapped_column(NUMERIC_20_2, nullable=True)
    trades: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        UniqueConstraint("secid", "board", "timeframe", "timestamp", name="uq_candles_series"),
    )


class IndexCandle(Base):
    __tablename__ = "index_candles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    index_code: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    timeframe: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    open: Mapped[float] = mapped_column(NUMERIC_18_6, nullable=False)
    high: Mapped[float] = mapped_column(NUMERIC_18_6, nullable=False)
    low: Mapped[float] = mapped_column(NUMERIC_18_6, nullable=False)
    close: Mapped[float] = mapped_column(NUMERIC_18_6, nullable=False)
    volume: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    value: Mapped[float | None] = mapped_column(NUMERIC_20_2, nullable=True)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        UniqueConstraint("index_code", "timeframe", "timestamp", name="uq_index_candles_series"),
    )


class FeatureWindow(Base):
    __tablename__ = "feature_windows"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    secid: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    timeframe: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    feature_set: Mapped[str] = mapped_column(String(64), nullable=False)
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    checksum: Mapped[str | None] = mapped_column(String(64), nullable=True)

    numeric_features: Mapped[list["FeatureNumeric"]] = relationship(
        back_populates="window",
        cascade="all, delete-orphan",
    )
    categorical_features: Mapped[list["FeatureCategorical"]] = relationship(
        back_populates="window",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        UniqueConstraint(
            "secid",
            "timeframe",
            "window_start",
            "window_end",
            "feature_set",
            name="uq_feature_windows_range",
        ),
        Index(
            "ix_feature_windows_timeframe_feature_set_window_end_secid",
            "timeframe",
            "feature_set",
            "window_end",
            "secid",
        ),
    )


class FeatureNumeric(Base):
    __tablename__ = "feature_numeric"

    feature_window_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("feature_windows.id", ondelete="CASCADE"),
        primary_key=True,
    )
    feature_name: Mapped[str] = mapped_column(String(128), primary_key=True)
    value_numeric: Mapped[float] = mapped_column(Float, nullable=False)

    window: Mapped[FeatureWindow] = relationship(back_populates="numeric_features")


class FeatureCategorical(Base):
    __tablename__ = "feature_categorical"

    feature_window_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("feature_windows.id", ondelete="CASCADE"),
        primary_key=True,
    )
    feature_name: Mapped[str] = mapped_column(String(128), primary_key=True)
    value_text: Mapped[str] = mapped_column(String(256), nullable=False)

    window: Mapped[FeatureWindow] = relationship(back_populates="categorical_features")


class TradeLabel(Base):
    __tablename__ = "trade_labels"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    secid: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    timeframe: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    label_set: Mapped[str] = mapped_column(String(64), nullable=False, default="basic_v1")
    signal_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    horizon_minutes: Mapped[int] = mapped_column(Integer, nullable=False)
    take_profit_pct: Mapped[float] = mapped_column(Float, nullable=False)
    stop_loss_pct: Mapped[float] = mapped_column(Float, nullable=False)
    entry_price: Mapped[float] = mapped_column(NUMERIC_18_6, nullable=False)
    horizon_close: Mapped[float] = mapped_column(NUMERIC_18_6, nullable=False)
    forward_return_pct: Mapped[float] = mapped_column(Float, nullable=False)
    max_runup_pct: Mapped[float] = mapped_column(Float, nullable=False)
    max_drawdown_pct: Mapped[float] = mapped_column(Float, nullable=False)
    label_long: Mapped[bool] = mapped_column(Boolean, nullable=False)
    label_short: Mapped[bool] = mapped_column(Boolean, nullable=False)
    long_pnl_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    short_pnl_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        UniqueConstraint(
            "secid",
            "timeframe",
            "signal_time",
            "horizon_minutes",
            "label_set",
            name="uq_trade_labels_unique",
        ),
        Index(
            "ix_trade_labels_timeframe_label_set_signal_time_secid",
            "timeframe",
            "label_set",
            "signal_time",
            "secid",
        ),
    )
