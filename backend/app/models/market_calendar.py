from __future__ import annotations

from datetime import date, time
from typing import Optional

from sqlalchemy import Boolean, Date, Integer, String, Time, JSON, Index
from sqlalchemy.orm import Mapped, mapped_column

from ..db.session import Base


class ExchangeCalendar(Base):
    __tablename__ = "exchange_calendar"

    date: Mapped[date] = mapped_column(Date, primary_key=True)
    is_trading_day: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    session_open: Mapped[Optional[time]] = mapped_column(Time(timezone=False), nullable=True)
    session_close: Mapped[Optional[time]] = mapped_column(Time(timezone=False), nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    __table_args__ = (
        Index("ix_exchange_calendar_is_trading", "is_trading_day"),
    )


class ScheduleChange(Base):
    __tablename__ = "schedule_changes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    effective_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    change_type: Mapped[str] = mapped_column(String(64), nullable=False)
    details_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    source_ref: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)

    __table_args__ = (
        Index("ix_schedule_changes_type", "change_type"),
    )
