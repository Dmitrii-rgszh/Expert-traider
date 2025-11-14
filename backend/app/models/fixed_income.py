from __future__ import annotations

from datetime import date
from typing import Optional

from sqlalchemy import Date, Integer, Numeric, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from ..db.session import Base

NUMERIC_18_6 = Numeric(18, 6)
NUMERIC_20_2 = Numeric(20, 2)


class OfzYield(Base):
    __tablename__ = "ofz_yields"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    isin: Mapped[str] = mapped_column(String(12), nullable=False)
    maturity_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    ytm: Mapped[Optional[float]] = mapped_column(NUMERIC_18_6, nullable=True)
    dirty_price: Mapped[Optional[float]] = mapped_column(NUMERIC_18_6, nullable=True)
    duration: Mapped[Optional[float]] = mapped_column(NUMERIC_18_6, nullable=True)
    convexity: Mapped[Optional[float]] = mapped_column(NUMERIC_18_6, nullable=True)
    coupon: Mapped[Optional[float]] = mapped_column(NUMERIC_18_6, nullable=True)
    next_coupon_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)

    __table_args__ = (UniqueConstraint("isin", "date", name="uq_ofz_yields_isin_date"),)


class OfzAuction(Base):
    __tablename__ = "ofz_auctions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    auction_date: Mapped[date] = mapped_column(Date, nullable=False)
    isin: Mapped[str] = mapped_column(String(12), nullable=False)
    offered: Mapped[Optional[float]] = mapped_column(NUMERIC_20_2, nullable=True)
    placed: Mapped[Optional[float]] = mapped_column(NUMERIC_20_2, nullable=True)
    yield_min: Mapped[Optional[float]] = mapped_column(NUMERIC_18_6, nullable=True)
    yield_avg: Mapped[Optional[float]] = mapped_column(NUMERIC_18_6, nullable=True)
    yield_max: Mapped[Optional[float]] = mapped_column(NUMERIC_18_6, nullable=True)
    bid_cover: Mapped[Optional[float]] = mapped_column(NUMERIC_18_6, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    __table_args__ = (UniqueConstraint("auction_date", "isin", name="uq_ofz_auctions_isin_date"),)
