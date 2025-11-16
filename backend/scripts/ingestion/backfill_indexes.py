"""
Скрипт для загрузки исторических данных индексов (IMOEX, RTSI) в таблицу index_candles.
Использует MOEX ISS API для получения свечей по индексам.

Usage:
    python backend/scripts/ingestion/backfill_indexes.py \
        --indexes IMOEX RTSI \
        --timeframes 5m 1h \
        --start-date 2025-08-18 \
        --end-date 2025-11-16
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

import requests
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from backend.app.db.session import SessionLocal
from backend.app.models.market_data import IndexCandle
from backend.scripts.ingestion.base import logger

# Маппинг таймфреймов в интервалы MOEX API
INTERVAL_MAP = {
    "1m": 1,
    "5m": 10,
    "10m": 2,
    "15m": 3,
    "30m": 4,
    "1h": 7,
    "1d": 24,
}

MOEX_BASE_URL = "https://iss.moex.com/iss"


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def fetch_index_candles(
    index_code: str,
    interval: int,
    start_date: date,
    end_date: date,
) -> list[dict]:
    """
    Запрашивает свечи индекса через MOEX ISS API.
    
    Args:
        index_code: Код индекса (IMOEX, RTSI)
        interval: Интервал MOEX (1=1m, 10=5m, 7=1h, 24=1d)
        start_date: Начальная дата
        end_date: Конечная дата
    
    Returns:
        Список словарей с полями: timestamp, open, high, low, close, volume, value
    """
    url = f"{MOEX_BASE_URL}/engines/stock/markets/index/securities/{index_code}/candles.json"
    params = {
        "from": start_date.strftime("%Y-%m-%d"),
        "till": end_date.strftime("%Y-%m-%d"),
        "interval": interval,
        "start": 0,
    }
    
    all_candles = []
    
    while True:
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if "candles" not in data or not data["candles"]["data"]:
                break
            
            columns = data["candles"]["columns"]
            rows = data["candles"]["data"]
            
            for row in rows:
                record = dict(zip(columns, row))
                # Парсим timestamp (формат ISO 8601)
                ts_str = record.get("begin") or record.get("end")
                if not ts_str:
                    continue
                
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    continue
                
                all_candles.append({
                    "timestamp": ts,
                    "open": float(record["open"]),
                    "high": float(record["high"]),
                    "low": float(record["low"]),
                    "close": float(record["close"]),
                    "volume": int(record.get("volume") or 0),
                    "value": float(record.get("value") or 0.0),
                })
            
            # Проверяем, есть ли ещё данные
            if len(rows) < 500:  # MOEX обычно отдаёт до 500 свечей за запрос
                break
            
            params["start"] += len(rows)
            
        except Exception as e:
            logger.warning(f"Ошибка при загрузке {index_code} interval={interval}: {e}")
            break
    
    return all_candles


def upsert_index_candles(session, index_code: str, timeframe: str, candles: list[dict]) -> int:
    """
    Вставляет/обновляет свечи индекса в таблицу index_candles.
    
    Returns:
        Количество вставленных записей
    """
    if not candles:
        return 0
    
    stmt = sqlite_insert(IndexCandle).values(
        [
            {
                "index_code": index_code,
                "timeframe": timeframe,
                "timestamp": c["timestamp"],
                "open": c["open"],
                "high": c["high"],
                "low": c["low"],
                "close": c["close"],
                "volume": c["volume"],
                "value": c["value"],
                "ingested_at": datetime.now(timezone.utc),
            }
            for c in candles
        ]
    )
    
    stmt = stmt.on_conflict_do_update(
        index_elements=["index_code", "timeframe", "timestamp"],
        set_={
            "open": stmt.excluded.open,
            "high": stmt.excluded.high,
            "low": stmt.excluded.low,
            "close": stmt.excluded.close,
            "volume": stmt.excluded.volume,
            "value": stmt.excluded.value,
            "ingested_at": stmt.excluded.ingested_at,
        },
    )
    
    result = session.execute(stmt)
    session.commit()
    return result.rowcount


def chunk_date_ranges(start: date, end: date, chunk_days: int) -> list[tuple[date, date]]:
    """Разбивает диапазон дат на чанки для постепенной загрузки."""
    ranges = []
    cursor = start
    delta = timedelta(days=chunk_days - 1)
    
    while cursor <= end:
        chunk_end = min(cursor + delta, end)
        ranges.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    
    return ranges


def main():
    parser = argparse.ArgumentParser(description="Загрузка исторических данных индексов MOEX")
    parser.add_argument(
        "--indexes",
        nargs="+",
        default=["IMOEX", "RTSI"],
        help="Коды индексов для загрузки (по умолчанию: IMOEX RTSI)",
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=["5m", "1h"],
        help="Таймфреймы для загрузки (по умолчанию: 5m 1h)",
    )
    parser.add_argument(
        "--start-date",
        type=parse_date,
        required=True,
        help="Начальная дата (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=parse_date,
        required=True,
        help="Конечная дата (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=30,
        help="Размер чанка в днях для постепенной загрузки (по умолчанию: 30)",
    )
    
    args = parser.parse_args()
    
    logger.info(f"Начинаем загрузку индексов: {args.indexes}")
    logger.info(f"Таймфреймы: {args.timeframes}")
    logger.info(f"Период: {args.start_date} до {args.end_date}")
    
    session = SessionLocal()
    
    try:
        for index_code in args.indexes:
            index_code = index_code.upper()
            
            for timeframe in args.timeframes:
                if timeframe not in INTERVAL_MAP:
                    logger.warning(f"Пропуск неизвестного таймфрейма {timeframe}")
                    continue
                
                interval = INTERVAL_MAP[timeframe]
                logger.info(f"Загружаем {index_code} {timeframe} (interval={interval})")
                
                # Разбиваем на чанки для больших диапазонов
                chunks = chunk_date_ranges(args.start_date, args.end_date, args.chunk_days)
                total_inserted = 0
                
                for chunk_start, chunk_end in chunks:
                    logger.info(f"  Чанк: {chunk_start} до {chunk_end}")
                    
                    candles = fetch_index_candles(
                        index_code=index_code,
                        interval=interval,
                        start_date=chunk_start,
                        end_date=chunk_end,
                    )
                    
                    if candles:
                        inserted = upsert_index_candles(session, index_code, timeframe, candles)
                        total_inserted += inserted
                        logger.info(f"    Вставлено/обновлено: {inserted} свечей")
                    else:
                        logger.warning(f"    Нет данных для {index_code} {timeframe} в {chunk_start}..{chunk_end}")
                
                logger.info(f"✓ {index_code} {timeframe}: итого {total_inserted} свечей")
        
        logger.info("Загрузка завершена успешно!")
        
    finally:
        session.close()


if __name__ == "__main__":
    main()
