from __future__ import annotations

import argparse
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Iterator, Sequence

import pandas as pd

from sqlalchemy import select

from backend.app.models import Candle
from .base import HttpSource, IngestionStats, fetch_json, session_scope, logger

MOEX_URL = "https://iss.moex.com/iss/engines/stock/markets/shares/securities/{secid}/candles.json"
INTERVAL_MAP = {
    "1m": 1,
    "5m": 5,
    "10m": 10,
    "15m": 15,
    "1h": 60,
    "1d": 24,
}


def parse_date(value: str) -> date:
    return datetime.fromisoformat(value).date()


class CandleIngestor:
    def __init__(self, board: str = "TQBR") -> None:
        self.board = board

    def run(
        self,
        secids: Sequence[str],
        timeframe: str,
        start_date: date,
        end_date: date,
        dry_run: bool = False,
        export_dir: Path | None = None,
    ) -> IngestionStats:
        stats = IngestionStats()
        interval = INTERVAL_MAP.get(timeframe)
        if not interval:
            raise ValueError(f"Unsupported timeframe {timeframe}")

        export_path = Path(export_dir).expanduser().resolve() if export_dir else None
        for secid in secids:
            logger.info(
                "Fetching candles for %s timeframe=%s range=%s..%s",
                secid,
                timeframe,
                start_date,
                end_date,
            )
            series = list(
                self._fetch_series(
                    secid=secid,
                    interval=interval,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                )
            )
            stats.processed += len(series)
            if dry_run or not series:
                continue
            if export_path:
                self._export_series(export_path, secid, timeframe, start_date, end_date, series)
            with session_scope() as session:
                for candle in series:
                    stmt = select(Candle).where(
                        Candle.secid == candle["secid"],
                        Candle.board == candle["board"],
                        Candle.timeframe == candle["timeframe"],
                        Candle.timestamp == candle["timestamp"],
                    )
                    existing = session.execute(stmt).scalar_one_or_none()
                    if existing:
                        for field in ("open", "high", "low", "close", "volume", "value", "trades"):
                            setattr(existing, field, candle[field])
                        stats.updated += 1
                    else:
                        session.add(Candle(**candle))
                        stats.inserted += 1
        return stats

    def _export_series(
        self,
        export_dir: Path,
        secid: str,
        timeframe: str,
        start_date: date,
        end_date: date,
        series: list[dict],
    ) -> None:
        if not series:
            return
        target_dir = export_dir / secid.upper() / timeframe
        target_dir.mkdir(parents=True, exist_ok=True)
        file_stem = f"{start_date.isoformat()}_{end_date.isoformat()}"
        parquet_path = target_dir / f"{file_stem}.parquet"
        df = pd.DataFrame(series)
        if df.empty:
            return
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.sort_values("timestamp", inplace=True)
        try:
            df.to_parquet(parquet_path, index=False)
            logger.info("Exported %s rows to %s", len(df), parquet_path)
        except (ImportError, ValueError) as exc:
            fallback_path = target_dir / f"{file_stem}.csv"
            df.to_csv(fallback_path, index=False)
            logger.warning(
                "Parquet export failed (%s). Wrote CSV fallback to %s",
                exc,
                fallback_path,
            )

    def _fetch_series(
        self,
        secid: str,
        interval: int,
        timeframe: str,
        start_date: date,
        end_date: date,
    ) -> Iterator[dict]:
        start = 0
        while True:
            params = {
                "from": start_date.isoformat(),
                "till": end_date.isoformat(),
                "interval": interval,
                "start": start,
                "boardid": self.board,
            }
            source = HttpSource(url=MOEX_URL.format(secid=secid), params=params)
            payload = fetch_json(source)
            if not payload or "candles" not in payload:
                break
            candles_payload = payload["candles"]
            rows: Sequence[Sequence] = candles_payload.get("data", [])
            columns: Sequence[str] = candles_payload.get("columns", [])
            if not rows:
                break
            index = {name: idx for idx, name in enumerate(columns)}
            for row in rows:
                ts_raw = row[index["begin"]]
                ts = datetime.fromisoformat(ts_raw)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                volume_idx = index.get("volume")
                value_idx = index.get("value")
                trades_idx = index.get("trades")
                yield {
                    "secid": secid,
                    "board": self.board,
                    "timeframe": timeframe,
                    "timestamp": ts,
                    "open": row[index["open"]],
                    "high": row[index["high"]],
                    "low": row[index["low"]],
                    "close": row[index["close"]],
                    "volume": row[volume_idx] if volume_idx is not None else None,
                    "value": row[value_idx] if value_idx is not None else None,
                    "trades": row[trades_idx] if trades_idx is not None else None,
                }
            start += len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest MOEX candles into the database")
    parser.add_argument("secids", nargs="+", help="Tickers to load (e.g. SBER GAZP)")
    parser.add_argument("--timeframe", default="1m", help="Timeframe key (default: 1m)")
    parser.add_argument("--start-date", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--board", default="TQBR", help="MOEX board id")
    parser.add_argument("--dry-run", action="store_true", help="Do not write to DB")
    parser.add_argument(
        "--export-dir",
        default=None,
        help="Optional directory for parquet exports (e.g. data/raw/candles)",
    )
    args = parser.parse_args()

    start_dt = parse_date(args.start_date)
    end_dt = parse_date(args.end_date)
    export_dir = Path(args.export_dir).expanduser() if args.export_dir else None
    stats = CandleIngestor(board=args.board).run(
        secids=args.secids,
        timeframe=args.timeframe,
        start_date=start_dt,
        end_date=end_dt,
        dry_run=args.dry_run,
        export_dir=export_dir,
    )
    logger.info("Candle ingestion finished: %s", stats)


if __name__ == "__main__":
    main()
