from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Sequence

from backend.scripts.ingestion.base import logger
from backend.scripts.ingestion.candles import CandleIngestor, INTERVAL_MAP, parse_date

DEFAULT_CONFIG = Path("config/universe_core.json")
DEFAULT_EXPORT_DIR = Path("data/raw/candles")


def load_secids(secids: Sequence[str], secids_file: Path | None) -> list[str]:
    collected: list[str] = [sec.upper() for sec in secids]
    excluded: set[str] = set()
    if secids_file and secids_file.exists():
        payload = json.loads(secids_file.read_text(encoding="utf-8"))
        file_secids = (
            payload.get("core_equities")
            or payload.get("secids")
            or (payload.get("universe") or {}).get("equities")
            or []
        )
        excluded.update(payload.get("exclude") or [])
        excluded.update(payload.get("intraday_exclude") or [])
        excluded.update(payload.get("intraday_blacklist") or [])
        collected.extend(sec.upper() for sec in file_secids)
    unique = sorted({sec for sec in collected if sec and sec not in excluded})
    if not unique:
        raise ValueError("No tickers provided. Specify --secids or --secids-file")
    return unique


def resolve_timeframes(requested: Sequence[str] | None, secids_file: Path | None) -> list[str]:
    if requested:
        return [tf for tf in requested if tf]
    if secids_file and secids_file.exists():
        payload = json.loads(secids_file.read_text(encoding="utf-8"))
        config_timeframes = payload.get("default_timeframes")
        if config_timeframes:
            return config_timeframes
    return ["1m", "5m", "15m", "1h", "1d"]


def chunk_ranges(start: date, end: date, chunk_days: int) -> Iterable[tuple[date, date]]:
    cursor = start
    delta = timedelta(days=chunk_days - 1)
    while cursor <= end:
        chunk_end = min(end, cursor + delta)
        yield cursor, chunk_end
        cursor = chunk_end + timedelta(days=1)


def estimate_expected_rows(timeframe: str, start: date, end: date, secid_count: int) -> int:
    interval = INTERVAL_MAP.get(timeframe)
    if not interval:
        raise ValueError(f"Unsupported timeframe '{timeframe}'")
    days = (end - start).days + 1
    total_minutes = days * 24 * 60
    bars_per_ticker = max(1, total_minutes // interval)
    return bars_per_ticker * secid_count


def validate_volume(
    timeframe: str,
    start: date,
    end: date,
    secid_count: int,
    processed: int,
    min_ratio: float,
) -> None:
    expected = estimate_expected_rows(timeframe, start, end, secid_count)
    threshold = int(expected * min_ratio)
    if processed < threshold:
        logger.warning(
            "Low volume detected for %s %s..%s: processed=%s expected~%s (ratio %.2f)",
            timeframe,
            start,
            end,
            processed,
            expected,
            processed / expected if expected else 0.0,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill MOEX candles into DB + parquet storage")
    parser.add_argument("--secids", nargs="*", default=[], help="Explicit list of tickers to backfill")
    parser.add_argument(
        "--secids-file",
        default=str(DEFAULT_CONFIG),
        help="JSON config with core tickers (defaults to config/universe_core.json)",
    )
    parser.add_argument(
        "--timeframes",
        nargs="*",
        default=None,
        help="Timeframes to process (defaults to config or preset list)",
    )
    parser.add_argument("--start-date", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--chunk-days", type=int, default=14, help="Chunk size for pagination")
    parser.add_argument("--board", default="TQBR", help="MOEX board id (default: TQBR)")
    parser.add_argument(
        "--export-dir",
        default=str(DEFAULT_EXPORT_DIR),
        help="Directory for raw parquet dumps (default: data/raw/candles)",
    )
    parser.add_argument(
        "--min-ratio",
        type=float,
        default=0.25,
        help="Minimal processed/expected volume ratio before warning",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not write to DB")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    secids_file = Path(args.secids_file).expanduser()
    secids = load_secids(args.secids, secids_file)
    timeframes = resolve_timeframes(args.timeframes, secids_file)
    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date)
    export_dir = Path(args.export_dir).expanduser()

    logger.info(
        "Starting backfill: %s tickers=%s timeframes=%s %s..%s",
        args.board,
        ",".join(secids),
        ",".join(timeframes),
        start_date,
        end_date,
    )
    ingestor = CandleIngestor(board=args.board)

    for timeframe in timeframes:
        for chunk_start, chunk_end in chunk_ranges(start_date, end_date, max(1, args.chunk_days)):
            logger.info("Backfill chunk %s %s..%s", timeframe, chunk_start, chunk_end)
            stats = ingestor.run(
                secids=secids,
                timeframe=timeframe,
                start_date=chunk_start,
                end_date=chunk_end,
                dry_run=args.dry_run,
                export_dir=export_dir,
            )
            logger.info(
                "Chunk done %s %s..%s processed=%s inserted=%s updated=%s",
                timeframe,
                chunk_start,
                chunk_end,
                stats.processed,
                stats.inserted,
                stats.updated,
            )
            if not args.dry_run:
                validate_volume(timeframe, chunk_start, chunk_end, len(secids), stats.processed, args.min_ratio)


if __name__ == "__main__":
    main()
