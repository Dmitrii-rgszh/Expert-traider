from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from backend.scripts.ingestion.backfill_candles import chunk_ranges, load_secids
from backend.scripts.ingestion.candles import parse_date
from backend.scripts.ingestion.base import logger

DEFAULT_BASE_DIR = Path("data/raw/candles")
TARGET_RULES = {
    "5m": "5min",
    "15m": "15min",
    "1h": "60min",
}
AGGREGATIONS = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
    "value": "sum",
    "trades": "sum",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resample local 1m candle parquet files to higher timeframes")
    parser.add_argument("--secids", nargs="*", default=[], help="Tickers to process (e.g. SBER GAZP)")
    parser.add_argument("--secids-file", default="config/universe_core.json", help="Optional JSON config file")
    parser.add_argument(
        "--include-config-secids",
        action="store_true",
        help="Append tickers from the JSON config (default: use only --secids)",
    )
    parser.add_argument("--source-timeframe", default="1m", help="Base timeframe to resample from (default: 1m)")
    parser.add_argument(
        "--target-timeframes",
        nargs="*",
        default=("5m", "15m"),
        help="Timeframes to generate (default: 5m 15m)",
    )
    parser.add_argument("--start-date", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--chunk-days", type=int, default=5, help="Chunk size aligned with parquet backfill files")
    parser.add_argument(
        "--base-dir",
        default=str(DEFAULT_BASE_DIR),
        help="Directory containing raw parquet files (default: data/raw/candles)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing resampled files")
    return parser.parse_args()


def ensure_rule(timeframe: str) -> str:
    try:
        return TARGET_RULES[timeframe]
    except KeyError as exc:  # pragma: no cover - guarded by CLI choices
        raise ValueError(f"Unsupported target timeframe '{timeframe}'") from exc


def load_chunk_frame(path: Path, chunk_start: date, chunk_end: date) -> pd.DataFrame:
    if not path.exists():
        logger.warning("Missing source parquet %s for %s..%s", path, chunk_start, chunk_end)
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    for column in ("value", "trades"):
        if column not in df.columns:
            df[column] = 0.0
        else:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)
    start_ts = pd.Timestamp(chunk_start).tz_localize("UTC")
    end_ts = pd.Timestamp(chunk_end + timedelta(days=1)).tz_localize("UTC")
    mask = (df["timestamp"] >= start_ts) & (df["timestamp"] < end_ts)
    return df.loc[mask].copy()


def resample_frame(df: pd.DataFrame, secid: str, board: str, target_tf: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    work = df.sort_values("timestamp").set_index("timestamp")
    resampled = work.resample(ensure_rule(target_tf), label="right", closed="right").agg(AGGREGATIONS)
    resampled = resampled.dropna(subset=["close"])
    if resampled.empty:
        return pd.DataFrame()
    resampled.reset_index(inplace=True)
    resampled["secid"] = secid
    resampled["board"] = board
    resampled["timeframe"] = target_tf
    columns = [
        "secid",
        "board",
        "timeframe",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "value",
        "trades",
    ]
    return resampled[columns]


def write_parquet(df: pd.DataFrame, target_path: Path) -> None:
    if df.empty:
        return
    target_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(target_path, index=False)
    logger.info("Wrote %s rows -> %s", len(df), target_path)


def chunk_file(base_dir: Path, secid: str, timeframe: str, chunk_start: date, chunk_end: date) -> Path:
    file_stem = f"{chunk_start.isoformat()}_{chunk_end.isoformat()}"
    return base_dir / secid.upper() / timeframe / f"{file_stem}.parquet"


def gather_secids(args: argparse.Namespace) -> list[str]:
    cli_secids = sorted({sec.upper() for sec in args.secids if sec})
    if args.include_config_secids:
        secids_file = Path(args.secids_file).expanduser()
        return load_secids(cli_secids, secids_file)
    if not cli_secids:
        raise ValueError("No tickers provided. Use --secids or enable --include-config-secids")
    return cli_secids


def main() -> None:
    args = parse_args()
    secids = gather_secids(args)
    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date)
    base_dir = Path(args.base_dir).expanduser()
    targets = [tf for tf in args.target_timeframes if tf]
    if not targets:
        raise ValueError("No target timeframes provided")

    logger.info(
        "Resampling %s tickers from %s to %s for %s..%s",
        len(secids),
        args.source_timeframe,
        ",".join(targets),
        start_date,
        end_date,
    )

    for secid in secids:
        logger.info("Processing %s", secid)
        for chunk_start, chunk_end in chunk_ranges(start_date, end_date, max(1, args.chunk_days)):
            source_path = chunk_file(base_dir, secid, args.source_timeframe, chunk_start, chunk_end)
            chunk_df = load_chunk_frame(source_path, chunk_start, chunk_end)
            if chunk_df.empty:
                continue
            if "board" in chunk_df.columns:
                board_series = chunk_df["board"].dropna()
                board = str(board_series.iloc[0]) if not board_series.empty else "TQBR"
            else:
                board = "TQBR"
            for target_tf in targets:
                target_path = chunk_file(base_dir, secid, target_tf, chunk_start, chunk_end)
                if target_path.exists() and not args.force:
                    continue
                resampled = resample_frame(chunk_df, secid, board, target_tf)
                if resampled.empty:
                    continue
                write_parquet(resampled, target_path)


if __name__ == "__main__":
    main()
