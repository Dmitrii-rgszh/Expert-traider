from __future__ import annotations

import argparse
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

import json
import numpy as np
import pandas as pd
import yaml
from sqlalchemy import delete

from backend.app.db.session import SessionLocal
from backend.app.models import TradeLabel
from backend.scripts.features.build_window import load_candles, timeframe_to_minutes


def _ensure_timezone(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build trade labels from candles")
    parser.add_argument("secids", nargs="+", help="Tickers to process")
    parser.add_argument("--timeframe", default="1m", help="Timeframe key (default: 1m)")
    parser.add_argument("--start-date", required=True, help="Signal window start (ISO datetime)")
    parser.add_argument("--end-date", required=True, help="Signal window end (ISO datetime)")
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[60, 240, 1440],
        help="Forward horizons in minutes",
    )
    parser.add_argument(
        "--take-profit",
        type=float,
        default=0.02,
        help="Target return for TP (fraction, e.g. 0.02 = 2%)",
    )
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=0.01,
        help="Drawdown threshold for SL (fraction)",
    )
    parser.add_argument("--label-set", default="basic_v1", help="Label set identifier")
    parser.add_argument(
        "--label-config",
        type=Path,
        default=Path("config/label_sets.yaml"),
        help="Path to label set config (default: config/label_sets.yaml)",
    )
    parser.add_argument(
        "--use-config-params",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Override horizons/TP/SL from config if available (default: true)",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        help="Optional path to write quality summary",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute metrics without writing to the database",
    )
    return parser.parse_args()


def _compute_labels(
    df: pd.DataFrame,
    horizons: Sequence[int],
    bar_minutes: int,
    take_profit: float,
    stop_loss: float,
    start: datetime,
    end: datetime,
) -> list[dict]:
    if df.empty or not horizons:
        return []

    closes = df["close"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    timestamps = df["timestamp"].to_numpy()

    records: list[dict] = []
    for horizon_minutes in horizons:
        if horizon_minutes <= 0:
            continue
        horizon_bars = max(1, math.ceil(horizon_minutes / bar_minutes))
        max_index = len(df) - horizon_bars
        for idx in range(max_index):
            entry_ts = pd.Timestamp(timestamps[idx])
            if entry_ts < start or entry_ts > end:
                continue
            end_idx = idx + horizon_bars
            entry_price = closes[idx]
            if entry_price <= 0:
                continue
            horizon_close = closes[end_idx]
            window_high = np.max(highs[idx + 1 : end_idx + 1]) if end_idx > idx else entry_price
            window_low = np.min(lows[idx + 1 : end_idx + 1]) if end_idx > idx else entry_price
            forward_return = (horizon_close - entry_price) / entry_price
            max_runup = max(0.0, (window_high - entry_price) / entry_price)
            max_drawdown = max(0.0, (entry_price - window_low) / entry_price)

            long_tp_hit = max_runup >= take_profit
            long_sl_hit = max_drawdown >= stop_loss
            label_long = bool(long_tp_hit and not long_sl_hit)

            short_gain = max_drawdown
            short_loss = max_runup
            short_tp_hit = short_gain >= take_profit
            short_sl_hit = short_loss >= stop_loss
            label_short = bool(short_tp_hit and not short_sl_hit)
            long_pnl = _estimate_pnl(long_tp_hit, long_sl_hit, forward_return, take_profit, stop_loss)
            if short_tp_hit and not short_sl_hit:
                short_pnl = take_profit
            elif short_sl_hit and not short_tp_hit:
                short_pnl = -stop_loss
            else:
                short_pnl = -forward_return

            records.append(
                {
                    "signal_time": entry_ts.to_pydatetime(),
                    "horizon_minutes": int(horizon_minutes),
                    "entry_price": float(entry_price),
                    "horizon_close": float(horizon_close),
                    "forward_return_pct": float(forward_return),
                    "max_runup_pct": float(max_runup),
                    "max_drawdown_pct": float(max_drawdown),
                    "label_long": label_long,
                    "label_short": label_short,
                    "long_pnl_pct": float(long_pnl),
                    "short_pnl_pct": float(short_pnl),
                }
            )
    return records


def _persist_labels(
    secid: str,
    timeframe: str,
    label_set: str,
    take_profit: float,
    stop_loss: float,
    rows: list[dict],
) -> int:
    if not rows:
        return 0

    with SessionLocal() as session:
        signal_times = [row["signal_time"] for row in rows]
        min_signal = min(signal_times)
        max_signal = max(signal_times)
        horizons = sorted({row["horizon_minutes"] for row in rows})
        session.execute(
            delete(TradeLabel).where(
                TradeLabel.secid == secid,
                TradeLabel.timeframe == timeframe,
                TradeLabel.label_set == label_set,
                TradeLabel.horizon_minutes.in_(horizons),
                TradeLabel.signal_time >= min_signal,
                TradeLabel.signal_time <= max_signal,
            )
        )
        session.commit()
        inserted = 0
        for row in rows:
            label = TradeLabel(
                secid=secid,
                timeframe=timeframe,
                label_set=label_set,
                signal_time=row["signal_time"],
                horizon_minutes=row["horizon_minutes"],
                take_profit_pct=take_profit,
                stop_loss_pct=stop_loss,
                entry_price=row["entry_price"],
                horizon_close=row["horizon_close"],
                forward_return_pct=row["forward_return_pct"],
                max_runup_pct=row["max_runup_pct"],
                max_drawdown_pct=row["max_drawdown_pct"],
                label_long=row["label_long"],
                label_short=row["label_short"],
                long_pnl_pct=row["long_pnl_pct"],
                short_pnl_pct=row["short_pnl_pct"],
            )
            session.add(label)
            inserted += 1
        session.commit()
    return inserted


def run_pipeline(
    secids: Sequence[str],
    timeframe: str,
    start: datetime,
    end: datetime,
    horizons: Sequence[int],
    take_profit: float,
    stop_loss: float,
    label_set: str,
    dry_run: bool,
) -> tuple[int, list[dict]]:
    if not horizons:
        raise ValueError("At least one horizon must be specified")

    start = _ensure_timezone(start)
    end = _ensure_timezone(end)
    bar_minutes = timeframe_to_minutes(timeframe)
    max_horizon = max(horizons)
    fetch_end = end + timedelta(minutes=max_horizon + bar_minutes)
    total = 0
    overall_rows: list[dict] = []
    for secid in secids:
        candles = load_candles(secid, timeframe, start, fetch_end)
        if candles.empty:
            print(f"{secid}: no candles found for labeling")
            continue
        candles["timestamp"] = pd.to_datetime(candles["timestamp"], utc=True)
        label_rows = _compute_labels(candles, horizons, bar_minutes, take_profit, stop_loss, start, end)
        inserted = 0
        if not dry_run:
            inserted = _persist_labels(secid, timeframe, label_set, take_profit, stop_loss, label_rows)
        total += inserted
        overall_rows.extend(label_rows)
        if dry_run:
            print(f"{secid}: computed {len(label_rows)} labels (dry-run)")
        else:
            print(f"{secid}: inserted {inserted} labels")
    if dry_run:
        print(f"Dry run finished. Computed {len(overall_rows)} labels.")
    else:
        print(f"Done. Total labels: {total}")
    return total, overall_rows


def main() -> None:
    args = parse_args()
    config_horizons = args.horizons
    config_tp = args.take_profit
    config_sl = args.stop_loss
    if args.label_config and args.use_config_params:
        config_entry = _load_label_config(args.label_config, args.label_set)
        if config_entry:
            config_horizons = config_entry.get("horizons", config_horizons)
            config_tp = config_entry.get("take_profit", config_tp)
            config_sl = config_entry.get("stop_loss", config_sl)
            timeframe_override = config_entry.get("timeframe")
            if timeframe_override:
                args.timeframe = timeframe_override
            print(
                f"[label-config] {args.label_set}: horizons={config_horizons}, "
                f"TP={config_tp} SL={config_sl}"
            )
    start = _ensure_timezone(datetime.fromisoformat(args.start_date))
    end = _ensure_timezone(datetime.fromisoformat(args.end_date))
    total, rows = run_pipeline(
        secids=[sec.upper() for sec in args.secids],
        timeframe=args.timeframe,
        start=start,
        end=end,
        horizons=config_horizons,
        take_profit=config_tp,
        stop_loss=config_sl,
        label_set=args.label_set,
        dry_run=args.dry_run,
    )
    summary = _summarize_rows(rows, args.label_set, args.timeframe, start, end, total)
    if summary:
        print("\n=== Label quality summary ===")
        print(summary["text_summary"])
        if args.summary_json:
            args.summary_json.parent.mkdir(parents=True, exist_ok=True)
            args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Summary saved to {args.summary_json}")


def _load_label_config(path: Path, label_set: str) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    return (payload.get("label_sets") or {}).get(label_set, {})


def _estimate_pnl(tp_hit: bool, sl_hit: bool, forward_return: float, take_profit: float, stop_loss: float) -> float:
    if tp_hit and not sl_hit:
        return take_profit
    if sl_hit and not tp_hit:
        return -stop_loss
    return forward_return


def _summarize_rows(
    rows: list[dict],
    label_set: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    total: int,
) -> dict:
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    total_rows = total or len(rows)
    summary: dict = {
        "label_set": label_set,
        "timeframe": timeframe,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "total": total_rows,
        "horizons": {},
    }
    lines = [
        f"Label set={label_set} timeframe={timeframe} rows={total_rows} window={start.date()}..{end.date()}",
    ]
    for horizon in sorted(df["horizon_minutes"].unique()):
        chunk = df[df["horizon_minutes"] == horizon]
        long_ratio = float(chunk["label_long"].mean()) if not chunk.empty else 0.0
        short_ratio = float(chunk["label_short"].mean()) if not chunk.empty else 0.0
        summary["horizons"][int(horizon)] = {
            "rows": int(len(chunk)),
            "long_positive_ratio": long_ratio,
            "short_positive_ratio": short_ratio,
            "long_pnl_mean": float(chunk["long_pnl_pct"].mean()) if not chunk.empty else 0.0,
            "short_pnl_mean": float(chunk["short_pnl_pct"].mean()) if not chunk.empty else 0.0,
        }
        lines.append(
            f"- H={horizon}m rows={len(chunk)} long+= {long_ratio:.2%} short+= {short_ratio:.2%}"
        )
    summary["text_summary"] = "\n".join(lines)
    return summary


if __name__ == "__main__":
    main()
