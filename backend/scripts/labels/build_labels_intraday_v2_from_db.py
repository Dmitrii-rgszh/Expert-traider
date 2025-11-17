"""Build intraday_v2 labels directly from SQLite `candles` table.

This is a thin adapter around `build_labels_intraday_v2.py` that:
- pulls OHLCV data from the `candles` table for a given timeframe,
  date range and universe of tickers;
- feeds it into the intraday_v2 label builder;
- writes labels CSV + JSON summary suitable for EDA and dataset building.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backend.scripts.labels.build_labels_intraday_v2 import (
    IntradayV2Config,
    build_labels_intraday_v2,
)
from backend.app.db.session import SessionLocal
from backend.app.models.market_data import Candle


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build intraday_v2 labels from DB")
    p.add_argument("--timeframe", default="5m", help="Timeframe, e.g. 5m")
    p.add_argument("--secids", nargs="*", default=None,
                   help="Optional list of tickers; if omitted, use all in universe")
    p.add_argument("--start-date", type=str, required=True,
                   help="Start date (YYYY-MM-DD)")
    p.add_argument("--end-date", type=str, required=True,
                   help="End date (YYYY-MM-DD, inclusive)")
    p.add_argument("--horizon-minutes", type=int, default=60)
    p.add_argument("--atr-window", type=int, default=14)
    p.add_argument("--tp-r", type=float, default=1.5)
    p.add_argument("--sl-r", type=float, default=1.0)
    p.add_argument("--min-turnover-quantile", type=float, default=0.3)
    p.add_argument("--max-spread-quantile", type=float, default=0.7)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--summary-json", type=Path, required=True)
    return p.parse_args()


def load_candles(
    timeframe: str,
    start_date: str,
    end_date: str,
    secids: list[str] | None = None,
) -> pd.DataFrame:
    session = SessionLocal()
    try:
        q = session.query(Candle).filter(
            Candle.timeframe == timeframe,
            Candle.timestamp >= start_date,
            Candle.timestamp <= end_date,
        )
        if secids:
            q = q.filter(Candle.secid.in_(secids))
        rows = q.all()
        if not rows:
            raise RuntimeError("No candles found for given filters")
        data = [
            {
                "secid": r.secid,
                "timestamp": r.timestamp,
                "open": float(r.open),
                "high": float(r.high),
                "low": float(r.low),
                "close": float(r.close),
                "volume": float(r.volume) if r.volume is not None else None,
                "turnover": float(r.value) if getattr(r, "value", None) is not None else None,
            }
            for r in rows
        ]
        df = pd.DataFrame(data)
        return df
    finally:
        session.close()


def main() -> None:
    args = parse_args()
    cfg = IntradayV2Config(
        timeframe=args.timeframe,
        horizon_minutes=args.horizon_minutes,
        atr_window=args.atr_window,
        tp_r_multiple=args.tp_r,
        sl_r_multiple=args.sl_r,
        min_turnover_quantile=args.min_turnover_quantile,
        max_spread_quantile=args.max_spread_quantile,
    )

    secids: list[str] | None = args.secids or None
    df = load_candles(
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date,
        secids=secids,
    )

    labels = build_labels_intraday_v2(df, cfg)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(args.output_csv, index=False)

    summary = {
        "label_set": "intraday_v2",
        "timeframe": cfg.timeframe,
        "horizon_minutes": cfg.horizon_minutes,
        "tp_r_multiple": cfg.tp_r_multiple,
        "sl_r_multiple": cfg.sl_r_multiple,
        "num_rows": int(labels.shape[0]),
        "positive_rate": float(labels["label_long"].mean()),
        "r_multiple_mean": float(labels["r_multiple"].mean()),
        "r_multiple_median": float(labels["r_multiple"].median()),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    import json

    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
