"""Label builder for intraday_v2.

Goal: generate *tradable* intraday labels with ATR-normalised TP/SL,
liquidity/spread filters and basic regime tags, to be used as a cleaner
foundation than intraday_v1.

This script is deliberately minimal and focused on:
- clearly defined label formula;
- explicit filters;
- simple JSON summary for EDA.

It is *not* wired into the full pipeline yet; think of it as a
prototype we can iterate on.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class IntradayV2Config:
    timeframe: str = "5m"
    horizon_minutes: int = 60
    atr_window: int = 14
    tp_r_multiple: float = 1.5
    sl_r_multiple: float = 1.0
    min_turnover_quantile: float = 0.3
    max_spread_quantile: float = 0.7
    max_holding_minutes: int | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build intraday_v2 labels")
    p.add_argument("--candles-csv", type=Path, required=True,
                   help="Input OHLCV file for one timeframe and universe")
    p.add_argument("--output-csv", type=Path, required=True,
                   help="Where to write labels CSV")
    p.add_argument("--summary-json", type=Path, required=True,
                   help="Where to write summary JSON with basic stats")
    p.add_argument("--timeframe", default="5m", help="Candle timeframe (e.g. 5m)")
    p.add_argument("--horizon-minutes", type=int, default=60,
                   help="Intraday horizon in minutes (default: 60)")
    p.add_argument("--atr-window", type=int, default=14,
                   help="ATR window in bars (default: 14)")
    p.add_argument("--tp-r", type=float, default=1.5,
                   help="TP in units of ATR (R multiple)")
    p.add_argument("--sl-r", type=float, default=1.0,
                   help="SL in units of ATR (R multiple)")
    p.add_argument("--min-turnover-quantile", type=float, default=0.3,
                   help="Filter out candles with turnover below this quantile")
    p.add_argument("--max-spread-quantile", type=float, default=0.7,
                   help="Filter out candles with spread above this quantile")
    return p.parse_args()


def compute_atr(df: pd.DataFrame, window: int) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=window).mean()


def build_labels_intraday_v2(df: pd.DataFrame, cfg: IntradayV2Config) -> pd.DataFrame:
    df = df.sort_values(["secid", "timestamp"]).reset_index(drop=True)

    # Basic liquidity proxies; assumes columns `turnover` and optional `spread`.
    turnover = df.get("turnover")
    if turnover is not None:
        thr_turnover = turnover.quantile(cfg.min_turnover_quantile)
        liquid_mask = turnover >= thr_turnover
    else:
        liquid_mask = pd.Series(True, index=df.index)

    spread = df.get("spread")
    if spread is not None:
        thr_spread = spread.quantile(cfg.max_spread_quantile)
        tight_mask = spread <= thr_spread
    else:
        tight_mask = pd.Series(True, index=df.index)

    df["atr"] = df.groupby("secid", group_keys=False).apply(
        lambda g: compute_atr(g, cfg.atr_window)
    )

    # One bar duration in minutes inferred from timeframe.
    step_minutes = int(cfg.timeframe.rstrip("mhd")) if cfg.timeframe.endswith("m") else cfg.horizon_minutes
    horizon_bars = max(1, cfg.horizon_minutes // step_minutes)

    labels = []
    for secid, g in df.groupby("secid", sort=False):
        # we work in numpy for speed
        close = g["close"].to_numpy(dtype=float)
        high = g["high"].to_numpy(dtype=float)
        low = g["low"].to_numpy(dtype=float)
        atr = g["atr"].to_numpy(dtype=float)
        lm = liquid_mask.loc[g.index].to_numpy()
        tm = tight_mask.loc[g.index].to_numpy()

        n = len(g)
        label_long = np.zeros(n, dtype=int)
        r_multiple = np.zeros(n, dtype=float)

        for i in range(n):
            if np.isnan(atr[i]) or not (lm[i] and tm[i]):
                continue
            entry = close[i]
            a = atr[i]
            if a <= 0:
                continue
            tp = entry + cfg.tp_r_multiple * a
            sl = entry - cfg.sl_r_multiple * a
            end = min(n, i + horizon_bars + 1)
            # path within horizon
            h_path = high[i + 1 : end]
            l_path = low[i + 1 : end]
            if len(h_path) == 0:
                continue
            hit_tp = np.where(h_path >= tp)[0]
            hit_sl = np.where(l_path <= sl)[0]
            t_tp = hit_tp[0] if hit_tp.size else None
            t_sl = hit_sl[0] if hit_sl.size else None
            if t_tp is not None and (t_sl is None or t_tp <= t_sl):
                label_long[i] = 1
                # realised R multiple: TP / ATR
                r_multiple[i] = cfg.tp_r_multiple
            elif t_sl is not None and (t_tp is None or t_sl < t_tp):
                label_long[i] = 0
                r_multiple[i] = -cfg.sl_r_multiple
            else:
                # neither TP nor SL hit; mark as 0 but with realised R from close at horizon end
                final_close = close[end - 1]
                r_multiple[i] = (final_close - entry) / a

        out = g[["secid", "timestamp", "close", "atr"]].copy()
        out["label_set"] = "intraday_v2"
        out["horizon_minutes"] = cfg.horizon_minutes
        out["label_long"] = label_long
        out["r_multiple"] = r_multiple
        labels.append(out)

    return pd.concat(labels, axis=0, ignore_index=True)


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

    df = pd.read_csv(args.candles_csv)
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
