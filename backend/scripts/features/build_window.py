from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sqlalchemy import delete, select

from backend.app.db.session import SessionLocal
from backend.app.models import Candle, FeatureNumeric, FeatureWindow, IndexCandle
from backend.scripts.features.registry import FeatureStoreConfig

BAR_MINUTES = {
    "1m": 1,
    "5m": 5,
    "10m": 10,
    "15m": 15,
    "1h": 60,
    "1d": 1440,
}

ALL_FEATURE_COLUMNS = [
    "return_1",
    "sma_ratio_5_20",
    "ema_ratio_12_26",
    "macd_line",
    "macd_signal",
    "macd_hist",
    "rsi_14",
    "volatility_20",
    "atr_14",
    "volume_zscore_20",
    "price_position_20",
    "intraday_range_pct",
    "candle_body_pct",
    "upper_shadow_pct",
    "lower_shadow_pct",
    "bullish_engulfing",
    "bearish_engulfing",
    "bollinger_band_pct",
    "stoch_k_14",
    "stoch_d_3",
    # New features for tech_v3
    "return_vs_imoex",
    "return_vs_rts",
    "volatility_60",
    "volatility_120",
    "ema_ratio_5_20",
    "ema_ratio_20_60",
    "atr_ratio_5_14",
    "atr_ratio_14_30",
]


def timeframe_to_minutes(timeframe: str) -> int:
    try:
        return BAR_MINUTES[timeframe]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Unsupported timeframe '{timeframe}'") from exc


def load_index_candles(
    index_code: str,
    timeframe: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Load index candles (IMOEX, RTSI) for relative return calculation."""
    with SessionLocal() as session:
        stmt = (
            select(IndexCandle)
            .where(
                IndexCandle.index_code == index_code,
                IndexCandle.timeframe == timeframe,
                IndexCandle.timestamp >= start,
                IndexCandle.timestamp <= end,
            )
            .order_by(IndexCandle.timestamp.asc())
        )
        rows = session.execute(stmt).scalars().all()
    
    if not rows:
        return pd.DataFrame()
    
    records = [
        {
            "timestamp": row.timestamp,
            "close": float(row.close),
        }
        for row in rows
    ]
    df = pd.DataFrame.from_records(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def load_candles(
    secid: str,
    timeframe: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    with SessionLocal() as session:
        stmt = (
            select(Candle)
            .where(
                Candle.secid == secid,
                Candle.timeframe == timeframe,
                Candle.timestamp >= start,
                Candle.timestamp <= end,
            )
            .order_by(Candle.timestamp.asc())
        )
        rows = session.execute(stmt).scalars().all()

    records: list[dict] = []
    for row in rows:
        records.append(
            {
                "timestamp": row.timestamp,
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
                "volume": float(row.volume or 0.0),
            }
        )
    df = pd.DataFrame.from_records(records)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if df.empty:
        df = _resample_from_lower_timeframe(secid, timeframe, start, end)
    return df


def _resample_from_lower_timeframe(
    secid: str,
    target_timeframe: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    minutes = timeframe_to_minutes(target_timeframe)
    if minutes <= 1:
        return pd.DataFrame()
    base_timeframe = "1m"
    base_minutes = timeframe_to_minutes(base_timeframe)
    if minutes % base_minutes != 0:
        return pd.DataFrame()
    fetch_start = start - timedelta(minutes=minutes)
    with SessionLocal() as session:
        stmt = (
            select(Candle)
            .where(
                Candle.secid == secid,
                Candle.timeframe == base_timeframe,
                Candle.timestamp >= fetch_start,
                Candle.timestamp <= end,
            )
            .order_by(Candle.timestamp.asc())
        )
        rows = session.execute(stmt).scalars().all()
    if not rows:
        return pd.DataFrame()
    records = [
        {
            "timestamp": row.timestamp,
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
            "volume": float(row.volume or 0.0),
        }
        for row in rows
    ]
    base_df = pd.DataFrame(records)
    if base_df.empty:
        return pd.DataFrame()
    base_df["timestamp"] = pd.to_datetime(base_df["timestamp"], utc=True)
    base_df.set_index("timestamp", inplace=True)
    base_df.sort_index(inplace=True)
    rule = f"{minutes}min"
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    resampled = base_df.resample(rule, label="right", closed="right").agg(agg).dropna()
    tz = start.tzinfo or timezone.utc
    resampled.index = resampled.index.tz_convert(tz)
    resampled = resampled[(resampled.index >= start) & (resampled.index <= end)]
    resampled.reset_index(inplace=True)
    return resampled


def compute_features(
    df: pd.DataFrame,
    required_features: Sequence[str] | None = None,
    timeframe: str | None = None,
    imoex_df: pd.DataFrame | None = None,
    rts_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    work = df.copy()
    work.set_index("timestamp", inplace=True)
    work.sort_index(inplace=True)

    if work.empty:
        return work.reset_index()

    target_features = list(required_features or ALL_FEATURE_COLUMNS)

    work["return_1"] = work["close"].pct_change().fillna(0.0)
    work["sma_5"] = work["close"].rolling(window=5, min_periods=5).mean()
    work["sma_20"] = work["close"].rolling(window=20, min_periods=20).mean()
    work["sma_ratio_5_20"] = (work["sma_5"] / work["sma_20"]) - 1

    work["ema_12"] = work["close"].ewm(span=12, adjust=False).mean()
    work["ema_26"] = work["close"].ewm(span=26, adjust=False).mean()
    work["ema_ratio_12_26"] = (work["ema_12"] / work["ema_26"]) - 1
    work["macd_line"] = work["ema_12"] - work["ema_26"]
    work["macd_signal"] = work["macd_line"].ewm(span=9, adjust=False).mean()
    work["macd_hist"] = work["macd_line"] - work["macd_signal"]

    delta = work["close"].diff()
    up = np.where(delta > 0, delta, 0)
    down = np.where(delta < 0, -delta, 0)
    roll_up = pd.Series(up, index=work.index).rolling(window=14, min_periods=14).mean()
    roll_down = pd.Series(down, index=work.index).rolling(window=14, min_periods=14).mean()
    rs = roll_up / roll_down
    work["rsi_14"] = 100 - (100 / (1 + rs))

    work["volatility_20"] = work["return_1"].rolling(window=20, min_periods=20).std().fillna(0.0)

    prev_close = work["close"].shift(1)
    tr_components = pd.concat(
        [
            (work["high"] - work["low"]).abs(),
            (work["high"] - prev_close).abs(),
            (prev_close - work["low"]).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1)
    work["atr_14"] = true_range.rolling(window=14, min_periods=14).mean()

    work["volume_zscore_20"] = (
        (work["volume"] - work["volume"].rolling(window=20, min_periods=20).mean())
        / work["volume"].rolling(window=20, min_periods=20).std(ddof=0)
    )

    rolling_max = work["close"].rolling(window=20, min_periods=20).max()
    rolling_min = work["close"].rolling(window=20, min_periods=20).min()
    denom = (rolling_max - rolling_min).replace(0, np.nan)
    work["price_position_20"] = (work["close"] - rolling_min) / denom

    work["intraday_range_pct"] = (work["high"] - work["low"]) / work["close"].replace(0, np.nan)

    body = work["close"] - work["open"]
    work["candle_body_pct"] = body / work["open"].replace(0, np.nan)
    upper_shadow = work["high"] - work[["open", "close"]].max(axis=1)
    lower_shadow = work[["open", "close"]].min(axis=1) - work["low"]
    work["upper_shadow_pct"] = upper_shadow / work["open"].replace(0, np.nan)
    work["lower_shadow_pct"] = lower_shadow / work["open"].replace(0, np.nan)

    prev_body = body.shift(1)
    prev_open = work["open"].shift(1)
    prev_close = work["close"].shift(1)
    work["bullish_engulfing"] = (
        (body > 0)
        & (prev_body < 0)
        & (work["open"] <= prev_close)
        & (work["close"] >= prev_open)
    ).astype(int)
    work["bearish_engulfing"] = (
        (body < 0)
        & (prev_body > 0)
        & (work["open"] >= prev_close)
        & (work["close"] <= prev_open)
    ).astype(int)

    mid = work["close"].rolling(window=20, min_periods=20).mean()
    std = work["close"].rolling(window=20, min_periods=20).std(ddof=0)
    upper = mid + (2 * std)
    lower = mid - (2 * std)
    bb_denom = (upper - lower).replace(0, np.nan)
    work["bollinger_band_pct"] = (work["close"] - lower) / bb_denom

    lowest_low = work["low"].rolling(window=14, min_periods=14).min()
    highest_high = work["high"].rolling(window=14, min_periods=14).max()
    stoch_denom = (highest_high - lowest_low).replace(0, np.nan)
    stoch_k = (work["close"] - lowest_low) / stoch_denom
    work["stoch_k_14"] = stoch_k
    work["stoch_d_3"] = stoch_k.rolling(window=3, min_periods=3).mean()

    # New features for tech_v3
    # Relative returns vs indices
    if imoex_df is not None and not imoex_df.empty:
        imoex_indexed = imoex_df.set_index("timestamp")
        imoex_indexed = imoex_indexed.reindex(work.index, method="ffill")
        imoex_return = imoex_indexed["close"].pct_change().fillna(0.0)
        work["return_vs_imoex"] = work["return_1"] - imoex_return
    else:
        work["return_vs_imoex"] = 0.0
    
    if rts_df is not None and not rts_df.empty:
        rts_indexed = rts_df.set_index("timestamp")
        rts_indexed = rts_indexed.reindex(work.index, method="ffill")
        rts_return = rts_indexed["close"].pct_change().fillna(0.0)
        work["return_vs_rts"] = work["return_1"] - rts_return
    else:
        work["return_vs_rts"] = 0.0
    
    # Long-window volatility
    work["volatility_60"] = work["return_1"].rolling(window=60, min_periods=60).std().fillna(0.0)
    work["volatility_120"] = work["return_1"].rolling(window=120, min_periods=120).std().fillna(0.0)
    
    # Multi-timeframe EMA ratios
    ema_5 = work["close"].ewm(span=5, adjust=False).mean()
    ema_20 = work["close"].ewm(span=20, adjust=False).mean()
    ema_60 = work["close"].ewm(span=60, adjust=False).mean()
    work["ema_ratio_5_20"] = (ema_5 / ema_20) - 1
    work["ema_ratio_20_60"] = (ema_20 / ema_60) - 1
    
    # Multi-timeframe ATR ratios
    atr_5 = true_range.rolling(window=5, min_periods=5).mean()
    atr_30 = true_range.rolling(window=30, min_periods=30).mean()
    work["atr_ratio_5_14"] = (atr_5 / work["atr_14"]).replace([np.inf, -np.inf], np.nan)
    work["atr_ratio_14_30"] = (work["atr_14"] / atr_30).replace([np.inf, -np.inf], np.nan)

    work.replace([np.inf, -np.inf], np.nan, inplace=True)
    needed_columns = [col for col in target_features if col in work.columns]
    if needed_columns:
        work.dropna(subset=needed_columns, inplace=True)
    work.reset_index(inplace=True)
    return work


def persist_features(
    secid: str,
    timeframe: str,
    feature_set: str,
    window_size: int,
    df: pd.DataFrame,
    feature_columns: Sequence[str],
) -> int:
    if df.empty:
        return 0

    bar_minutes = timeframe_to_minutes(timeframe)
    df["window_start"] = df["timestamp"] - pd.to_timedelta(window_size * bar_minutes, unit="m")

    with SessionLocal() as session:
        min_start = df["window_start"].min().to_pydatetime()
        max_end = df["timestamp"].max().to_pydatetime()
        session.execute(
            delete(FeatureWindow).where(
                FeatureWindow.secid == secid,
                FeatureWindow.timeframe == timeframe,
                FeatureWindow.feature_set == feature_set,
                FeatureWindow.window_end >= min_start,
                FeatureWindow.window_end <= max_end,
            )
        )
        session.commit()

        inserted = 0
        for row in df.itertuples():
            window = FeatureWindow(
                secid=secid,
                timeframe=timeframe,
                window_start=row.window_start.to_pydatetime(),
                window_end=row.timestamp.to_pydatetime(),
                feature_set=feature_set,
                generated_at=datetime.now(timezone.utc),
            )
            session.add(window)
            session.flush()
            for feature in feature_columns:
                if feature not in df.columns:
                    continue
                value = getattr(row, feature, None)
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    continue
                session.add(
                    FeatureNumeric(
                        feature_window_id=window.id,
                        feature_name=feature,
                        value_numeric=float(value),
                    )
                )
            inserted += 1
        session.commit()
    return inserted


def run_pipeline(
    secids: Sequence[str],
    timeframe: str,
    start: datetime,
    end: datetime,
    feature_set: str,
    window_size: int,
    feature_columns: Sequence[str],
) -> None:
    # Load index data once for all tickers
    print("Loading index data...")
    imoex_df = load_index_candles("IMOEX", timeframe, start, end)
    rts_df = load_index_candles("RTSI", timeframe, start, end)
    print(f"IMOEX: {len(imoex_df)} candles, RTSI: {len(rts_df)} candles")
    
    total = 0
    for secid in secids:
        candles = load_candles(secid, timeframe, start, end)
        if candles.empty:
            continue
        features = compute_features(
            candles,
            required_features=feature_columns,
            timeframe=timeframe,
            imoex_df=imoex_df,
            rts_df=rts_df,
        )
        inserted = persist_features(
            secid,
            timeframe,
            feature_set,
            window_size,
            features,
            feature_columns,
        )
        total += inserted
        print(f"{secid}: inserted {inserted} feature windows")
    print(f"Done. Total windows: {total}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build technical feature windows")
    parser.add_argument("secids", nargs="+", help="Tickers to process")
    parser.add_argument("--timeframe", default="1m", help="Timeframe key (default: 1m)")
    parser.add_argument("--start-date", required=True, help="Start datetime (YYYY-MM-DD or ISO)")
    parser.add_argument("--end-date", required=True, help="End datetime (YYYY-MM-DD or ISO)")
    parser.add_argument("--feature-set", default="tech_v1", help="Feature set name")
    parser.add_argument(
        "--window-size",
        type=int,
        help="Window size in bars for window_start calculation (overrides config)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/feature_store.yaml"),
        help="Path to feature store config (default: config/feature_store.yaml)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec_window_size = None
    feature_columns = list(ALL_FEATURE_COLUMNS)
    if args.config and args.config.exists():
        store = FeatureStoreConfig(args.config)
        try:
            spec = store.get_feature_set(args.feature_set, args.timeframe)
            spec_window_size = spec.window_size
            if spec.features:
                feature_columns = [feat for feat in spec.features if feat]
            print(
                f"[feature-store] {args.feature_set}/{args.timeframe}: "
                f"window_size={spec.window_size}, version={spec.version}, features={len(feature_columns)}"
            )
        except (KeyError, ValueError) as exc:
            print(f"[feature-store] {exc}")
    start = datetime.fromisoformat(args.start_date)
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end_date)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    window_size = args.window_size or spec_window_size or 60
    run_pipeline(
        secids=[sec.upper() for sec in args.secids],
        timeframe=args.timeframe,
        start=start,
        end=end,
        feature_set=args.feature_set,
        window_size=window_size,
        feature_columns=feature_columns,
    )


if __name__ == "__main__":
    main()
