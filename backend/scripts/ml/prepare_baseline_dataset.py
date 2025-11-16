from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from sqlalchemy import and_, select
from sqlalchemy.orm import selectinload

from backend.app.db.session import SessionLocal
from backend.app.models.market_data import FeatureWindow, TradeLabel
from backend.app.models.news_data import NewsEvent, RiskAlert


def _parse_datetime(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _uppercase_list(values: Iterable[str] | None) -> list[str] | None:
    if not values:
        return None
    return [val.upper() for val in values]


def build_dataset(
    secids: list[str] | None,
    timeframe: str,
    feature_set: str,
    label_set: str,
    start_dt: datetime,
    end_dt: datetime,
) -> pd.DataFrame:
    with SessionLocal() as session:
        window_stmt = (
            select(FeatureWindow)
            .options(selectinload(FeatureWindow.numeric_features))
            .where(
                FeatureWindow.feature_set == feature_set,
                FeatureWindow.timeframe == timeframe,
                FeatureWindow.window_end >= start_dt,
                FeatureWindow.window_end <= end_dt,
            )
        )
        label_stmt = (
            select(TradeLabel)
            .where(
                TradeLabel.label_set == label_set,
                TradeLabel.timeframe == timeframe,
                TradeLabel.signal_time >= start_dt,
                TradeLabel.signal_time <= end_dt,
            )
            .order_by(TradeLabel.signal_time.asc(), TradeLabel.secid.asc())
        )
        if secids:
            window_stmt = window_stmt.where(FeatureWindow.secid.in_(secids))
            label_stmt = label_stmt.where(TradeLabel.secid.in_(secids))

        feature_windows = session.execute(window_stmt).scalars().all()
        labels = session.execute(label_stmt).scalars().all()

        if not feature_windows or not labels:
            return pd.DataFrame()

        window_map = {(window.secid, window.window_end): window for window in feature_windows}

        records: list[dict[str, object]] = []
        for label in labels:
            window = window_map.get((label.secid, label.signal_time))
            if not window:
                continue

            record: dict[str, object] = {
                "secid": window.secid,
                "timeframe": window.timeframe,
                "feature_set": window.feature_set,
                "label_set": label.label_set,
                "signal_time": label.signal_time.isoformat(),
                "horizon_minutes": label.horizon_minutes,
                "forward_return_pct": label.forward_return_pct,
                "max_runup_pct": label.max_runup_pct,
                "max_drawdown_pct": label.max_drawdown_pct,
                "label_long": label.label_long,
                "label_short": label.label_short,
                "long_pnl_pct": getattr(label, "long_pnl_pct", label.forward_return_pct),
                "short_pnl_pct": getattr(label, "short_pnl_pct", -label.forward_return_pct),
            }
            for feature in window.numeric_features:
                record[feature.feature_name] = feature.value_numeric
            records.append(record)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df.sort_values(["signal_time", "secid"], inplace=True)
    return df


def export_dataset(df: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare baseline ML dataset by joining features and labels")
    parser.add_argument("--secids", nargs="*", help="Restrict to specific tickers")
    parser.add_argument("--timeframe", default="1m", help="Timeframe to filter (default: 1m)")
    parser.add_argument("--feature-set", default="tech_v1", help="Feature set to pull windows from")
    parser.add_argument("--label-set", default="basic_v1", help="Label set to join")
    parser.add_argument("--start-date", required=True, help="ISO start datetime (UTC if no TZ provided)")
    parser.add_argument("--end-date", required=True, help="ISO end datetime (UTC if no TZ provided)")
    parser.add_argument(
        "--output",
        default=str(Path("data/training/baseline_dataset.csv")),
        help="Where to write the merged dataset (CSV)",
    )
    parser.add_argument(
        "--include-news-features",
        action="store_true",
        help="Attach aggregated news/risk counts (requires DB access to news tables)",
    )
    parser.add_argument(
        "--news-windows",
        nargs="*",
        type=int,
        default=[60, 240, 1440],
        help="Rolling windows (minutes) for news features (default: 60 240 1440)",
    )
    parser.add_argument(
        "--horizon-filter",
        type=int,
        help="If set, keep only rows with this horizon_minutes value",
    )
    parser.add_argument(
        "--one-hot-horizon",
        action="store_true",
        help="If set, add one-hot columns for horizon_minutes and keep them as features",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    secids = _uppercase_list(args.secids)
    start_dt = _parse_datetime(args.start_date)
    end_dt = _parse_datetime(args.end_date)

    df = build_dataset(secids, args.timeframe, args.feature_set, args.label_set, start_dt, end_dt)
    if df.empty:
        print("No joined rows found for the requested interval.")
        return

    if args.horizon_filter is not None:
        df = df[df["horizon_minutes"] == args.horizon_filter].copy()
        print(f"Filtered to horizon_minutes={args.horizon_filter}, {len(df)} rows remaining.")

    if args.one_hot_horizon:
        dummies = pd.get_dummies(df["horizon_minutes"], prefix="horizon", dtype=float)
        df = pd.concat([df, dummies], axis=1)
        print(f"Added one-hot horizon columns: {list(dummies.columns)}")

    if args.include_news_features:
        windows = args.news_windows or [60, 240, 1440]
        df = enrich_with_news_features(df, secids, start_dt, end_dt, windows)

    output_path = Path(args.output)
    export_dataset(df, output_path)
    print(
        f"Saved {len(df)} rows with {len(df.columns)} columns to {output_path}."
        " Use this file as input for baseline notebooks or training scripts."
    )


def enrich_with_news_features(
    df: pd.DataFrame,
    secids: Sequence[str] | None,
    start_dt: datetime,
    end_dt: datetime,
    windows: Sequence[int],
) -> pd.DataFrame:
    if df.empty:
        return df
    enriched = df.copy()
    enriched["signal_time"] = pd.to_datetime(enriched["signal_time"], utc=True)
    tickers = list(secids or enriched["secid"].unique().tolist())
    max_window = max(windows or [60])
    news_map = _load_event_map(NewsEvent, NewsEvent.published_at, tickers, start_dt, end_dt, max_window)
    risk_map = _load_event_map(RiskAlert, RiskAlert.timestamp, tickers, start_dt, end_dt, max_window)
    for window in windows:
        enriched[f"news_count_{window}m"] = 0
        enriched[f"risk_alerts_{window}m"] = 0
    for secid, indices in enriched.groupby("secid").groups.items():
        target_times = pd.DatetimeIndex(enriched.loc[indices, "signal_time"])
        if target_times.empty:
            continue
        news_times = news_map.get(secid)
        risk_times = risk_map.get(secid)
        for window in windows:
            if news_times is not None and not news_times.empty:
                counts = _window_count_series(news_times, target_times, window)
                enriched.loc[indices, f"news_count_{window}m"] = counts
            if risk_times is not None and not risk_times.empty:
                counts = _window_count_series(risk_times, target_times, window)
                enriched.loc[indices, f"risk_alerts_{window}m"] = counts
    enriched["signal_time"] = (
        enriched["signal_time"].dt.tz_convert(timezone.utc).dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    )
    return enriched


def _load_event_map(
    model,
    timestamp_column,
    secids: Sequence[str],
    start_dt: datetime,
    end_dt: datetime,
    max_window: int,
) -> dict[str, pd.DatetimeIndex]:
    lookback_start = start_dt - timedelta(minutes=max_window)
    with SessionLocal() as session:
        stmt = (
            select(model.secid, timestamp_column)
            .where(
                timestamp_column >= lookback_start,
                timestamp_column <= end_dt,
            )
            .order_by(timestamp_column.asc())
        )
        if secids:
            stmt = stmt.where(model.secid.in_(list(secids)))
        rows = session.execute(stmt).all()
    buckets: dict[str, list[datetime]] = {}
    for secid, ts in rows:
        if not secid or not ts:
            continue
        buckets.setdefault(secid, []).append(ts)
    return {secid: pd.DatetimeIndex(times).tz_convert(timezone.utc) for secid, times in buckets.items()}


def _window_count_series(
    events: pd.DatetimeIndex,
    target_times: pd.DatetimeIndex,
    window_minutes: int,
) -> pd.Series:
    if events.empty:
        return pd.Series([0] * len(target_times), index=target_times)
    events = events.sort_values()
    series = pd.Series(1, index=events)
    cumsum = series.cumsum()
    delta = pd.Timedelta(minutes=window_minutes)
    shifted = target_times - delta
    combined_index = cumsum.index.union(target_times).union(shifted)
    cumsum_full = cumsum.reindex(combined_index, method="ffill").fillna(0)
    at_times = cumsum_full.reindex(target_times, method="ffill").fillna(0)
    before_times = cumsum_full.reindex(shifted, method="ffill").fillna(0)
    counts = (at_times - before_times).astype(int).to_numpy()
    return counts


if __name__ == "__main__":
    main()
