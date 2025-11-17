from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from backend.app.db.session import SessionLocal
from backend.app.models.fundamental import FundamentalMetric
from backend.app.models.market_data import FeatureWindow, TradeLabel
from backend.app.models.news_data import NewsEvent, RiskAlert
from backend.app.ml.rules import evaluate_rules


def _log(msg: str) -> None:
    """Lightweight progress logger with UTC timestamp."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{now}] {msg}", flush=True)


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
    parser.add_argument(
        "--include-fundamentals",
        action="store_true",
        help="Join latest fundamental metrics (earnings_yoy, dividend_yield, etc.)",
    )
    parser.add_argument(
        "--fundamental-metrics",
        nargs="*",
        default=["earnings_yoy", "dividend_yield", "net_debt_to_ebitda"],
        help="Metric types to pull from fundamental_metrics table",
    )
    parser.add_argument(
        "--fundamental-lookback-days",
        type=int,
        default=420,
        help="How far back to search for fundamentals (default: 420 days)",
    )
    parser.add_argument(
        "--include-news-text",
        action="store_true",
        help="Build TF-IDF + sentiment aggregates from raw news text bodies",
    )
    parser.add_argument(
        "--news-text-windows",
        nargs="*",
        type=int,
        default=[60, 240, 720],
        help="Rolling windows (minutes) for news text features",
    )
    parser.add_argument(
        "--news-tfidf-dims",
        type=int,
        default=32,
        help="Number of TF-IDF dimensions to keep per window",
    )
    parser.add_argument(
        "--news-tfidf-vocab",
        type=str,
        help="Optional path to persist/reuse TF-IDF vocabulary JSON",
    )
    parser.add_argument(
        "--strong-label-long-quantile",
        type=float,
        default=0.9,
        help="Quantile for defining strong long labels based on forward_return_pct among label_long==1 (default: 0.9)",
    )
    parser.add_argument(
        "--strong-label-short-quantile",
        type=float,
        default=0.1,
        help="Quantile for defining strong short labels based on forward_return_pct among label_short==1 (default: 0.1)",
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

    _log(f"Joined base dataset: {len(df)} rows, {len(df.columns)} cols.")

    # Derive stronger labels for top-tail trades to focus the model on truly profitable opportunities.
    try:
        long_mask = (df["label_long"] == 1) & df["forward_return_pct"].notna()
        short_mask = (df["label_short"] == 1) & df["forward_return_pct"].notna()
        df["label_long_strong"] = 0
        df["label_short_strong"] = 0
        if long_mask.any():
            q_long = df.loc[long_mask, "forward_return_pct"].quantile(args.strong_label_long_quantile)
            df.loc[long_mask & (df["forward_return_pct"] >= q_long), "label_long_strong"] = 1
        if short_mask.any():
            q_short = df.loc[short_mask, "forward_return_pct"].quantile(args.strong_label_short_quantile)
            df.loc[short_mask & (df["forward_return_pct"] <= q_short), "label_short_strong"] = 1
        _log(
            "Derived strong labels: "
            f"long_strong_rate={float(df['label_long_strong'].mean()):.4f}, "
            f"short_strong_rate={float(df['label_short_strong'].mean()):.4f}"
        )
    except Exception as exc:  # safeguard so dataset build never fails solely on strong labels
        _log(f"Skipping strong label derivation due to error: {exc!r}")

    if args.horizon_filter is not None:
        df = df[df["horizon_minutes"] == args.horizon_filter].copy()
        print(f"Filtered to horizon_minutes={args.horizon_filter}, {len(df)} rows remaining.")

    if args.one_hot_horizon:
        dummies = pd.get_dummies(df["horizon_minutes"], prefix="horizon", dtype=float)
        df = pd.concat([df, dummies], axis=1)
        print(f"Added one-hot horizon columns: {list(dummies.columns)}")

    if args.include_news_features:
        windows = args.news_windows or [60, 240, 1440]
        _log(f"Enriching with news counts for windows={windows} ...")
        df = enrich_with_news_features(df, secids, start_dt, end_dt, windows)
        _log("Done news count enrichment.")

    if args.include_fundamentals:
        metrics = args.fundamental_metrics or []
        _log(f"Enriching with fundamentals metrics={metrics} lookback_days={args.fundamental_lookback_days} ...")
        df = enrich_with_fundamental_features(
            df,
            secids,
            start_dt,
            end_dt,
            metrics=[metric.lower() for metric in metrics],
            lookback_days=max(1, args.fundamental_lookback_days),
        )
        _log("Done fundamentals enrichment.")

    if args.include_news_text:
        _log(
            "Enriching with news text TF-IDF/"
            f"sentiment windows={args.news_text_windows or [60, 240, 720]}, dims={args.news_tfidf_dims or 32} ..."
        )
        df = enrich_with_news_text_features(
            df,
            secids,
            start_dt,
            end_dt,
            windows=args.news_text_windows or [60, 240, 720],
            tfidf_dims=max(4, args.news_tfidf_dims or 32),
            vocab_path=Path(args.news_tfidf_vocab).expanduser() if args.news_tfidf_vocab else None,
        )
        _log("Done news text enrichment.")

    output_path = Path(args.output)
    _log(f"Exporting to {output_path} ...")
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


def enrich_with_fundamental_features(
    df: pd.DataFrame,
    secids: Sequence[str] | None,
    start_dt: datetime,
    end_dt: datetime,
    metrics: Sequence[str],
    lookback_days: int,
) -> pd.DataFrame:
    if df.empty or not metrics:
        return df

    fundamentals = _load_fundamental_frame(secids, metrics, start_dt, end_dt, lookback_days)
    if fundamentals.empty:
        return df

    enriched = df.copy()
    enriched["signal_time"] = pd.to_datetime(enriched["signal_time"], utc=True)
    metrics = [metric.lower() for metric in metrics]

    for metric in metrics:
        enriched[f"fund_{metric}_value"] = np.nan
        enriched[f"fund_{metric}_days_since"] = np.nan

    groups = enriched.groupby("secid").groups
    for secid, indices in groups.items():
        ticker_fundamentals = fundamentals[fundamentals["secid"] == secid]
        if ticker_fundamentals.empty:
            continue
        signal_series = enriched.loc[indices, "signal_time"]
        for metric in metrics:
            metric_frame = ticker_fundamentals[ticker_fundamentals["metric_type"] == metric]
            if metric_frame.empty:
                continue
            values, ages = _merge_latest_metric(signal_series, metric_frame)
            value_col = f"fund_{metric}_value"
            age_col = f"fund_{metric}_days_since"
            enriched.loc[signal_series.index, value_col] = values.to_numpy()
            enriched.loc[signal_series.index, age_col] = ages.to_numpy()

    enriched["signal_time"] = (
        enriched["signal_time"].dt.tz_convert(timezone.utc).dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    )
    return enriched


def _load_fundamental_frame(
    secids: Sequence[str] | None,
    metrics: Sequence[str],
    start_dt: datetime,
    end_dt: datetime,
    lookback_days: int,
) -> pd.DataFrame:
    lookback_start = start_dt - timedelta(days=lookback_days)
    with SessionLocal() as session:
        stmt = (
            select(
                FundamentalMetric.secid,
                FundamentalMetric.metric_type,
                FundamentalMetric.metric_date,
                FundamentalMetric.metric_value,
            )
            .where(
                FundamentalMetric.metric_date >= lookback_start,
                FundamentalMetric.metric_date <= end_dt,
            )
            .order_by(FundamentalMetric.metric_date.asc())
        )
        if secids:
            stmt = stmt.where(FundamentalMetric.secid.in_(list(secids)))
        if metrics:
            stmt = stmt.where(FundamentalMetric.metric_type.in_(list(metrics)))
        rows = session.execute(stmt).all()
    if not rows:
        return pd.DataFrame()
    records = [
        {
            "secid": row.secid,
            "metric_type": row.metric_type,
            "metric_date": row.metric_date.astimezone(timezone.utc),
            "metric_value": float(row.metric_value),
        }
        for row in rows
    ]
    frame = pd.DataFrame.from_records(records)
    frame["metric_type"] = frame["metric_type"].str.lower()
    frame.sort_values(["secid", "metric_type", "metric_date"], inplace=True)
    return frame


def _merge_latest_metric(signal_times: pd.Series, metric_frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    if signal_times.empty or metric_frame.empty:
        empty = pd.Series([np.nan] * len(signal_times), index=signal_times.index)
        return empty, empty

    signal_copy = signal_times.to_frame(name="signal_ts").copy()
    signal_copy["_order"] = range(len(signal_copy))
    signal_copy.sort_values("signal_ts", inplace=True)

    metric_sorted = metric_frame.sort_values("metric_date")[["metric_date", "metric_value"]]
    merged = pd.merge_asof(
        signal_copy,
        metric_sorted,
        left_on="signal_ts",
        right_on="metric_date",
        direction="backward",
    )
    merged.sort_values("_order", inplace=True)
    merged.index = signal_times.index

    values = merged["metric_value"].astype(float)
    age_days = (
        (merged["signal_ts"] - merged["metric_date"])
        .dt.total_seconds()
        .div(86400)
        .astype(float)
    )
    return values, age_days


def enrich_with_news_text_features(
    df: pd.DataFrame,
    secids: Sequence[str] | None,
    start_dt: datetime,
    end_dt: datetime,
    windows: Sequence[int],
    tfidf_dims: int,
    vocab_path: Path | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df

    enriched = df.copy()
    enriched["signal_time"] = pd.to_datetime(enriched["signal_time"], utc=True)
    enriched.reset_index(drop=True, inplace=True)
    tickers = list(secids or enriched["secid"].unique())
    if not tickers:
        return enriched

    windows = list(windows or [60, 240, 720])
    max_window = max(windows)
    news_df = _load_news_events_for_text(tickers, start_dt, end_dt, max_window)

    if news_df.empty:
        zero_frames = []
        n = len(enriched)
        for window in windows:
            cols: dict[str, np.ndarray] = {
                f"news_bullish_{window}m": np.zeros(n, dtype=float),
                f"news_bearish_{window}m": np.zeros(n, dtype=float),
                f"news_rule_score_avg_{window}m": np.zeros(n, dtype=float),
                f"news_event_count_{window}m": np.zeros(n, dtype=float),
            }
            for idx in range(tfidf_dims):
                cols[f"news_tfidf_{window}m_{idx:02d}"] = np.zeros(n, dtype=float)
            zero_frames.append(pd.DataFrame(cols, index=enriched.index))
        if zero_frames:
            enriched = pd.concat([enriched, *zero_frames], axis=1)
        enriched["signal_time"] = (
            enriched["signal_time"].dt.tz_convert(timezone.utc).dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        )
        return enriched

    vectorizer = _fit_or_load_vectorizer(news_df["text"].tolist(), tfidf_dims, vocab_path)
    groups = enriched.groupby("secid").groups

    for window in windows:
        bull_col = f"news_bullish_{window}m"
        bear_col = f"news_bearish_{window}m"
        score_col = f"news_rule_score_avg_{window}m"
        count_col = f"news_event_count_{window}m"
        n_rows = len(enriched)
        bull_arr = np.zeros(n_rows, dtype=float)
        bear_arr = np.zeros(n_rows, dtype=float)
        score_arr = np.zeros(n_rows, dtype=float)
        count_arr = np.zeros(n_rows, dtype=float)
        tfidf_mat = np.zeros((n_rows, tfidf_dims), dtype=float)

        # Process per-ticker to keep memory bounded
        for secid, indices in groups.items():
            ticker_events = news_df[news_df["secid"] == secid]
            if ticker_events.empty:
                continue
            idx = np.fromiter(indices, dtype=int)
            signal_times = enriched.loc[idx, "signal_time"].sort_values()
            if signal_times.empty:
                continue
            idx_sorted = signal_times.index.to_numpy(dtype=int)

            agg_texts, bulls, bears, score_sums, counts = _aggregate_window_payloads(
                ticker_events,
                signal_times,
                window_minutes=window,
            )
            avg_scores = np.divide(
                score_sums,
                counts,
                out=np.zeros_like(score_sums, dtype=float),
                where=counts > 0,
            )

            bull_arr[idx_sorted] = bulls
            bear_arr[idx_sorted] = bears
            score_arr[idx_sorted] = avg_scores
            count_arr[idx_sorted] = counts

            # Vectorize texts for this ticker only (much smaller than full frame)
            if len(agg_texts) > 0:
                matrix = vectorizer.transform(agg_texts).toarray()
                for col_idx in range(matrix.shape[1]):
                    tfidf_mat[idx_sorted, col_idx] = matrix[:, col_idx]

        tfidf_cols = {f"news_tfidf_{window}m_{idx:02d}": tfidf_mat[:, idx] for idx in range(tfidf_dims)}
        window_df = pd.DataFrame(
            {
                bull_col: bull_arr,
                bear_col: bear_arr,
                score_col: score_arr,
                count_col: count_arr,
                **tfidf_cols,
            },
            index=enriched.index,
        )
        enriched = pd.concat([enriched, window_df], axis=1)

    enriched["signal_time"] = (
        enriched["signal_time"].dt.tz_convert(timezone.utc).dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    )
    return enriched


def _load_news_events_for_text(
    secids: Sequence[str],
    start_dt: datetime,
    end_dt: datetime,
    max_window: int,
) -> pd.DataFrame:
    lookback_start = start_dt - timedelta(minutes=max_window)
    with SessionLocal() as session:
        stmt = (
            select(NewsEvent)
            .where(
                NewsEvent.published_at >= lookback_start,
                NewsEvent.published_at <= end_dt,
                NewsEvent.secid.in_(list(secids)),
            )
            .order_by(NewsEvent.published_at.asc())
        )
        events = session.execute(stmt).scalars().all()
    if not events:
        return pd.DataFrame()

    records = []
    for event in events:
        secid = (event.secid or "").upper()
        if not secid:
            continue
        text = " ".join(filter(None, [event.title, event.body]))
        text = text.strip()
        if not text:
            continue
        rules = evaluate_rules(text)
        direction = _direction_from_votes(rules.direction_votes)
        records.append(
            {
                "secid": secid,
                "published_at": event.published_at.astimezone(timezone.utc),
                "text": text,
                "direction": direction,
                "score_bonus": float(rules.score_bonus),
            }
        )
    frame = pd.DataFrame.from_records(records)
    frame.sort_values(["secid", "published_at"], inplace=True)
    frame["is_bullish"] = (frame["direction"] == "bullish").astype(int)
    frame["is_bearish"] = (frame["direction"] == "bearish").astype(int)
    return frame


def _direction_from_votes(votes: dict[str, int]) -> str:
    if not votes:
        return "neutral"
    best = max(votes.items(), key=lambda item: item[1])
    if best[1] == 0:
        return "neutral"
    return best[0]


def _aggregate_window_payloads(
    events: pd.DataFrame,
    signal_times: pd.Series,
    window_minutes: int,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if signal_times.empty:
        m = len(signal_times)
        zeros = np.zeros(m, dtype=float)
        return [""] * m, zeros, zeros, zeros, zeros

    events_sorted = events.sort_values("published_at")
    event_times = events_sorted["published_at"].to_numpy(dtype="datetime64[ns]")
    texts = events_sorted["text"].tolist()
    bullish = events_sorted["is_bullish"].to_numpy(dtype=float)
    bearish = events_sorted["is_bearish"].to_numpy(dtype=float)
    scores = events_sorted["score_bonus"].to_numpy(dtype=float)

    target_times = signal_times.to_numpy(dtype="datetime64[ns]")
    window_delta = np.timedelta64(window_minutes, "m")

    aggregated_texts: list[str] = []
    bullish_counts: list[float] = []
    bearish_counts: list[float] = []
    score_sums: list[float] = []
    counts: list[float] = []

    start_idx = 0
    end_idx = 0
    for target in target_times:
        while end_idx < len(event_times) and event_times[end_idx] <= target:
            end_idx += 1
        lower_bound = target - window_delta
        while start_idx < end_idx and event_times[start_idx] < lower_bound:
            start_idx += 1
        if start_idx >= end_idx:
            aggregated_texts.append("")
            bullish_counts.append(0.0)
            bearish_counts.append(0.0)
            score_sums.append(0.0)
            counts.append(0.0)
            continue
        aggregated_texts.append(" ".join(texts[start_idx:end_idx]))
        bullish_counts.append(float(bullish[start_idx:end_idx].sum()))
        bearish_counts.append(float(bearish[start_idx:end_idx].sum()))
        score_sums.append(float(scores[start_idx:end_idx].sum()))
        counts.append(float(end_idx - start_idx))

    return (
        aggregated_texts,
        np.array(bullish_counts, dtype=float),
        np.array(bearish_counts, dtype=float),
        np.array(score_sums, dtype=float),
        np.array(counts, dtype=float),
    )


def _fit_or_load_vectorizer(
    corpus: Sequence[str],
    tfidf_dims: int,
    vocab_path: Path | None,
) -> TfidfVectorizer:
    if vocab_path and vocab_path.exists():
        vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        return TfidfVectorizer(vocabulary=vocab, ngram_range=(1, 2))

    vectorizer = TfidfVectorizer(max_features=tfidf_dims, ngram_range=(1, 2), min_df=2)
    if corpus:
        vectorizer.fit(corpus)
    else:
        vectorizer.fit(["placeholder"])

    if vocab_path:
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        vocab_path.write_text(json.dumps(vectorizer.vocabulary_), encoding="utf-8")
    return vectorizer


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
