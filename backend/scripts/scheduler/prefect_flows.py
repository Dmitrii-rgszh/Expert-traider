from __future__ import annotations

import json
import os
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import httpx
from prefect import flow, task
from prefect.futures import PrefectFuture

from backend.scripts.features.build_window import ALL_FEATURE_COLUMNS, run_pipeline as build_features
from backend.scripts.features.export_snapshots import export_feature_snapshots
from backend.scripts.features.registry import FeatureStoreConfig
from backend.scripts.labels.build_labels import run_pipeline as build_labels
from backend.scripts.ml.prepare_baseline_dataset import (
    build_dataset as build_training_dataset,
    enrich_with_news_features,
    export_dataset,
)
from backend.scripts.monitoring.snapshot_report import generate_snapshot_report
from backend.scripts.ingestion.base import IngestionStats, logger
from backend.scripts.ingestion.calendar_ingestor import CalendarIngestor, parse_date
from backend.scripts.ingestion.candles import CandleIngestor, INTERVAL_MAP
from backend.scripts.ingestion.fx import FxIngestor
from backend.scripts.ingestion.macro import MacroSeriesIngestor, PolicyRateIngestor
from backend.scripts.ingestion.news import MoexNewsClient, NewsIngestor
from backend.scripts.ingestion.ofz import OfzAuctionIngestor, OfzYieldIngestor
from backend.scripts.ingestion.sanctions import SanctionIngestor, SanctionsApiClient
from backend.scripts.monitoring.check_trade_labels import run_check as run_trade_label_monitor
from backend.scripts.monitoring.freshness import FreshnessMonitor
from backend.scripts.pipelines.regime_policy import MarketRegimePipeline, PolicyFeedbackPipeline
from backend.scripts.training.train_temporal_cnn import run_training as train_temporal_cnn


DEFAULT_TIMEFRAME_SEQUENCE = ["1m", "5m", "15m", "1h", "1d"]


def _load_default_secids() -> list[str]:
    env_value = os.getenv("CANDLE_SECIDS")
    if env_value:
        configured = [secid.strip().upper() for secid in env_value.split(",") if secid.strip()]
        if configured:
            return configured
    config_path = Path("config/universe_core.json")
    if config_path.exists():
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            secids = payload.get("core_equities")
            if isinstance(secids, list) and secids:
                return [str(sec).upper() for sec in secids if str(sec).strip()]
        except json.JSONDecodeError:
            logger.warning("Failed to parse %s for core tickers", config_path)
    return ["SBER", "GAZP", "LKOH", "GMKN"]


DEFAULT_CANDLE_SECIDS = _load_default_secids()
TIMEFRAME_LOOKBACK_DAYS: dict[str, int] = {
    "1m": 2,
    "5m": 3,
    "15m": 5,
    "1h": 10,
    "1d": 365,
}


def _parse_iso_datetime(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _estimate_expected_rows(timeframe: str, start: date, end: date, secids_count: int) -> int:
    interval = INTERVAL_MAP.get(timeframe)
    if not interval:
        raise ValueError(f"Unsupported timeframe '{timeframe}'")
    days = (end - start).days + 1
    total_minutes = days * 24 * 60
    bars_per_ticker = max(1, total_minutes // interval)
    return bars_per_ticker * max(1, secids_count)


@task(name="ingest-candles")
def ingest_candles_task(
    secids: Optional[list[str]] = None,
    timeframe: str = "1m",
    board: str = "TQBR",
    start_dt: Optional[date] = None,
    end_dt: Optional[date] = None,
    export_dir: Optional[str] = None,
) -> IngestionStats:
    tickers = secids or DEFAULT_CANDLE_SECIDS
    today = date.today()
    start_date = start_dt or today - timedelta(days=5)
    end_date = end_dt or today
    export_path = Path(export_dir).expanduser() if export_dir else None
    stats = CandleIngestor(board=board).run(
        tickers,
        timeframe,
        start_date,
        end_date,
        export_dir=export_path,
    )
    logger.info("Candle ingestion stats: %s", stats)
    return stats


@task(name="validate-candle-volume")
def validate_candle_volume_task(
    stats: IngestionStats,
    timeframe: str,
    start_dt: date,
    end_dt: date,
    secids_count: int,
    min_ratio: float = 0.2,
    fail_on_low: bool = False,
) -> None:
    expected = _estimate_expected_rows(timeframe, start_dt, end_dt, secids_count)
    if expected == 0:
        return
    ratio = stats.processed / expected if expected else 0.0
    if ratio < min_ratio:
        message = (
            f"Low candle volume for {timeframe} {start_dt}..{end_dt}: "
            f"processed={stats.processed} expected~{expected} ratio={ratio:.2f}"
        )
        if fail_on_low:
            raise ValueError(message)
        logger.warning(message)


@task(name="build-feature-windows")
def build_feature_windows_task(
    secids: Optional[list[str]] = None,
    timeframe: str = "1m",
    feature_set: str = "tech_v1",
    window_size: Optional[int] = None,
    start_dt: Optional[date] = None,
    end_dt: Optional[date] = None,
) -> None:
    tickers = secids or DEFAULT_CANDLE_SECIDS
    today = date.today()
    start_date = start_dt or (today - timedelta(days=5))
    end_date = end_dt or today
    start = datetime.combine(start_date, time.min, tzinfo=timezone.utc)
    end = datetime.combine(end_date, time.max, tzinfo=timezone.utc)
    feature_columns = list(ALL_FEATURE_COLUMNS)
    config_path = Path("config/feature_store.yaml")
    resolved_window_size = window_size
    if config_path.exists():
        try:
            store = FeatureStoreConfig(config_path)
            spec = store.get_feature_set(feature_set, timeframe)
            if spec.features:
                feature_columns = [feat for feat in spec.features if feat]
            if not resolved_window_size:
                resolved_window_size = spec.window_size
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to load feature store config: %s", exc)
    build_features(
        tickers,
        timeframe,
        start,
        end,
        feature_set,
        resolved_window_size or 60,
        feature_columns,
    )


@task(name="export-feature-snapshots")
def export_feature_snapshots_task(
    secids: Optional[list[str]] = None,
    timeframe: str = "1m",
    feature_set: str = "tech_v1",
    start_dt: Optional[date] = None,
    end_dt: Optional[date] = None,
    output_dir: str = "data/processed/features",
    file_format: str = "parquet",
    daily_partition: bool = True,
) -> list[str]:
    tickers = secids or DEFAULT_CANDLE_SECIDS
    today = date.today()
    start_date = start_dt or (today - timedelta(days=5))
    end_date = end_dt or today
    start = datetime.combine(start_date, time.min, tzinfo=timezone.utc)
    end = datetime.combine(end_date, time.max, tzinfo=timezone.utc)
    files = export_feature_snapshots(
        tickers,
        timeframe=timeframe,
        feature_set=feature_set,
        start_dt=start,
        end_dt=end,
        output_dir=Path(output_dir),
        file_format=file_format,
        partition_daily=daily_partition,
    )
    logger.info("Exported %s snapshot files to %s", len(files), output_dir)
    return [str(path) for path in files]


@task(name="publish-snapshot-report")
def publish_snapshot_report_task(
    feature_set: str,
    timeframe: str,
    since_iso: Optional[str] = None,
    output_path: str = "docs/data_quality/train_snapshots.json",
) -> str:
    path = generate_snapshot_report(
        feature_set=feature_set,
        timeframe=timeframe,
        since=since_iso,
        output=Path(output_path),
    )
    logger.info("Snapshot report written to %s", path)
    return str(path)


@task(name="build-trade-labels")
def build_trade_labels_task(
    secids: Optional[list[str]] = None,
    timeframe: str = "1m",
    label_set: str = "basic_v1",
    horizons: Optional[list[int]] = None,
    take_profit: float = 0.02,
    stop_loss: float = 0.01,
    start_dt: Optional[date] = None,
    end_dt: Optional[date] = None,
) -> None:
    tickers = secids or DEFAULT_CANDLE_SECIDS
    today = date.today()
    start_date = start_dt or (today - timedelta(days=5))
    end_date = end_dt or today
    start = datetime.combine(start_date, time.min, tzinfo=timezone.utc)
    end = datetime.combine(end_date, time.max, tzinfo=timezone.utc)
    label_horizons = horizons or [60, 240, 1440]
    build_labels(
        tickers,
        timeframe,
        start,
        end,
        label_set,
        label_horizons,
        take_profit,
        stop_loss,
    )


@task(name="prepare-baseline-dataset")
def prepare_baseline_dataset_task(
    secids: Optional[list[str]] = None,
    timeframe: str = "1m",
    feature_set: str = "tech_v1",
    label_set: str = "basic_v1",
    start_iso: Optional[str] = None,
    end_iso: Optional[str] = None,
    output_path: str = "data/training/baseline_dataset.csv",
    include_news_features: bool = False,
    news_windows: Optional[list[int]] = None,
) -> str:
    tickers = secids or DEFAULT_CANDLE_SECIDS
    if not start_iso or not end_iso:
        raise ValueError("start_iso and end_iso must be provided")
    start_dt = _parse_iso_datetime(start_iso)
    end_dt = _parse_iso_datetime(end_iso)
    df = build_training_dataset(
        tickers,
        timeframe,
        feature_set,
        label_set,
        start_dt,
        end_dt,
    )
    if df.empty:
        raise ValueError("Baseline dataset returned zero rows; adjust time range or secids.")
    if include_news_features:
        df = enrich_with_news_features(
            df,
            tickers,
            start_dt,
            end_dt,
            news_windows or [60, 240, 1440],
        )
    path = Path(output_path)
    export_dataset(df, path)
    logger.info("Baseline dataset saved to %s (%s rows)", path, len(df))
    return str(path)


@task(name="train-temporal-cnn")
def train_temporal_cnn_task(
    dataset_path: str,
    seq_len: int = 32,
    batch_size: int = 64,
    max_epochs: int = 15,
    learning_rate: float = 1e-3,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    num_workers: int = 0,
    gradient_clip: float = 1.0,
    accelerator: str = "auto",
    devices: int = 1,
    precision: str = "32-true",
    log_dir: str = "logs/temporal_cnn",
    logger_type: str = "both",
    plot_roc: bool = False,
    mlflow_tracking_uri: Optional[str] = None,
    mlflow_experiment: Optional[str] = None,
    mlflow_run_name: Optional[str] = None,
    mlflow_tags: Optional[list[str]] = None,
    model_type: str = "tcn",
    hidden_dim: int = 128,
    attn_heads: int = 4,
    dropout: float = 0.2,
    walk_forward_splits: Optional[list[dict[str, object]]] = None,
    report_dir: str = "docs/modeling/train_runs",
) -> dict[str, Any]:
    args = SimpleNamespace(
        dataset_path=Path(dataset_path),
        seq_len=seq_len,
        batch_size=batch_size,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        num_workers=num_workers,
        gradient_clip=gradient_clip,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        log_dir=Path(log_dir),
        logger=logger_type,
        plot_roc=plot_roc,
        seed=1337,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment=mlflow_experiment,
        mlflow_run_name=mlflow_run_name,
        mlflow_tags=mlflow_tags,
        model_type=model_type,
        hidden_dim=hidden_dim,
        attn_heads=attn_heads,
        dropout=dropout,
        walk_forward_splits=walk_forward_splits,
        walk_forward_json=None,
        report_dir=Path(report_dir),
    )
    metrics = train_temporal_cnn(args)
    logger.info("Temporal CNN metrics: %s", metrics)
    return metrics


@task(name="notify-training")
def notify_training_task(
    metrics: dict[str, Any],
    notification_url: Optional[str],
    context: Optional[dict[str, object]] = None,
) -> None:
    if not notification_url:
        return
    payload = {
        "text": "Training completed",
        "metrics": metrics,
        "context": context or {},
    }
    try:
        response = httpx.post(notification_url, json=payload, timeout=10)
        response.raise_for_status()
        logger.info("Notification sent to %s", notification_url)
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("Failed to send notification: %s", exc)


@task(name="ingest-calendar")
def ingest_calendar(start: date, end: date) -> None:
    stats = CalendarIngestor().run(start, end)
    logger.info("Calendar stats: %s", stats)


@task(name="ingest-fx")
def ingest_fx() -> None:
    FxIngestor().run(["USDRUB", "EURRUB", "CNYRUB"])


@task(name="ingest-news")
def ingest_news(json_file: Optional[Path] = None) -> None:
    file_arg = json_file if json_file and json_file.exists() else None
    moex_client = None if file_arg else MoexNewsClient()
    NewsIngestor(file_path=file_arg, moex_client=moex_client).run()


@task(name="ingest-sanctions")
def ingest_sanctions(file_path: Optional[Path], use_api: bool = True) -> None:
    file_arg = file_path if file_path and file_path.exists() else None
    api_client = SanctionsApiClient() if use_api or not file_arg else None
    SanctionIngestor(file_path=file_arg, api_client=api_client).run()


@task(name="ingest-ofz")
def ingest_ofz() -> None:
    OfzYieldIngestor().run()
    OfzAuctionIngestor().run()


@task(name="ingest-macro")
def ingest_macro(macro_csv: Optional[Path], policy_csv: Optional[Path]) -> None:
    MacroSeriesIngestor().run(macro_csv)
    PolicyRateIngestor().run(policy_csv)


@task(name="run-regime-pipeline")
def run_regime_pipeline() -> None:
    MarketRegimePipeline().run()


@task(name="run-policy-feedback")
def run_policy_feedback(events_path: Optional[Path]) -> None:
    events: list[dict] = []
    if events_path and events_path.exists():
        events = json.loads(events_path.read_text(encoding="utf-8"))
    PolicyFeedbackPipeline("contextual_bandit", "v1").run(events)


@task(name="freshness-check")
def freshness_check(table: str, max_lag_minutes: int, severity: str = "medium") -> None:
    FreshnessMonitor().run(table, max_lag_minutes, severity=severity)


@task(name="monitor-trade-labels")
def monitor_trade_labels_task(
    label_set: str = "basic_v1",
    secids: Optional[list[str]] = None,
    timeframe: Optional[str] = "1m",
    lookback_days: int = 3,
    min_rows: Optional[int] = None,
    fail_on_empty: bool = True,
) -> None:
    run_trade_label_monitor(
        label_set=label_set,
        secids=secids,
        timeframe=timeframe,
        lookback_days=lookback_days,
        fail_on_empty=fail_on_empty,
        min_rows=min_rows,
    )


@flow(name="daily-ingestion-flow")
def daily_ingestion_flow(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    news_json: Optional[str] = None,
    sanctions_file: Optional[str] = None,
    sanctions_use_api: bool = True,
    macro_csv: Optional[str] = None,
    policy_csv: Optional[str] = None,
) -> None:
    today = date.today()
    start_dt = parse_date(start_date) if start_date else today - timedelta(days=1)
    end_dt = parse_date(end_date) if end_date else today
    ingest_candles_task(start_dt=start_dt, end_dt=end_dt)
    build_feature_windows_task(start_dt=start_dt, end_dt=end_dt)
    build_trade_labels_task(start_dt=start_dt, end_dt=end_dt)
    ingest_calendar(start_dt, end_dt)
    ingest_fx()
    ingest_news(Path(news_json) if news_json else None)
    ingest_sanctions(Path(sanctions_file) if sanctions_file else None, sanctions_use_api)
    ingest_ofz()
    ingest_macro(Path(macro_csv) if macro_csv else None, Path(policy_csv) if policy_csv else None)
    freshness_check("fx_rates", 60, "high")
    freshness_check("news_events", 30, "high")
    freshness_check("sanction_links", 1440, "medium")


@flow(name="candles-and-features-flow")
def candles_and_features_flow(
    secids: Optional[list[str]] = None,
    timeframe: str = "1m",
    board: str = "TQBR",
    feature_set: str = "tech_v1",
    window_size: int = 60,
    export_snapshots: bool = True,
    snapshot_output_dir: str = "data/processed/features",
    snapshot_file_format: str = "parquet",
    snapshot_daily_partition: bool = True,
    label_set: str = "basic_v1",
    label_horizons: Optional[list[int]] = None,
    label_take_profit: float = 0.02,
    label_stop_loss: float = 0.01,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> None:
    today = date.today()
    start_dt = parse_date(start_date) if start_date else today - timedelta(days=1)
    end_dt = parse_date(end_date) if end_date else today
    tickers = secids or DEFAULT_CANDLE_SECIDS
    ingest_future = ingest_candles_task.submit(
        tickers,
        timeframe=timeframe,
        board=board,
        start_dt=start_dt,
        end_dt=end_dt,
    )
    features_future = build_feature_windows_task.submit(
        tickers,
        timeframe=timeframe,
        feature_set=feature_set,
        window_size=window_size,
        start_dt=start_dt,
        end_dt=end_dt,
        wait_for=[ingest_future],
    )
    if export_snapshots:
        snapshot_future = export_feature_snapshots_task.submit(
            tickers,
            timeframe=timeframe,
            feature_set=feature_set,
            start_dt=start_dt,
            end_dt=end_dt,
            output_dir=snapshot_output_dir,
            file_format=snapshot_file_format,
            daily_partition=snapshot_daily_partition,
            wait_for=[features_future],
        )
        publish_snapshot_report_task.submit(
            feature_set=feature_set,
            timeframe=timeframe,
            since_iso=datetime.combine(start_dt, time.min, tzinfo=timezone.utc).isoformat(),
            output_path=f"docs/data_quality/train_snapshots_{feature_set}_{timeframe}.json",
            wait_for=[snapshot_future],
        )
    build_trade_labels_task.submit(
        tickers,
        timeframe=timeframe,
        label_set=label_set,
        horizons=label_horizons,
        take_profit=label_take_profit,
        stop_loss=label_stop_loss,
        start_dt=start_dt,
        end_dt=end_dt,
        wait_for=[features_future],
    )


@flow(name="train-temporal-model-flow")
def train_temporal_model_flow(
    secids: Optional[list[str]] = None,
    timeframe: str = "1m",
    feature_set: str = "tech_v1",
    label_set: str = "basic_v1",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    dataset_path: str = "data/training/baseline_dataset.csv",
    seq_len: int = 32,
    batch_size: int = 64,
    max_epochs: int = 15,
    learning_rate: float = 1e-3,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    num_workers: int = 0,
    gradient_clip: float = 1.0,
    accelerator: str = "auto",
    devices: int = 1,
    precision: str = "32-true",
    log_dir: str = "logs/temporal_cnn",
    logger_type: str = "both",
    plot_roc: bool = False,
    mlflow_tracking_uri: Optional[str] = None,
    mlflow_experiment: Optional[str] = None,
    mlflow_run_name: Optional[str] = None,
    mlflow_tags: Optional[list[str]] = None,
    train_grid: Optional[list[dict[str, object]]] = None,
    train_notification_url: Optional[str] = None,
    include_news_features: bool = False,
    news_windows: Optional[list[int]] = None,
    model_type: str = "tcn",
    hidden_dim: int = 128,
    attn_heads: int = 4,
    dropout: float = 0.2,
    walk_forward_splits: Optional[list[dict[str, object]]] = None,
    report_dir: str = "docs/modeling/train_runs",
) -> None:
    today = date.today()
    start_dt = parse_date(start_date) if start_date else today - timedelta(days=30)
    end_dt = parse_date(end_date) if end_date else today
    dataset_future = prepare_baseline_dataset_task.submit(
        secids=[sec.upper() for sec in (secids or DEFAULT_CANDLE_SECIDS)],
        timeframe=timeframe,
        feature_set=feature_set,
        label_set=label_set,
        start_iso=datetime.combine(start_dt, time.min, tzinfo=timezone.utc).isoformat(),
        end_iso=datetime.combine(end_dt, time.max, tzinfo=timezone.utc).isoformat(),
        output_path=dataset_path,
        include_news_features=include_news_features,
        news_windows=news_windows,
    )
    base_params = {
        "seq_len": seq_len,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "learning_rate": learning_rate,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "num_workers": num_workers,
        "gradient_clip": gradient_clip,
        "accelerator": accelerator,
        "devices": devices,
        "precision": precision,
        "log_dir": log_dir,
        "logger_type": logger_type,
        "plot_roc": plot_roc,
        "mlflow_tracking_uri": mlflow_tracking_uri,
        "mlflow_experiment": mlflow_experiment,
        "mlflow_run_name": mlflow_run_name,
        "mlflow_tags": mlflow_tags or [],
        "model_type": model_type,
        "hidden_dim": hidden_dim,
        "attn_heads": attn_heads,
        "dropout": dropout,
        "walk_forward_splits": walk_forward_splits,
        "report_dir": report_dir,
    }
    grid = train_grid or [{}]
    metrics_futures: list[PrefectFuture] = []
    notify_futures: list[PrefectFuture] = []
    for idx, variant in enumerate(grid):
        params = base_params.copy()
        params.update(variant or {})
        extra_tags = params.pop("mlflow_tags", [])
        variant_tags = [f"{k}={v}" for k, v in (variant or {}).items()]
        metrics_future = train_temporal_cnn_task.submit(
            dataset_path=dataset_future,
            mlflow_tags=list(extra_tags) + variant_tags,
            wait_for=[dataset_future],
            **params,
        )
        metrics_futures.append(metrics_future)
        notify_future = notify_training_task.submit(
            metrics=metrics_future,
            notification_url=train_notification_url,
            context={
                "run_index": idx,
                "feature_set": feature_set,
                "label_set": label_set,
                "timeframe": timeframe,
                **variant,
            },
            wait_for=[metrics_future],
        )
        notify_futures.append(notify_future)
    for future in metrics_futures:
        try:
            future.result()
        except Exception as exc:  # pragma: no cover - propagate failure
            logger.error("Training variant failed: %s", exc)
            raise
    for future in notify_futures:
        future.result()
    dataset_future.result()


@flow(name="candles-ingest-flow")
def candles_ingest_flow(
    secids: Optional[list[str]] = None,
    timeframes: Optional[list[str]] = None,
    board: str = "TQBR",
    since: Optional[str] = None,
    until: Optional[str] = None,
    export_dir: Optional[str] = None,
    min_ratio: float = 0.2,
    fail_on_low: bool = False,
) -> None:
    tickers = secids or DEFAULT_CANDLE_SECIDS
    selected_timeframes = timeframes or DEFAULT_TIMEFRAME_SEQUENCE
    today = date.today()
    for timeframe in selected_timeframes:
        lookback_days = TIMEFRAME_LOOKBACK_DAYS.get(timeframe, 3)
        start_dt = parse_date(since) if since else today - timedelta(days=lookback_days)
        end_dt = parse_date(until) if until else today
        stats = ingest_candles_task(
            tickers,
            timeframe=timeframe,
            board=board,
            start_dt=start_dt,
            end_dt=end_dt,
            export_dir=export_dir,
        )
        validate_candle_volume_task(
            stats,
            timeframe,
            start_dt,
            end_dt,
            len(tickers),
            min_ratio,
            fail_on_low,
        )


@flow(name="regime-policy-flow")
def regime_policy_flow(events_json: Optional[str] = None) -> None:
    run_regime_pipeline()
    run_policy_feedback(Path(events_json) if events_json else None)
    freshness_check("market_regimes", 120, "medium")


@flow(name="trade-label-monitoring-flow")
def trade_label_monitoring_flow(
    label_set: str = "basic_v1",
    secids: Optional[list[str]] = None,
    timeframe: Optional[str] = "1m",
    lookback_days: int = 3,
    min_rows: Optional[int] = None,
    fail_on_empty: bool = True,
) -> None:
    monitor_trade_labels_task(
        label_set=label_set,
        secids=secids,
        timeframe=timeframe,
        lookback_days=lookback_days,
        min_rows=min_rows,
        fail_on_empty=fail_on_empty,
    )


if __name__ == "__main__":
    today = date.today()
    yesterday = today - timedelta(days=1)
    daily_ingestion_flow(start_date=str(yesterday), end_date=str(today))
    regime_policy_flow()
