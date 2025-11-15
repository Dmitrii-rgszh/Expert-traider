from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from prefect.client.schemas.schedules import RRuleSchedule
from prefect.filesystems import LocalFileSystem
from prefect.flows import Flow

from backend.scripts.scheduler.prefect_flows import (
    candles_and_features_flow,
    trade_label_monitoring_flow,
    train_temporal_model_flow,
)


DEFAULT_SECIDS = ["SBER", "GAZP", "LKOH", "GMKN"]
DEFAULT_FLOW_NAME = "candles-and-features-flow"
FLOW_REGISTRY: dict[str, dict[str, Any]] = {
    "candles-and-features-flow": {
        "flow": candles_and_features_flow,
        "entrypoint": "backend/scripts/scheduler/prefect_flows.py:candles_and_features_flow",
        "tags": ["candles", "features", "labels"],
    },
    "trade-label-monitoring-flow": {
        "flow": trade_label_monitoring_flow,
        "entrypoint": "backend/scripts/scheduler/prefect_flows.py:trade_label_monitoring_flow",
        "tags": ["monitoring", "trade-labels"],
    },
    "train-temporal-model-flow": {
        "flow": train_temporal_model_flow,
        "entrypoint": "backend/scripts/scheduler/prefect_flows.py:train_temporal_model_flow",
        "tags": ["training", "ml"],
    },
}


def load_flow(flow_name: str, storage_block_name: str | None) -> Flow:
    entry = FLOW_REGISTRY[flow_name]
    if not storage_block_name:
        return entry["flow"]
    storage_block = LocalFileSystem.load(storage_block_name)
    return Flow.from_source(storage_block, entry["entrypoint"])


def build_schedule(timezone_name: str) -> RRuleSchedule:
    return RRuleSchedule(
        rrule="FREQ=DAILY;BYHOUR=7;BYMINUTE=5;BYSECOND=0",
        timezone=timezone_name,
    )


def _resolve_path(path_value: str | Path | None) -> str:
    path = Path(path_value or "").expanduser().resolve()
    return str(path)


def default_parameters(flow_name: str, args: argparse.Namespace) -> dict[str, Any]:
    secids = args.secids or DEFAULT_SECIDS
    train_grid = None
    if args.train_grid_json and args.train_grid_json.exists():
        train_grid = json.loads(args.train_grid_json.read_text(encoding="utf-8"))
    walk_forward = None
    if args.train_walk_forward_json and args.train_walk_forward_json.exists():
        walk_forward = json.loads(args.train_walk_forward_json.read_text(encoding="utf-8"))
    if flow_name == "candles-and-features-flow":
        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=args.window_days)
        return {
            "secids": secids,
            "timeframe": "1m",
            "board": "TQBR",
            "feature_set": "tech_v1",
            "window_size": 60,
            "export_snapshots": not args.disable_snapshot_export,
            "snapshot_output_dir": args.snapshot_output_dir,
            "snapshot_file_format": args.snapshot_file_format,
            "snapshot_daily_partition": args.snapshot_daily_partition,
            "label_set": "basic_v1",
            "label_horizons": [60, 240, 1440],
            "label_take_profit": 0.02,
            "label_stop_loss": 0.01,
            "start_date": str(start_date),
            "end_date": str(end_date),
        }
    if flow_name == "trade-label-monitoring-flow":
        return {
            "label_set": args.monitor_label_set,
            "secids": secids,
            "timeframe": args.monitor_timeframe,
            "lookback_days": args.monitor_lookback_days,
            "min_rows": args.monitor_min_rows,
            "fail_on_empty": args.monitor_fail_on_empty,
        }
    if flow_name == "train-temporal-model-flow":
        end_date = args.train_end_date or datetime.now(timezone.utc).date().isoformat()
        start_date = args.train_start_date or (datetime.now(timezone.utc).date() - timedelta(days=args.window_days)).isoformat()
        dataset_path = _resolve_path(args.train_dataset_path)
        log_dir = _resolve_path(args.train_log_dir)
        report_dir = _resolve_path(args.train_report_dir)
        return {
            "secids": secids,
            "timeframe": args.train_timeframe,
            "feature_set": args.train_feature_set,
            "label_set": args.train_label_set,
            "start_date": start_date,
            "end_date": end_date,
            "dataset_path": dataset_path,
            "seq_len": args.train_seq_len,
            "batch_size": args.train_batch_size,
            "max_epochs": args.train_max_epochs,
            "learning_rate": args.train_learning_rate,
            "train_ratio": args.train_ratio,
            "val_ratio": args.train_val_ratio,
            "num_workers": args.train_num_workers,
            "gradient_clip": args.train_gradient_clip,
            "accelerator": args.train_accelerator,
            "devices": args.train_devices,
            "precision": args.train_precision,
            "log_dir": log_dir,
            "logger_type": args.train_logger_type,
            "plot_roc": args.train_plot_roc,
            "mlflow_tracking_uri": args.train_mlflow_tracking_uri,
            "mlflow_experiment": args.train_mlflow_experiment,
            "mlflow_run_name": args.train_mlflow_run_name,
            "mlflow_tags": args.train_mlflow_tags,
            "train_grid": train_grid,
            "train_notification_url": args.train_notification_url,
            "include_news_features": args.train_include_news_features,
            "news_windows": args.train_news_windows,
            "model_type": args.train_model_type,
            "hidden_dim": args.train_hidden_dim,
            "attn_heads": args.train_attn_heads,
            "dropout": args.train_dropout,
            "walk_forward_splits": walk_forward,
            "report_dir": report_dir,
        }
    raise ValueError(f"Unsupported flow '{flow_name}'")


def load_parameters(path: Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def register_deployment(
    name: str,
    flow_name: str,
    work_pool: str | None,
    work_queue: str | None,
    schedule_tz: str,
    parameters: dict[str, Any],
    dry_run: bool,
    storage_block: str | None,
    ignore_warnings: bool,
    tags: list[str],
) -> None:
    schedule = build_schedule(schedule_tz)
    flow_obj = load_flow(flow_name, storage_block)
    if dry_run:
        print("Dry run: deployment configuration preview")
        preview = {
            "name": name,
            "flow_name": flow_name,
            "parameters": parameters,
            "schedule": schedule.dict(),
            "work_pool": work_pool,
            "work_queue": work_queue,
            "tags": tags,
            "storage_block": storage_block,
            "ignore_warnings": ignore_warnings,
        }
        print(json.dumps(preview, indent=2, default=str))
        return
    deployment_id = flow_obj.deploy(
        name=name,
        parameters=parameters,
        schedules=[schedule],
        work_pool_name=work_pool,
        work_queue_name=work_queue,
        tags=tags,
        build=False,
        push=False,
        ignore_warnings=ignore_warnings,
    )
    print(
        f"Deployment '{name}' registered for flow '{flow_name}' (queue={work_queue or 'default'}, pool={work_pool or 'default'}) as {deployment_id}."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register Prefect deployments for scheduler flows")
    parser.add_argument(
        "--name",
        default="candles-and-features-daily",
        help="Deployment name",
    )
    parser.add_argument(
        "--flow-name",
        choices=sorted(FLOW_REGISTRY.keys()),
        default=DEFAULT_FLOW_NAME,
        help="Prefect flow to deploy",
    )
    parser.add_argument(
        "--secids",
        nargs="+",
        default=DEFAULT_SECIDS,
        help="Tickers to process",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=5,
        help="How many calendar days to backfill per run",
    )
    parser.add_argument(
        "--train-timeframe",
        default="1m",
        help="Timeframe for training dataset (train-temporal-model-flow)",
    )
    parser.add_argument(
        "--train-feature-set",
        default="tech_v1",
        help="Feature set for training dataset",
    )
    parser.add_argument(
        "--train-label-set",
        default="basic_v1",
        help="Label set for training dataset",
    )
    parser.add_argument("--train-start-date", help="Override start date (YYYY-MM-DD) for training flow")
    parser.add_argument("--train-end-date", help="Override end date (YYYY-MM-DD) for training flow")
    parser.add_argument(
        "--train-dataset-path",
        default="data/training/baseline_dataset.csv",
        help="Where to store merged dataset before training",
    )
    parser.add_argument("--train-seq-len", type=int, default=32, help="Temporal CNN sequence length")
    parser.add_argument("--train-batch-size", type=int, default=64, help="Temporal CNN batch size")
    parser.add_argument("--train-max-epochs", type=int, default=15, help="Temporal CNN epochs")
    parser.add_argument("--train-learning-rate", type=float, default=1e-3, help="Temporal CNN learning rate")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio")
    parser.add_argument("--train-val-ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--train-num-workers", type=int, default=0, help="Dataloader workers")
    parser.add_argument("--train-gradient-clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--train-accelerator", default="auto", help="Trainer accelerator (cpu/gpu/auto)")
    parser.add_argument("--train-devices", type=int, default=1, help="Number of devices")
    parser.add_argument("--train-precision", default="32-true", help="Precision setting (e.g. 16-mixed)")
    parser.add_argument("--train-log-dir", default="logs/temporal_cnn", help="Directory for training logs")
    parser.add_argument(
        "--train-model-type",
        choices=["tcn", "tft"],
        default="tcn",
        help="Model architecture for train-temporal-model-flow",
    )
    parser.add_argument("--train-hidden-dim", type=int, default=128, help="Hidden dimension for TFT model")
    parser.add_argument("--train-attn-heads", type=int, default=4, help="Attention heads for TFT model")
    parser.add_argument("--train-dropout", type=float, default=0.2, help="Dropout for temporal models")
    parser.add_argument(
        "--train-report-dir",
        default="docs/modeling/train_runs",
        help="Directory to store JSON training reports",
    )
    parser.add_argument(
        "--train-logger-type",
        choices=["csv", "tensorboard", "both"],
        default="both",
        help="Which loggers to enable",
    )
    parser.add_argument(
        "--train-walk-forward-json",
        type=Path,
        help="Path to JSON file describing walk-forward split windows",
    )
    parser.add_argument("--train-plot-roc", action="store_true", help="Store ROC curve after training")
    parser.add_argument("--train-mlflow-tracking-uri", help="MLflow tracking URI for training flow")
    parser.add_argument("--train-mlflow-experiment", help="MLflow experiment for training flow")
    parser.add_argument("--train-mlflow-run-name", help="MLflow run name for training flow")
    parser.add_argument(
        "--train-mlflow-tags",
        nargs="*",
        help="MLflow tags (key=value) for training flow",
    )
    parser.add_argument(
        "--train-grid-json",
        type=Path,
        help="Path to JSON file with grid search variants (list of objects)",
    )
    parser.add_argument(
        "--train-include-news-features",
        action="store_true",
        help="Include aggregated news/risk counters in training dataset",
    )
    parser.add_argument(
        "--train-news-windows",
        nargs="+",
        type=int,
        help="List of rolling windows (minutes) for news features (e.g. 60 240 1440)",
    )
    parser.add_argument(
        "--train-notification-url",
        help="Webhook URL (Slack/Telegram) for training notifications",
    )
    parser.add_argument(
        "--disable-snapshot-export",
        action="store_true",
        help="Disable feature snapshot export in the candles-and-features flow",
    )
    parser.add_argument(
        "--snapshot-output-dir",
        default="data/processed/features",
        help="Where to store exported feature snapshots",
    )
    parser.add_argument(
        "--snapshot-file-format",
        choices=["parquet", "feather"],
        default="parquet",
        help="Snapshot file format",
    )
    parser.add_argument(
        "--snapshot-daily-partition",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Partition snapshots per day (default: true)",
    )
    parser.add_argument(
        "--schedule-tz",
        default="Europe/Moscow",
        help="Timezone for schedule",
    )
    parser.add_argument(
        "--parameters-json",
        type=Path,
        help="Optional JSON file overriding deployment parameters",
    )
    parser.add_argument(
        "--work-pool",
        help="Prefect work pool name",
    )
    parser.add_argument(
        "--work-queue",
        help="Prefect work queue name",
    )
    parser.add_argument(
        "--storage-block",
        default="local-repo-storage",
        help="Prefect block name (LocalFileSystem) that stores the repo",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually register deployment (otherwise dry run)",
    )
    parser.add_argument(
        "--ignore-warnings",
        action="store_true",
        help="Suppress Prefect CLI warnings (e.g. process work pool hint)",
    )
    parser.add_argument(
        "--monitor-label-set",
        default="basic_v1",
        help="Label set for trade-label monitoring flow",
    )
    parser.add_argument(
        "--monitor-timeframe",
        default="1m",
        help="Timeframe filter for monitoring flow (default: 1m)",
    )
    parser.add_argument(
        "--monitor-lookback-days",
        type=int,
        default=3,
        help="Lookback days for monitoring flow",
    )
    parser.add_argument(
        "--monitor-min-rows",
        type=int,
        help="Minimum acceptable rows for monitoring flow",
    )
    parser.add_argument(
        "--monitor-fail-on-empty",
        action="store_true",
        help="Fail monitoring flow deployment runs if no rows are detected",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.secids = [sec.upper() for sec in (args.secids or [])] or DEFAULT_SECIDS
    params = load_parameters(args.parameters_json) or default_parameters(args.flow_name, args)
    tags = FLOW_REGISTRY[args.flow_name]["tags"]
    register_deployment(
        name=args.name,
        flow_name=args.flow_name,
        work_pool=args.work_pool,
        work_queue=args.work_queue,
        schedule_tz=args.schedule_tz,
        parameters=params,
        dry_run=not args.apply,
        storage_block=args.storage_block,
        ignore_warnings=args.ignore_warnings,
        tags=tags,
    )


if __name__ == "__main__":
    main()
