from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from sqlalchemy import func

from backend.app.db.session import SessionLocal
from backend.app.models import EtlJob, TradeLabel


def _ensure_tz(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _write_etl_job(
    started_at: datetime,
    status: str,
    rows_processed: int,
    label_set: str,
    secids: list[str] | None,
    timeframe: str | None,
    lookback_days: int,
    error: str | None = None,
) -> None:
    with SessionLocal() as session:
        job = EtlJob(
            job_name="check_trade_labels",
            scheduled_at=None,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            status=status,
            rows_processed=rows_processed,
            error=error,
            metadata_json={
                "label_set": label_set,
                "secids": secids,
                "timeframe": timeframe,
                "lookback_days": lookback_days,
            },
        )
        session.add(job)
        session.commit()


def run_check(
    label_set: str,
    secids: list[str] | None,
    timeframe: str | None,
    lookback_days: int,
    *,
    fail_on_empty: bool = False,
    min_rows: int | None = None,
) -> Dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    session = SessionLocal()
    summary: Dict[str, Any] = {
        "label_set": label_set,
        "secids": secids,
        "timeframe": timeframe,
        "lookback_days": lookback_days,
        "status": "pending",
        "rows": 0,
    }
    try:
        cutoff_end = datetime.now(timezone.utc)
        cutoff_start = cutoff_end - timedelta(days=lookback_days)
        query = session.query(
            TradeLabel.secid,
            TradeLabel.timeframe,
            TradeLabel.horizon_minutes,
            func.count().label("rows"),
            func.min(TradeLabel.signal_time).label("min_time"),
            func.max(TradeLabel.signal_time).label("max_time"),
            func.min(TradeLabel.forward_return_pct).label("min_return"),
            func.max(TradeLabel.forward_return_pct).label("max_return"),
        ).filter(
            TradeLabel.label_set == label_set,
            TradeLabel.signal_time >= cutoff_start,
            TradeLabel.signal_time <= cutoff_end,
        )
        if secids:
            query = query.filter(TradeLabel.secid.in_(secids))
        if timeframe:
            query = query.filter(TradeLabel.timeframe == timeframe)
        query = query.group_by(
            TradeLabel.secid,
            TradeLabel.timeframe,
            TradeLabel.horizon_minutes,
        ).order_by(
            TradeLabel.secid.asc(),
            TradeLabel.timeframe.asc(),
            TradeLabel.horizon_minutes.asc(),
        )
        results = query.all()
        if not results:
            print("No trade_labels found for specified filters.")
            _write_etl_job(
                started_at,
                status="empty",
                rows_processed=0,
                label_set=label_set,
                secids=secids,
                timeframe=timeframe,
                lookback_days=lookback_days,
            )
            summary["status"] = "empty"
            if fail_on_empty:
                raise RuntimeError("No trade_labels found for specified filters.")
            return summary
        header = (
            f"Label set: {label_set} | timeframe: {timeframe or 'ALL'} | lookback days: {lookback_days}"
        )
        print(header)
        print("=" * len(header))
        total_rows = 0
        for row in results:
            min_time = _ensure_tz(row.min_time)
            max_time = _ensure_tz(row.max_time)
            total_rows += row.rows or 0
            print(
                (
                    "{secid:<6} {tf:<4} horizon={h:>4}m rows={rows:<6} "
                    "span=({min:%Y-%m-%d %H:%M}, {max:%Y-%m-%d %H:%M}) returns=({ret_min:.2f}, {ret_max:.2f})"
                ).format(
                    secid=row.secid,
                    tf=row.timeframe,
                    h=row.horizon_minutes,
                    rows=row.rows,
                    min=min_time,
                    max=max_time,
                    ret_min=row.min_return or 0,
                    ret_max=row.max_return or 0,
                )
            )
        _write_etl_job(
            started_at,
            status="success",
            rows_processed=total_rows,
            label_set=label_set,
            secids=secids,
            timeframe=timeframe,
            lookback_days=lookback_days,
        )
        summary["status"] = "success"
        summary["rows"] = total_rows
        summary["groups"] = len(results)
        if min_rows is not None and total_rows < min_rows:
            raise RuntimeError(
                f"Trade label rows ({total_rows}) fell below the minimum threshold ({min_rows})."
            )
        return summary
    except Exception as exc:
        _write_etl_job(
            started_at,
            status="failed",
            rows_processed=0,
            label_set=label_set,
            secids=secids,
            timeframe=timeframe,
            lookback_days=lookback_days,
            error=str(exc),
        )
        summary["status"] = "failed"
        summary["error"] = str(exc)
        raise
    finally:
        session.close()
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity-check trade_labels coverage")
    parser.add_argument("--label-set", default="basic_v1", help="Label set identifier")
    parser.add_argument("--secids", nargs="*", help="Subset of tickers to inspect")
    parser.add_argument("--timeframe", help="Filter by timeframe")
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=7,
        help="Days to inspect counting back from now",
    )
    parser.add_argument(
        "--fail-on-empty",
        action="store_true",
        help="Exit with error if no rows found",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        help="Fail if total rows within the lookback fall below this threshold",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    secids = [sec.upper() for sec in args.secids] if args.secids else None
    run_check(
        args.label_set,
        secids,
        args.timeframe,
        args.lookback_days,
        fail_on_empty=args.fail_on_empty,
        min_rows=args.min_rows,
    )


if __name__ == "__main__":
    main()
