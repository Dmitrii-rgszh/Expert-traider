from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query

from ..ml.service import get_strong_q80_super_trader
from backend.scripts.ingestion.base import HttpSource, fetch_json

router = APIRouter()


LONG_THRESHOLD = 0.005
SHORT_THRESHOLD = 0.001
MOEX_CANDLES_URL = (
    "https://iss.moex.com/iss/engines/stock/markets/shares/securities/{secid}/candles.json"
)
MOEX_SECURITY_URL = "https://iss.moex.com/iss/securities/{secid}.json"


def _bucket_volatility(value: Any) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if v < 0.0005:
        return "low_vol"
    if v < 0.0015:
        return "mid_vol"
    return "high_vol"


def _bucket_liquidity(value: Any) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if v < -0.5:
        return "low_liq"
    if v < 0.5:
        return "mid_liq"
    return "high_liq"


def _bucket_time_of_day(signal_time: Any) -> str:
    if not signal_time:
        return "unknown"
    text = str(signal_time)
    try:
        # handle both ...Z and +0000 formats
        if text.endswith("Z"):
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    except Exception:
        return "unknown"
    hour = dt.hour
    if 10 <= hour < 12:
        return "open"
    if 12 <= hour < 15:
        return "midday"
    if 15 <= hour <= 19:
        return "close"
    return "other"


def _moex_fetch_recent_1m_candles(secid: str, days: int = 5) -> List[dict]:
    """
    Fetch recent 1m candles for secid from MOEX ISS.
    Returns list of dicts with timestamp, close, volume.
    """
    end = date.today()
    start = end - timedelta(days=days)
    params = {
        "from": start.isoformat(),
        "till": end.isoformat(),
        "interval": 1,
        "start": 0,
        "boardid": "TQBR",
    }
    source = HttpSource(url=MOEX_CANDLES_URL.format(secid=secid), params=params)
    payload = fetch_json(source)
    if not payload or "candles" not in payload:
        return []
    candles_payload = payload["candles"]
    rows = candles_payload.get("data", [])
    columns = candles_payload.get("columns", [])
    if not rows:
        return []
    idx = {name: i for i, name in enumerate(columns)}
    out: List[dict] = []
    for row in rows:
        ts_raw = row[idx["begin"]]
        ts = datetime.fromisoformat(ts_raw)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        volume_idx = idx.get("volume")
        out.append(
            {
                "timestamp": ts,
                "close": float(row[idx["close"]]),
                "volume": float(row[volume_idx]) if volume_idx is not None else 0.0,
            }
        )
    return out


_LOT_CACHE: Dict[str, float] = {}


def _moex_get_lot_size(secid: str) -> float:
    secid = secid.upper()
    if secid in _LOT_CACHE:
        return _LOT_CACHE[secid]
    source = HttpSource(url=MOEX_SECURITY_URL.format(secid=secid), params={"iss.meta": "off"})
    payload = fetch_json(source)
    lot = 1.0
    try:
        if payload and "securities" in payload:
            data = payload["securities"].get("data", [])
            columns = payload["securities"].get("columns", [])
            idx = {name: i for i, name in enumerate(columns)}
            lotsize_idx = idx.get("LOTSIZE")
            if lotsize_idx is not None and data:
                raw = data[0][lotsize_idx]
                lot = float(raw) if raw not in (None, "") else 1.0
    except Exception:
        lot = 1.0
    _LOT_CACHE[secid] = lot
    return lot


@router.post("/trader/strong_q80/evaluate")
def evaluate_strong_q80(payload: Dict[str, Any]) -> Dict[str, float]:
    """
    Evaluate strong_q80 models and return raw probabilities.

    Payload should include all feature columns expected by the super-trader
    service (including meta like secid and signal_time).
    """
    service = get_strong_q80_super_trader()
    try:
        scores = service.predict(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return scores


@router.post("/trader/strong_q80/decision")
def decide_strong_q80(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply regime-aware decision policy on top of strong_q80 probabilities.

    Returns p_long, p_short, chosen action and simple regime buckets.
    """
    try:
        service = get_strong_q80_super_trader()
        scores = service.predict(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"strong_q80_decision_error: {exc!r}") from exc

    p_long = float(scores.get("p_long", 0.0))
    p_short = float(scores.get("p_short", 0.0))

    vol_bucket = _bucket_volatility(payload.get("volatility_20"))
    liq_bucket = _bucket_liquidity(payload.get("volume_zscore_20"))
    tod_bucket = _bucket_time_of_day(payload.get("signal_time"))

    long_ok = (
        p_long >= LONG_THRESHOLD
        and vol_bucket in {"mid_vol", "high_vol"}
        and liq_bucket in {"mid_liq", "high_liq"}
        and tod_bucket in {"open", "other"}
    )
    short_ok = p_short >= SHORT_THRESHOLD

    action = "HOLD"
    if long_ok and not short_ok:
        action = "OPEN_LONG"
    elif short_ok and not long_ok:
        action = "OPEN_SHORT"
    elif long_ok and short_ok:
        # break ties by higher probability
        action = "OPEN_LONG" if p_long >= p_short else "OPEN_SHORT"

    return {
        "p_long": p_long,
        "p_short": p_short,
        "action": action,
        "regime": {
            "vol_bucket": vol_bucket,
            "liq_bucket": liq_bucket,
            "tod_bucket": tod_bucket,
        },
    }


@router.get("/trader/strong_q80/live_decision")
def live_decide_strong_q80(
    secid: str = Query(..., min_length=1, description="MOEX ticker, e.g. SBER"),
) -> Dict[str, Any]:
    """
    Live decision using latest MOEX 1m candles for the given ticker.
    Applies the same regime-aware policy as /decision.
    """
    try:
        # Check Moscow exchange hours (approx. 10:00-18:45 MSK, weekdays)
        from zoneinfo import ZoneInfo

        now_utc = datetime.now(timezone.utc)
        now_msk = now_utc.astimezone(ZoneInfo("Europe/Moscow"))
        market_open = (
            now_msk.weekday() < 5 and (10 <= now_msk.hour < 18 or (now_msk.hour == 18 and now_msk.minute <= 45))
        )

        if not market_open:
            return {
                "market_open": False,
                "message": "Рынок МОЕХ сейчас закрыт",
                "action": "HOLD",
            }

        candles = _moex_fetch_recent_1m_candles(secid)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"strong_q80_live_error: {exc!r}") from exc
    if len(candles) < 2:
        raise HTTPException(status_code=400, detail="Недостаточно данных по тикеру для live-решения")

    candles_sorted = sorted(candles, key=lambda x: x["timestamp"])
    last = candles_sorted[-1]
    prev = candles_sorted[-2]

    # simple return and volatility/volume features
    try:
        ret_1 = (last["close"] / prev["close"]) - 1.0
    except ZeroDivisionError:
        ret_1 = 0.0

    returns = []
    volumes = []
    for a, b in zip(candles_sorted[1:], candles_sorted[:-1]):
        try:
            returns.append((a["close"] / b["close"]) - 1.0)
        except ZeroDivisionError:
            returns.append(0.0)
        volumes.append(a["volume"])

    import numpy as np

    tail = returns[-20:] or returns
    vol_tail = volumes[-20:] or volumes
    volatility_20 = float(np.std(tail)) if tail else 0.0
    if vol_tail and np.std(vol_tail) > 0:
        volume_zscore_20 = float((vol_tail[-1] - np.mean(vol_tail)) / np.std(vol_tail))
    else:
        volume_zscore_20 = 0.0

    lot_size = _moex_get_lot_size(secid)

    row: Dict[str, Any] = {
        "secid": secid.upper(),
        "timeframe": "1m",
        "feature_set": "live",
        "label_set": "intraday_v2",
        "signal_time": last["timestamp"].isoformat(),
        "horizon_minutes": 30,
        "forward_return_pct": 0.0,
        "max_runup_pct": 0.0,
        "max_drawdown_pct": 0.0,
        "label_long": 0,
        "label_short": 0,
        "long_pnl_pct": ret_1,
        "short_pnl_pct": -ret_1,
        "return_1": ret_1,
        "return_vs_imoex": 0.0,
        "return_vs_rts": 0.0,
        "volatility_20": volatility_20,
        "volume_zscore_20": volume_zscore_20,
    }

    service = get_strong_q80_super_trader()
    scores = service.predict(row)
    p_long = float(scores.get("p_long", 0.0))
    p_short = float(scores.get("p_short", 0.0))

    vol_bucket = _bucket_volatility(row["volatility_20"])
    liq_bucket = _bucket_liquidity(row["volume_zscore_20"])
    tod_bucket = _bucket_time_of_day(row["signal_time"])

    long_ok = (
        p_long >= LONG_THRESHOLD
        and vol_bucket in {"mid_vol", "high_vol"}
        and liq_bucket in {"mid_liq", "high_liq"}
        and tod_bucket in {"open", "other"}
    )
    short_ok = p_short >= SHORT_THRESHOLD

    action = "HOLD"
    if long_ok and not short_ok:
        action = "OPEN_LONG"
    elif short_ok and not long_ok:
        action = "OPEN_SHORT"
    elif long_ok and short_ok:
        action = "OPEN_LONG" if p_long >= p_short else "OPEN_SHORT"

    return {
        "market_open": True,
        "secid": secid.upper(),
        "p_long": p_long,
        "p_short": p_short,
        "action": action,
        "features": {
            "signal_time": row["signal_time"],
            "last_price": last["close"],
            "lot_size": lot_size,
            "return_1": row["return_1"],
            "volatility_20": row["volatility_20"],
            "volume_zscore_20": row["volume_zscore_20"],
        },
        "regime": {
            "vol_bucket": vol_bucket,
            "liq_bucket": liq_bucket,
            "tod_bucket": tod_bucket,
        },
    }
