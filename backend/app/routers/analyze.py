import json
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..db.session import get_db
from ..models import User, News, AnalysisResult
from ..schemas.analyze import AnalyzeRequest, AnalyzeResponse, HistoryItem
from ..ml import get_news_analyzer
from ..core.config import get_settings
from ..core.telegram import TelegramAuthError, validate_init_data

router = APIRouter()
news_analyzer = get_news_analyzer()


def _resolve_telegram_id(init_data: str | None, claimed_telegram_id: str | None) -> str | None:
    settings = get_settings()
    bot_token = settings.telegram_bot_token
    if not bot_token:
        return claimed_telegram_id

    if not init_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Telegram init data required",
        )

    try:
        payload = validate_init_data(
            init_data,
            bot_token,
            settings.telegram_init_max_age_seconds,
        )
    except TelegramAuthError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc

    user_payload = payload.get("user")
    if not isinstance(user_payload, dict) or "id" not in user_payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Telegram user payload missing",
        )

    resolved_id = str(user_payload["id"])
    if claimed_telegram_id and claimed_telegram_id != resolved_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Telegram ID mismatch",
        )
    return resolved_id


def _enforce_request_cap(db: Session, user: User) -> None:
    settings = get_settings()
    cap = settings.telegram_daily_request_cap
    if cap <= 0:
        return

    window_seconds = max(settings.telegram_request_window_seconds, 0)
    cutoff: datetime | None = None
    if window_seconds > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=window_seconds)

    count_query = (
        db.query(func.count(AnalysisResult.id))
        .join(News, AnalysisResult.news_id == News.id)
        .filter(News.user_id == user.id)
    )
    if cutoff is not None:
        count_query = count_query.filter(AnalysisResult.created_at >= cutoff)

    request_count = count_query.scalar() or 0
    if request_count >= cap:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Request limit exceeded",
        )


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_news(payload: AnalyzeRequest, db: Session = Depends(get_db)) -> AnalyzeResponse:
    telegram_id = _resolve_telegram_id(payload.init_data, payload.telegram_id)

    # Find or create user by telegram_id (optional for now)
    user: User | None = None
    if telegram_id:
        user = db.query(User).filter(User.telegram_id == telegram_id).first()
        if user is None:
            user = User(telegram_id=telegram_id)
            db.add(user)
            db.flush()
        _enforce_request_cap(db, user)

    # Store news
    news = News(
        user_id=user.id if user else None,
        source_url=payload.source_url,
        title=payload.title,
        text=payload.text,
    )
    db.add(news)
    db.flush()

    inference = news_analyzer.analyze(text=payload.text, title=payload.title)

    analysis = AnalysisResult(
        news_id=news.id,
        trigger_score=inference.trigger_score,
        direction=inference.direction,
        event_type=inference.event_type,
        horizon=inference.horizon,
        assets_json=json.dumps(inference.assets),
        summary=inference.summary,
    )
    db.add(analysis)
    db.commit()

    return AnalyzeResponse(
        news_id=news.id,
        analysis_id=analysis.id,
        trigger_score=inference.trigger_score,
        direction=inference.direction,
        event_type=inference.event_type,
        horizon=inference.horizon,
        assets=inference.assets,
        summary=inference.summary,
    )


@router.get("/history", response_model=list[HistoryItem])
def get_history(
    telegram_id: str = Query(..., min_length=1),
    limit: int = Query(20, ge=1, le=100),
    init_data: str | None = Query(None),
    db: Session = Depends(get_db),
) -> list[HistoryItem]:
    resolved_telegram_id = _resolve_telegram_id(init_data, telegram_id)
    if not resolved_telegram_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Telegram ID required")

    user = db.query(User).filter(User.telegram_id == resolved_telegram_id).first()
    if user is None:
        return []

    records = (
        db.query(AnalysisResult, News)
            .join(News, AnalysisResult.news_id == News.id)
            .filter(News.user_id == user.id)
            .order_by(AnalysisResult.created_at.desc())
            .limit(limit)
            .all()
    )

    history: list[HistoryItem] = []
    for analysis, news in records:
        assets = []
        if analysis.assets_json:
            try:
                assets = json.loads(analysis.assets_json)
            except json.JSONDecodeError:
                assets = []
        history.append(
            HistoryItem(
                news_id=news.id,
                analysis_id=analysis.id,
                created_at=analysis.created_at,
                title=news.title,
                text=news.text,
                trigger_score=analysis.trigger_score,
                direction=analysis.direction,
                event_type=analysis.event_type,
                horizon=analysis.horizon,
                assets=assets,
                summary=analysis.summary,
            )
        )
    return history
