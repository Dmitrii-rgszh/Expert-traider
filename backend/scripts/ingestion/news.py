from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import httpx
from sqlalchemy import select

if __package__ in (None, ""):  # pragma: no cover - allows `python backend/.../news.py`
    import sys

    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))

from backend.app.models import GlobalRiskEvent, NewsEvent, NewsSource, RiskAlert

try:  # pragma: no cover - fallback for relative import
    from .base import HttpSource, IngestionStats, fetch_json, session_scope, logger
except ImportError:  # pragma: no cover
    from backend.scripts.ingestion.base import HttpSource, IngestionStats, fetch_json, session_scope, logger

MOEX_ISS_BASE_URL = os.getenv("MOEX_ISS_BASE_URL", "https://iss.moex.com/iss")
MOEX_NEWS_LANG = os.getenv("MOEX_NEWS_LANG", "ru")
MOEX_NEWS_LIMIT = int(os.getenv("MOEX_NEWS_LIMIT", "100"))
MOEX_NEWS_CATEGORY = os.getenv("MOEX_NEWS_CATEGORY")
MOEX_API_USER = os.getenv("MOEX_API_USER")
MOEX_API_PASSWORD = os.getenv("MOEX_API_PASSWORD")
MOEX_HTTP_TIMEOUT = float(os.getenv("MOEX_HTTP_TIMEOUT", "15"))


class MoexNewsClient:
    """Small helper around ISS sitenews feed."""

    def __init__(
        self,
        base_url: str = MOEX_ISS_BASE_URL,
        lang: str = MOEX_NEWS_LANG,
        limit: int = MOEX_NEWS_LIMIT,
        category: Optional[str] = MOEX_NEWS_CATEGORY,
        username: Optional[str] = MOEX_API_USER,
        password: Optional[str] = MOEX_API_PASSWORD,
        timeout: float = MOEX_HTTP_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.lang = lang
        self.limit = limit
        self.category = category
        self.timeout = timeout
        self.auth = httpx.BasicAuth(username, password) if username and password else None

    def fetch(self) -> list[dict[str, Any]]:
        params = {"lang": self.lang, "limit": self.limit}
        if self.category:
            params["category"] = self.category
        url = f"{self.base_url}/sitenews.json"
        with httpx.Client(timeout=self.timeout, auth=self.auth) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            payload = response.json()
            block = payload.get("sitenews", {})
            columns = block.get("columns", [])
            data = block.get("data", [])
            idx = {name: i for i, name in enumerate(columns)}
            events: list[dict[str, Any]] = []
            for row in data:
                if not row:
                    continue
                id_pos = idx.get("id")
                news_id = row[id_pos] if id_pos is not None else row[0]
                detail = self._fetch_detail(client, news_id)
                events.append(self._to_event(row, idx, detail))
        return events

    def _fetch_detail(self, client: httpx.Client, news_id: Any) -> dict[str, Any]:
        try:
            resp = client.get(f"{self.base_url}/sitenews/{news_id}.json", params={"lang": self.lang})
            resp.raise_for_status()
        except httpx.HTTPError:
            return {}
        document = resp.json()
        block = document.get("sitenews") or document.get("newsitem") or {}
        columns = block.get("columns", [])
        data = block.get("data", [])
        if not data:
            return {}
        idx = {name: i for i, name in enumerate(columns)}
        row = data[0]
        return {
            "id": self._safe_value(row, idx, "id"),
            "title": self._safe_value(row, idx, "title"),
            "body": self._safe_value(row, idx, "body") or self._safe_value(row, idx, "content"),
            "published": self._safe_value(row, idx, "published"),
            "url": self._safe_value(row, idx, "url"),
            "source": self._safe_value(row, idx, "source"),
            "tags": self._safe_value(row, idx, "tags") or [],
            "securities": self._safe_value(row, idx, "securities"),
        }

    def _to_event(self, row: list[Any], idx: dict[str, int], detail: dict[str, Any]) -> dict[str, Any]:
        published = detail.get("published") or self._safe_value(row, idx, "published")
        title = detail.get("title") or self._safe_value(row, idx, "title") or "MOEX news"
        body = detail.get("body") or self._safe_value(row, idx, "content") or title
        url = detail.get("url") or self._safe_value(row, idx, "url")
        secid = detail.get("secid") or detail.get("security") or detail.get("securities")
        if isinstance(secid, list):
            secid = secid[0]
        return {
            "external_id": str(detail.get("id") or self._safe_value(row, idx, "id")),
            "title": title,
            "body": body,
            "published_at": published,
            "source_code": "moex_sitenews",
            "provider": "moex",
            "geo_scope": "RU",
            "source_ref": url,
            "secid": secid,
            "tags": detail.get("tags") or self._safe_value(row, idx, "tags") or [],
            "raw_payload": {
                "summary_row": row,
                "detail": detail,
            },
        }

    @staticmethod
    def _safe_value(row: list[Any], idx: dict[str, int], column: str, default: Any | None = None) -> Any | None:
        position = idx.get(column)
        if position is None:
            return default
        try:
            return row[position]
        except (IndexError, TypeError):
            return default

    @classmethod
    def from_args(
        cls,
        limit: Optional[int] = None,
        lang: Optional[str] = None,
        category: Optional[str] = None,
    ) -> "MoexNewsClient":
        return cls(
            limit=limit or MOEX_NEWS_LIMIT,
            lang=lang or MOEX_NEWS_LANG,
            category=category if category is not None else MOEX_NEWS_CATEGORY,
        )


def load_events(
    source: Optional[HttpSource],
    file_path: Optional[Path],
    moex_client: Optional[MoexNewsClient],
) -> list[dict[str, Any]]:
    if file_path:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
        return raw.get("events", raw) if isinstance(raw, dict) else raw
    if moex_client:
        try:
            return moex_client.fetch()
        except httpx.HTTPError as exc:
            logger.warning("Failed to load MOEX news: %s", exc)
            return []
    if source:
        payload = fetch_json(source)
        if not payload:
            return []
        return payload.get("events", payload)
    return []


class NewsIngestor:
    def __init__(
        self,
        source: Optional[HttpSource] = None,
        file_path: Optional[Path] = None,
        moex_client: Optional[MoexNewsClient] = None,
    ) -> None:
        self.source = source
        self.file_path = file_path
        self.moex_client = moex_client

    def run(self, dry_run: bool = False) -> IngestionStats:
        stats = IngestionStats()
        events = load_events(self.source, self.file_path, self.moex_client)
        if not events:
            logger.warning("No news events to ingest")
            return stats

        with session_scope() as session:
            for raw in events:
                stats.processed += 1
                source = self._ensure_source(session, raw)
                payload = self._to_payload(raw, source.id if source else None)
                if dry_run:
                    continue
                stmt = select(NewsEvent).where(
                    NewsEvent.external_id == payload.get("external_id"),
                    NewsEvent.source_id == payload.get("source_id"),
                )
                existing = session.execute(stmt).scalar_one_or_none()
                if existing:
                    for key, value in payload.items():
                        setattr(existing, key, value)
                    stats.updated += 1
                else:
                    session.add(NewsEvent(**payload))
                    stats.inserted += 1

                if raw.get("severity") is not None:
                    event_id = self._upsert_global_event(session, raw)
                    self._maybe_raise_alert(session, raw, event_id)
        return stats

    def _ensure_source(self, session, raw: dict[str, Any]) -> Optional[NewsSource]:
        code = raw.get("source_code") or "generic"
        stmt = select(NewsSource).where(NewsSource.code == code)
        source = session.execute(stmt).scalar_one_or_none()
        if source:
            return source
        source = NewsSource(
            code=code,
            provider=raw.get("provider"),
            geo_scope=raw.get("geo_scope"),
            reliability_score=raw.get("reliability_score"),
            ingest_params={"default_tags": raw.get("tags")},
        )
        session.add(source)
        session.flush()
        return source

    def _to_payload(self, raw: dict[str, Any], source_id: Optional[int]) -> dict[str, Any]:
        published_at = raw.get("published_at")
        return {
            "source_id": source_id,
            "external_id": raw.get("external_id") or f"{source_id}:{raw.get('title')}",
            "secid": raw.get("secid"),
            "published_at": self._parse_datetime(published_at),
            "title": raw.get("title"),
            "body": raw.get("body") or raw.get("summary") or "",
            "tags": raw.get("tags") or [],
            "raw_payload": raw,
        }

    def _upsert_global_event(self, session, raw: dict[str, Any]) -> Optional[int]:
        payload = {
            "event_time": self._parse_datetime(raw.get("event_time") or raw.get("published_at")),
            "event_type": raw.get("event_type") or "news",
            "geo_region": raw.get("geo_region"),
            "severity": int(raw.get("severity", 0)),
            "description": raw.get("body"),
            "affected_sectors": raw.get("affected_sectors"),
            "expected_impact_json": raw.get("impact"),
        }
        stmt = select(GlobalRiskEvent).where(
            GlobalRiskEvent.event_time == payload["event_time"],
            GlobalRiskEvent.event_type == payload["event_type"],
        )
        existing = session.execute(stmt).scalar_one_or_none()
        if existing:
            for key, value in payload.items():
                setattr(existing, key, value)
            return existing.id
        event = GlobalRiskEvent(**payload)
        session.add(event)
        session.flush()
        return event.id

    def _maybe_raise_alert(self, session, raw: dict[str, Any], global_event_id: Optional[int]) -> None:
        severity = raw.get("severity")
        if severity is None:
            return
        level = "low"
        if severity >= 80:
            level = "critical"
        elif severity >= 60:
            level = "high"
        elif severity >= 40:
            level = "medium"
        payload = {
            "secid": raw.get("secid"),
            "sector_code": raw.get("sector_code"),
            "timestamp": self._parse_datetime(raw.get("published_at")),
            "risk_level": level,
            "trigger_reason": raw.get("trigger_reason") or raw.get("event_type") or "news",
            "linked_global_event_id": global_event_id,
            "decay_at": self._parse_datetime(raw.get("decay_at"))
            or self._parse_datetime(raw.get("published_at")) + timedelta(minutes=raw.get("decay_minutes", 120)),
        }
        session.add(RiskAlert(**payload))

    @staticmethod
    def _parse_datetime(value: Optional[str]) -> datetime:
        if not value:
            return datetime.now(timezone.utc)
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest news events and derive risk alerts")
    parser.add_argument("--json-file", type=Path, help="Local JSON payload with events")
    parser.add_argument("--source-url", help="Provide custom JSON API endpoint", required=False)
    parser.add_argument("--moex-lang", default=MOEX_NEWS_LANG, help="Override MOEX news language")
    parser.add_argument("--moex-limit", type=int, default=MOEX_NEWS_LIMIT, help="Limit MOEX rows")
    parser.add_argument("--moex-category", default=MOEX_NEWS_CATEGORY, help="Optional MOEX category filter")
    parser.add_argument("--dry-run", action="store_true", help="Do not write to DB")
    args = parser.parse_args()

    source = HttpSource(url=args.source_url) if args.source_url else None
    moex_client = None
    if not args.json_file and not source:
        moex_client = MoexNewsClient.from_args(
            limit=args.moex_limit,
            lang=args.moex_lang,
            category=args.moex_category,
        )

    stats = NewsIngestor(source=source, file_path=args.json_file, moex_client=moex_client).run(dry_run=args.dry_run)
    logger.info("News ingestion stats: %s", stats)


if __name__ == "__main__":
    main()
