from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional
from urllib.parse import urljoin

import httpx
from sqlalchemy import select

from backend.app.models import SanctionEntity, SanctionLink
from .base import IngestionStats, session_scope, logger

SANCTIONS_API_BASE_URL = os.getenv("SANCTIONS_API_BASE_URL", "https://api.opensanctions.org")
SANCTIONS_API_ENDPOINT = os.getenv("SANCTIONS_API_ENDPOINT", "/datasets/default/en/entities/")
SANCTIONS_API_TOKEN = os.getenv("SANCTIONS_API_TOKEN")
SANCTIONS_API_DATASET = os.getenv("SANCTIONS_API_DATASET", "default")
SANCTIONS_API_PAGE_SIZE = int(os.getenv("SANCTIONS_API_PAGE_SIZE", "200"))
SANCTIONS_API_QUERY = os.getenv("SANCTIONS_API_QUERY")
SANCTIONS_SECID_FIELDS = [
    field.strip()
    for field in os.getenv("SANCTIONS_SECID_FIELDS", "secid,isin,ticker").split(",")
    if field.strip()
]


def load_rows(file_path: Optional[Path], api_client: Optional["SanctionsApiClient"]) -> Iterable[dict[str, Any]]:
    if file_path:
        if file_path.suffix.lower() == ".json":
            payload = json.loads(file_path.read_text(encoding="utf-8"))
            data = payload.get("sanctions", payload)
            for row in data:
                yield row
        else:
            with file_path.open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    yield row
    if api_client:
        yield from api_client.iter_rows()


def parse_datetime(value: Any | None) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not value:
        return datetime.now(timezone.utc)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


class SanctionsApiClient:
    def __init__(
        self,
        base_url: str = SANCTIONS_API_BASE_URL,
        endpoint: str = SANCTIONS_API_ENDPOINT,
        dataset: str = SANCTIONS_API_DATASET,
        token: Optional[str] = SANCTIONS_API_TOKEN,
        page_size: int = SANCTIONS_API_PAGE_SIZE,
        query: Optional[str] = SANCTIONS_API_QUERY,
        secid_fields: Optional[list[str]] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint.lstrip("/")
        self.dataset = dataset
        self.token = token
        self.page_size = page_size
        self.query = query
        self.secid_fields = secid_fields or SANCTIONS_SECID_FIELDS

    def iter_rows(self) -> Iterator[dict[str, Any]]:
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Apikey {self.token}"
        next_url: Optional[str] = urljoin(f"{self.base_url}/", self.endpoint)
        params: Optional[dict[str, Any]] = {"limit": self.page_size}
        if self.query:
            params["q"] = self.query
        while next_url:
            try:
                with httpx.Client(timeout=30.0) as client:
                    response = client.get(next_url, params=params, headers=headers)
                    response.raise_for_status()
            except httpx.HTTPError as exc:
                logger.warning("Sanctions API request failed: %s", exc)
                return
            payload = response.json()
            rows = payload.get("results") or payload.get("entities") or payload.get("data") or []
            for entity in rows:
                yield self._normalize_entity(entity)
            next_url = payload.get("next")
            params = None

    def _normalize_entity(self, entity: dict[str, Any]) -> dict[str, Any]:
        datasets = entity.get("datasets") or []
        if isinstance(datasets, list) and datasets:
            list_code = datasets[0]
        else:
            list_code = entity.get("dataset") or self.dataset
        return {
            "entity_name": entity.get("name") or entity.get("title") or entity.get("id") or "unknown",
            "list_code": list_code or "unknown",
            "listed_at": entity.get("first_seen") or entity.get("created_at") or entity.get("modified_at"),
            "status": entity.get("status") or entity.get("properties", {}).get("status", "active"),
            "source": entity.get("publisher") or entity.get("source") or self.dataset,
            "secids": self._extract_secids(entity),
        }

    def _extract_secids(self, entity: dict[str, Any]) -> list[str]:
        containers: list[dict[str, Any]] = []
        for key in ("properties", "identifiers", "topics", "targets"):
            value = entity.get(key)
            if isinstance(value, dict):
                containers.append(value)
            elif isinstance(value, list):
                for nested in value:
                    if isinstance(nested, dict):
                        containers.append(nested)
        containers.append(entity)
        secids: list[str] = []
        for container in containers:
            for field in self.secid_fields:
                val = container.get(field)
                if isinstance(val, list):
                    secids.extend([str(item).strip() for item in val if item])
                elif val:
                    secids.append(str(val).strip())
        return list(dict.fromkeys([value for value in secids if value]))

    @classmethod
    def from_args(
        cls,
        dataset: Optional[str] = None,
        query: Optional[str] = None,
        page_size: Optional[int] = None,
    ) -> "SanctionsApiClient":
        return cls(
            dataset=dataset or SANCTIONS_API_DATASET,
            query=query or SANCTIONS_API_QUERY,
            page_size=page_size or SANCTIONS_API_PAGE_SIZE,
        )


class SanctionIngestor:
    def __init__(self, file_path: Optional[Path] = None, api_client: Optional[SanctionsApiClient] = None) -> None:
        self.file_path = file_path
        self.api_client = api_client

    def run(self, dry_run: bool = False) -> IngestionStats:
        stats = IngestionStats()
        has_rows = False
        with session_scope() as session:
            for row in load_rows(self.file_path, self.api_client):
                has_rows = True
                stats.processed += 1
                if dry_run:
                    continue
                entity, created = self._upsert_entity(session, row)
                if created:
                    stats.inserted += 1
                self._sync_links(session, entity.id, row.get("secids", []))
        if not has_rows:
            logger.warning("No sanctions data to ingest")
        return stats

    def _upsert_entity(self, session, row: dict[str, str]) -> tuple[SanctionEntity, bool]:
        stmt = select(SanctionEntity).where(
            SanctionEntity.entity_name == row["entity_name"],
            SanctionEntity.list_code == row.get("list_code", "unknown"),
        )
        entity = session.execute(stmt).scalar_one_or_none()
        payload = {
            "entity_name": row["entity_name"],
            "list_code": row.get("list_code", "unknown"),
            "listed_at": parse_datetime(row.get("listed_at")),
            "status": row.get("status", "active"),
            "source": row.get("source"),
        }
        if entity:
            for key, value in payload.items():
                setattr(entity, key, value)
            return entity, False
        entity = SanctionEntity(**payload)
        session.add(entity)
        session.flush()
        return entity, True

    def _sync_links(self, session, entity_id: int, secids_field: str | Iterable[str]) -> None:
        secids = self._normalize_secids(secids_field)
        if not secids:
            return
        for secid in secids:
            stmt = select(SanctionLink).where(SanctionLink.entity_id == entity_id, SanctionLink.secid == secid)
            link = session.execute(stmt).scalar_one_or_none()
            if link:
                continue
            session.add(
                SanctionLink(
                    entity_id=entity_id,
                    secid=secid,
                    confidence=None,
                    notes="auto-ingested",
                )
            )

    @staticmethod
    def _normalize_secids(secids_field: str | Iterable[str]) -> list[str]:
        if isinstance(secids_field, str):
            items = [item.strip() for item in secids_field.split(";")]
        else:
            items = [str(item).strip() for item in secids_field]
        return [item for item in items if item]


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest sanctions lists and link to SECIDs")
    parser.add_argument("file", nargs="?", type=Path, help="CSV/JSON sanctions payload (optional when using API)")
    parser.add_argument("--use-api", action="store_true", help="Pull sanctions from configured API instead of file")
    parser.add_argument("--api-query", help="Override API filter query")
    parser.add_argument("--api-dataset", help="Override API dataset code")
    parser.add_argument("--api-page-size", type=int, help="Override API page size")
    parser.add_argument("--dry-run", action="store_true", help="Do not write to DB")
    args = parser.parse_args()

    api_client = None
    if args.use_api or not args.file:
        api_client = SanctionsApiClient.from_args(
            dataset=args.api_dataset,
            query=args.api_query,
            page_size=args.api_page_size,
        )

    stats = SanctionIngestor(file_path=args.file, api_client=api_client).run(dry_run=args.dry_run)
    logger.info("Sanctions ingestion stats: %s", stats)


if __name__ == "__main__":
    main()
