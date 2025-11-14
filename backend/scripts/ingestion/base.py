from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional

import httpx
import logging
from dotenv import load_dotenv

from backend.app.db.session import SessionLocal

load_dotenv()

logger = logging.getLogger("ingestion")
logging.basicConfig(level=logging.INFO)


@dataclass
class IngestionStats:
    processed: int = 0
    inserted: int = 0
    updated: int = 0


@dataclass
class HttpSource:
    url: str
    params: Optional[Dict[str, Any]] = None
    headers: Optional[Dict[str, str]] = None


@contextmanager
def session_scope() -> Iterator:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def fetch_json(source: HttpSource) -> Any:
    try:
        timeout = float(os.getenv("HTTPX_DEFAULT_TIMEOUT", "15"))
        with httpx.Client(timeout=timeout) as client:
            response = client.get(source.url, params=source.params, headers=source.headers)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as exc:
        logger.warning("HTTP error while fetching %s: %s", source.url, exc)
        return None
