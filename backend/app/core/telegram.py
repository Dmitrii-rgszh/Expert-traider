from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any, Dict
from urllib.parse import parse_qsl


class TelegramAuthError(Exception):
    """Raised when Telegram init data validation fails."""


def parse_init_data(init_data: str) -> Dict[str, str]:
    """Parse Telegram WebApp init data string into a dict."""
    if not init_data:
        raise TelegramAuthError("Init data is empty")
    try:
        pairs = parse_qsl(init_data, strict_parsing=True)
    except ValueError as exc:
        raise TelegramAuthError("Init data has invalid format") from exc
    data: Dict[str, str] = {}
    for key, value in pairs:
        data[key] = value
    if "hash" not in data:
        raise TelegramAuthError("Init data hash is missing")
    return data


def _build_data_check_string(data: Dict[str, str]) -> str:
    return "\n".join(f"{key}={value}" for key, value in sorted(data.items()))


def validate_init_data(init_data: str, bot_token: str, max_age_seconds: int = 86400) -> Dict[str, Any]:
    """Validate Telegram init data and return the parsed payload."""
    parsed = parse_init_data(init_data)
    incoming_hash = parsed.get("hash", "")

    data_check_payload = {key: value for key, value in parsed.items() if key != "hash"}
    secret_key = hashlib.sha256(f"WebAppData{bot_token}".encode("utf-8")).digest()
    check_string = _build_data_check_string(data_check_payload)
    calculated_hash = hmac.new(secret_key, check_string.encode("utf-8"), hashlib.sha256).hexdigest()

    if not hmac.compare_digest(calculated_hash, incoming_hash):
        raise TelegramAuthError("Init data hash mismatch")

    auth_date_raw = data_check_payload.get("auth_date")
    if auth_date_raw is None:
        raise TelegramAuthError("Init data auth_date missing")
    try:
        auth_date = int(auth_date_raw)
    except ValueError as exc:
        raise TelegramAuthError("Init data auth_date invalid") from exc

    if max_age_seconds > 0 and int(time.time()) - auth_date > max_age_seconds:
        raise TelegramAuthError("Init data expired")

    user_payload = data_check_payload.get("user")
    if user_payload:
        try:
            data_check_payload["user"] = json.loads(user_payload)
        except json.JSONDecodeError as exc:
            raise TelegramAuthError("Init data user payload invalid") from exc
    else:
        data_check_payload["user"] = {}

    data_check_payload["hash"] = incoming_hash
    return data_check_payload
