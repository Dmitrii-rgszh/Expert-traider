from functools import lru_cache
from pydantic import BaseModel
import os


class Settings(BaseModel):
    app_name: str = "auto-trader-backend"
    environment: str = os.getenv("ENV", "dev")
    api_prefix: str = "/api"
    cors_origins: list[str] = (
        os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
        .split(",")
        if os.getenv("CORS_ORIGINS") is not None
        else ["http://localhost:5173", "http://127.0.0.1:5173"]
    )

    database_url: str = os.getenv(
        "DATABASE_URL",
        # Fallback to local SQLite to simplify first run; override with Postgres in .env
        "sqlite:///./app.db",
    )
    telegram_bot_token: str | None = os.getenv("TELEGRAM_BOT_TOKEN") or None
    telegram_init_max_age_seconds: int = int(os.getenv("TELEGRAM_INIT_MAX_AGE_SECONDS", "86400"))
    telegram_daily_request_cap: int = int(os.getenv("TELEGRAM_DAILY_REQUEST_CAP", "100"))
    telegram_request_window_seconds: int = int(os.getenv("TELEGRAM_REQUEST_WINDOW_SECONDS", "86400"))


@lru_cache
def get_settings() -> Settings:
    return Settings()
