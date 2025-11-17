import sys
from contextlib import asynccontextmanager
from pathlib import Path

try:
    import python_multipart  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    python_multipart = None  # type: ignore
else:  # pragma: no cover
    sys.modules.setdefault("multipart", python_multipart)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from .core.config import get_settings
from .db.session import Base, engine
from .routers.analyze import router as analyze_router
from .routers.trader import router as trader_router

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


app.include_router(analyze_router, prefix=f"{settings.api_prefix}")
app.include_router(trader_router, prefix=f"{settings.api_prefix}")
