from __future__ import annotations

import os
from functools import lru_cache

from .model import HybridNewsAnalyzer


@lru_cache
def _cached_analyzer(disable_transformers: bool) -> HybridNewsAnalyzer:
    return HybridNewsAnalyzer(enable_transformers=not disable_transformers)


def get_news_analyzer() -> HybridNewsAnalyzer:
    disable_flag = os.getenv("DISABLE_TRANSFORMERS", "false").lower() in {"1", "true", "yes"}
    return _cached_analyzer(disable_flag)
