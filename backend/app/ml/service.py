from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from .model import HybridNewsAnalyzer
from .temporal_inference import TemporalInferenceService


@lru_cache
def _cached_analyzer(disable_transformers: bool) -> HybridNewsAnalyzer:
    return HybridNewsAnalyzer(enable_transformers=not disable_transformers)


def get_news_analyzer() -> HybridNewsAnalyzer:
    disable_flag = os.getenv("DISABLE_TRANSFORMERS", "false").lower() in {"1", "true", "yes"}
    return _cached_analyzer(disable_flag)


@lru_cache
def get_temporal_model_service() -> TemporalInferenceService:
    checkpoint = os.getenv("TEMPORAL_MODEL_CHECKPOINT")
    feature_cols = os.getenv("TEMPORAL_FEATURE_COLUMNS")
    seq_len = int(os.getenv("TEMPORAL_SEQ_LEN", "32"))
    if not checkpoint or not feature_cols:
        raise RuntimeError(
            "TEMPORAL_MODEL_CHECKPOINT and TEMPORAL_FEATURE_COLUMNS env vars must be set for temporal inference service."
        )
    columns: Sequence[str] = [col.strip() for col in feature_cols.split(",") if col.strip()]
    if not columns:
        raise RuntimeError("TEMPORAL_FEATURE_COLUMNS is empty.")
    device = os.getenv("TEMPORAL_DEVICE") or "cpu"
    return TemporalInferenceService(
        checkpoint_path=Path(checkpoint),
        feature_columns=columns,
        seq_len=seq_len,
        device=device,
    )
