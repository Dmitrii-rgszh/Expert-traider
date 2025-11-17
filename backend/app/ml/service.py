from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .model import HybridNewsAnalyzer
from .temporal_inference import TemporalInferenceService
from backend.scripts.training.train_temporal_cnn import META_COLUMNS


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


class StrongQ80SuperTraderService:
    def __init__(
        self,
        long_checkpoint: Path,
        short_checkpoint: Path,
        feature_columns: Sequence[str],
        seq_len: int = 32,
        device: str = "cpu",
    ) -> None:
        self.long_service = TemporalInferenceService(
            checkpoint_path=long_checkpoint,
            feature_columns=feature_columns,
            seq_len=seq_len,
            device=device,
        )
        self.short_service = TemporalInferenceService(
            checkpoint_path=short_checkpoint,
            feature_columns=feature_columns,
            seq_len=seq_len,
            device=device,
        )
        self.feature_columns = list(feature_columns)

    def _prepare_features(self, row: dict, seq_len: int) -> np.ndarray:
        df = pd.DataFrame([row])
        # ensure all expected feature columns are present
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0.0
        feature_cols = [c for c in df.columns if c not in META_COLUMNS]
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        base = df[self.feature_columns].to_numpy(dtype=np.float32)
        rows = base.shape[0]
        if rows == seq_len:
            return base
        if rows > seq_len:
            return base[-seq_len:]
        # если строк меньше, чем seq_len (например, один бар), просто повторяем последние значения
        reps = int(np.ceil(seq_len / max(rows, 1)))
        tiled = np.tile(base, (reps, 1))
        return tiled[:seq_len]

    def predict(self, row: dict) -> dict:
        seq_len = self.long_service.seq_len
        features = self._prepare_features(row, seq_len)
        p_long = float(self.long_service.predict_proba(features))
        # short модель должна иметь тот же seq_len; на всякий случай повторно формируем, если отличается
        if self.short_service.seq_len != seq_len:
            short_features = self._prepare_features(row, self.short_service.seq_len)
        else:
            short_features = features
        p_short = float(self.short_service.predict_proba(short_features))
        return {"p_long": p_long, "p_short": p_short}


@lru_cache
def get_strong_q80_super_trader() -> StrongQ80SuperTraderService:
    long_ckpt = os.getenv("STRONG_Q80_LONG_CHECKPOINT")
    short_ckpt = os.getenv("STRONG_Q80_SHORT_CHECKPOINT")
    feature_cols = os.getenv("STRONG_Q80_FEATURE_COLUMNS")
    seq_len = int(os.getenv("STRONG_Q80_SEQ_LEN", "32"))
    device = os.getenv("STRONG_Q80_DEVICE") or "cpu"
    if not long_ckpt or not short_ckpt or not feature_cols:
        raise RuntimeError(
            "STRONG_Q80_LONG_CHECKPOINT, STRONG_Q80_SHORT_CHECKPOINT and STRONG_Q80_FEATURE_COLUMNS must be set."
        )
    columns: Sequence[str] = [c.strip() for c in feature_cols.split(",") if c.strip()]
    if not columns:
        raise RuntimeError("STRONG_Q80_FEATURE_COLUMNS is empty.")
    return StrongQ80SuperTraderService(
        long_checkpoint=Path(long_ckpt),
        short_checkpoint=Path(short_ckpt),
        feature_columns=columns,
        seq_len=seq_len,
        device=device,
    )
