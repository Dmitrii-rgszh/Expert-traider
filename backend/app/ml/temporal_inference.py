from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn


class TemporalCNN(nn.Module):
    def __init__(self, feature_dim: int, seq_len: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.conv = nn.Sequential(
            nn.Conv1d(feature_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected [batch, seq_len, features], got {tuple(x.shape)}")
        x = x.permute(0, 2, 1)
        feats = self.conv(x)
        return self.head(feats)


@dataclass
class TemporalInferenceService:
    checkpoint_path: Path
    feature_columns: Sequence[str]
    seq_len: int
    device: str = "cpu"

    def __post_init__(self) -> None:
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model().to(self.device)
        self.model.eval()

    def _load_model(self) -> TemporalCNN:
        state = torch.load(self.checkpoint_path, map_location=self.device)
        hparams = state.get("hyper_parameters", {})
        feature_dim = int(hparams.get("feature_dim", len(self.feature_columns)))
        seq_len = int(hparams.get("seq_len", self.seq_len))
        dropout = float(hparams.get("dropout", 0.2))
        model = TemporalCNN(feature_dim=feature_dim, seq_len=seq_len, dropout=dropout)
        state_dict = dict(state.get("state_dict", {}))
        # lightning checkpoints may contain extra buffers like `pos_weight` that
        # are only used during training; drop them to avoid strict loading errors
        state_dict.pop("pos_weight", None)
        model.load_state_dict(state_dict, strict=False)
        self.seq_len = seq_len
        return model

    def predict_proba(self, sequence: np.ndarray) -> float:
        if sequence.shape != (self.seq_len, len(self.feature_columns)):
            raise ValueError(f"Expected input shape {(self.seq_len, len(self.feature_columns))}, got {sequence.shape}")
        tensor = torch.from_numpy(sequence.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor).squeeze(-1)
            prob = torch.sigmoid(logits).item()
        return float(prob)

    def predict_from_dataframe(self, df: "pd.DataFrame") -> float:
        import pandas as pd  # local import to avoid hard dependency during service bootstrap

        if df.empty:
            raise ValueError("Input dataframe is empty.")
        df = df.sort_values("signal_time").tail(self.seq_len)
        if len(df) < self.seq_len:
            raise ValueError(f"Need at least {self.seq_len} rows to form a sequence.")
        sequence = df[self.feature_columns].to_numpy(dtype=np.float32)
        return self.predict_proba(sequence)
