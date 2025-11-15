from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class FeatureSetSpec:
    name: str
    timeframe: str
    window_size: int
    features: list[str]
    version: int
    description: str
    parquet_root: Path
    partitioning: str


class FeatureStoreConfig:
    def __init__(self, config_path: Path) -> None:
        self.path = config_path
        if not self.path.exists():
            raise FileNotFoundError(f"Feature store config {self.path} not found")
        with self.path.open("r", encoding="utf-8") as fh:
            self.payload = yaml.safe_load(fh) or {}

    def get_feature_set(self, name: str, timeframe: str) -> FeatureSetSpec:
        feature_sets = self.payload.get("feature_sets", {})
        if name not in feature_sets:
            raise KeyError(f"Feature set '{name}' not defined in {self.path}")
        entry: dict[str, Any] = feature_sets[name]
        timeframes = entry.get("timeframes") or {}
        tf_entry = timeframes.get(str(timeframe))
        if not tf_entry:
            raise KeyError(f"Timeframe '{timeframe}' is not configured for feature set '{name}'")
        window_size = tf_entry.get("window_size") or entry.get("default_window_size")
        if not window_size:
            raise ValueError(f"No window_size configured for {name}/{timeframe}")
        features = entry.get("features") or []
        version = int(entry.get("version", 1))
        description = entry.get("description", "")
        parquet_root = Path(self.payload.get("parquet_root", "data/processed/features"))
        partitioning = self.payload.get("snapshot_partitioning", "daily")
        return FeatureSetSpec(
            name=name,
            timeframe=str(timeframe),
            window_size=int(window_size),
            features=list(features),
            version=version,
            description=description,
            parquet_root=parquet_root,
            partitioning=partitioning,
        )

    def dump(self) -> str:
        return json.dumps(self.payload, indent=2, ensure_ascii=False)
