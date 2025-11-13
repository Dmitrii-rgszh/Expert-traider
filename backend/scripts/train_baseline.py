"""Train baseline hybrid models on the labeled news dataset.

This script uses a compact ruBERT encoder to obtain sentence embeddings and fits
simple scikit-learn models for direction, event type, and trigger score.
Run:
    python backend/scripts/train_baseline.py --data data/processed/labeled_news.csv --out models/baseline.joblib
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "cointegrated/rubert-tiny2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline classification/regression models.")
    parser.add_argument("--data", type=Path, required=True, help="Path to processed labeled CSV dataset.")
    parser.add_argument("--out", type=Path, required=True, help="Path to output .joblib file.")
    parser.add_argument("--batch-size", type=int, default=8, help="Embedding batch size.")
    return parser.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset {path} not found.")
    df = pd.read_csv(path)
    required = {"title", "text", "direction", "event_type", "trigger_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    df.fillna({"title": "", "text": ""}, inplace=True)
    return df


def embed_texts(texts: list[str], batch_size: int) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    embeddings: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        tokens = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        )
        with torch.no_grad():
            outputs = model(**tokens)
            hidden_states = outputs.last_hidden_state
            mask = tokens.attention_mask.unsqueeze(-1)
            summed = (hidden_states * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            sentence_embeddings = (summed / counts).cpu().numpy()
        embeddings.append(sentence_embeddings)
    return np.vstack(embeddings)


def train_models(embeddings: np.ndarray, df: pd.DataFrame) -> dict:
    direction_encoder = LabelEncoder()
    direction_target = direction_encoder.fit_transform(df["direction"].astype(str))

    event_encoder = LabelEncoder()
    event_target = event_encoder.fit_transform(df["event_type"].astype(str))

    direction_model = LogisticRegression(max_iter=1000)
    direction_model.fit(embeddings, direction_target)

    event_model = LogisticRegression(max_iter=1000, multi_class="auto")
    event_model.fit(embeddings, event_target)

    score_model = RandomForestRegressor(n_estimators=300, random_state=42)
    score_model.fit(embeddings, df["trigger_score"].astype(float))

    return {
        "direction_model": direction_model,
        "direction_encoder": direction_encoder,
        "event_model": event_model,
        "event_encoder": event_encoder,
        "score_model": score_model,
    }


def main() -> None:
    args = parse_args()
    df = load_dataset(args.data)
    texts = (df["title"].fillna("") + " " + df["text"].fillna("")).tolist()
    embeddings = embed_texts(texts, batch_size=args.batch_size)
    artifacts = train_models(embeddings, df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"artifacts": artifacts, "model_name": MODEL_NAME}, args.out)
    print(f"Saved baseline artifacts to {args.out}")


if __name__ == "__main__":
    main()
