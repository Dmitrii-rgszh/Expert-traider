from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

from .rules import evaluate_rules

try:  # pragma: no cover - optional dependency
    from transformers import AutoModel, AutoTokenizer, pipeline  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pipeline = None  # type: ignore
    AutoModel = None  # type: ignore
    AutoTokenizer = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore

LOGGER = logging.getLogger(__name__)

DEFAULT_SENTIMENT_MODEL = "blanchefort/rubert-base-cased-sentiment"
EMBED_MODEL_NAME = "cointegrated/rubert-tiny2"
SENTIMENT_TO_DIRECTION: Dict[str, str] = {
    "positive": "bullish",
    "negative": "bearish",
    "neutral": "neutral",
    "very positive": "bullish",
    "very negative": "bearish",
}


@dataclass
class AnalyzerOutput:
    trigger_score: float
    direction: str
    event_type: Optional[str]
    horizon: Optional[str]
    assets: List[str]
    summary: str


class HybridNewsAnalyzer:
    """Baseline analyzer that blends a lightweight BERT sentiment model with simple rules."""

    def __init__(self, enable_transformers: bool | None = None) -> None:
        self._sentiment_pipeline = None
        self.enable_transformers = enable_transformers if enable_transformers is not None else True
        self._baseline_bundle: Optional[BaselineBundle] = None
        self._sentiment_model_id = os.getenv("SENTIMENT_MODEL_ID", DEFAULT_SENTIMENT_MODEL)
        self._hf_token = os.getenv("HUGGINGFACE_TOKEN") or None

        if not self.enable_transformers:
            LOGGER.info("Transformers support disabled via flag; using rule-based analysis only.")
            return

        if pipeline is None or AutoTokenizer is None or AutoModel is None or torch is None:
            LOGGER.warning("transformers package is not installed; falling back to rule-based heuristics.")
            return

        pipeline_kwargs: Dict[str, Any] = {}
        if self._hf_token:
            pipeline_kwargs["token"] = self._hf_token

        try:
            self._sentiment_pipeline = pipeline(
                task="text-classification",
                model=self._sentiment_model_id,
                tokenizer=self._sentiment_model_id,
                top_k=None,
                return_all_scores=True,
                **pipeline_kwargs,
            )
            LOGGER.info("Loaded sentiment pipeline %s", self._sentiment_model_id)
        except Exception as exc:  # pragma: no cover - depends on external model download
            LOGGER.warning("Failed to initialise sentiment model (%s); rule-based mode only.", exc)
            self._sentiment_pipeline = None

        baseline_path = os.getenv("BASELINE_MODEL_PATH")
        if baseline_path:
            try:
                self._baseline_bundle = self._load_baseline(Path(baseline_path))
                LOGGER.info("Loaded baseline classifiers from %s", baseline_path)
            except Exception as exc:  # pragma: no cover - depends on artifact presence
                LOGGER.warning("Failed to load baseline artifact (%s); continuing without it.", exc)

    def analyze(self, text: str, title: Optional[str] = None) -> AnalyzerOutput:
        combined_text = f"{title}. {text}" if title else text
        rules = evaluate_rules(combined_text)

        direction = self._direction_from_rules(rules.direction_votes)
        event_type = self._event_from_rules(rules.event_votes)
        horizon = self._horizon_from_rules(rules.horizon_votes)

        sentiment_direction = None
        sentiment_strength = 0.0
        if self._sentiment_pipeline is not None:
            try:
                sentiment_direction, sentiment_strength = self._direction_from_sentiment(combined_text)
            except Exception as exc:  # pragma: no cover - depends on external model
                LOGGER.warning("Sentiment model failed (%s); continuing with heuristic results.", exc)
                sentiment_direction = None
                sentiment_strength = 0.0

        direction = self._blend_directions(direction, sentiment_direction, rules.direction_votes, sentiment_strength)
        trigger_score = self._compute_score(direction, rules.score_bonus, sentiment_direction, sentiment_strength)

        baseline_direction: Optional[str] = None
        baseline_event: Optional[str] = None
        baseline_score: Optional[float] = None
        if self._baseline_bundle is not None:
            baseline_direction, baseline_event, baseline_score = self._predict_baseline(combined_text)
            direction = self._merge_with_baseline_direction(
                direction,
                baseline_direction,
                sentiment_direction,
                sentiment_strength,
                rules.direction_votes,
            )
            if baseline_event:
                event_type = baseline_event
            if baseline_score is not None:
                trigger_score = self._blend_scores(trigger_score, baseline_score)

        if horizon is None:
            horizon = self._estimate_horizon(trigger_score)

        if event_type is None:
            event_type = "other"

        assets = sorted(rules.assets)
        summary = self._format_summary(direction, trigger_score, event_type, horizon, assets)

        return AnalyzerOutput(
            trigger_score=trigger_score,
            direction=direction,
            event_type=event_type,
            horizon=horizon,
            assets=assets,
            summary=summary,
        )

    def _predict_baseline(self, text: str) -> Tuple[Optional[str], Optional[str], Optional[float]]:
        bundle = self._baseline_bundle
        if bundle is None:
            return None, None, None

        embedding = self._embed_text(text)
        if embedding is None:
            return None, None, None

        direction = bundle.direction_encoder.inverse_transform(
            bundle.direction_model.predict(embedding)
        )[0]
        event = bundle.event_encoder.inverse_transform(bundle.event_model.predict(embedding))[0]
        score = float(bundle.score_model.predict(embedding)[0])
        return direction, event, score

    def _embed_text(self, text: str) -> Optional[np.ndarray]:
        bundle = self._baseline_bundle
        if bundle is None:
            return None
        tokenizer = bundle.tokenizer
        model = bundle.model
        if tokenizer is None or model is None or torch is None:
            return None

        tokens = tokenizer(
            text,
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
        return sentence_embeddings

    def _load_baseline(self, path: Path) -> "BaselineBundle":
        if not path.exists():
            raise FileNotFoundError(path)
        if AutoTokenizer is None or AutoModel is None or torch is None:
            raise RuntimeError("transformers are required for baseline embeddings")

        payload = joblib.load(path)
        artifacts = payload.get("artifacts", payload)
        model_name = payload.get("model_name", EMBED_MODEL_NAME)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()

        expected = {"direction_model", "direction_encoder", "event_model", "event_encoder", "score_model"}
        missing = expected - artifacts.keys()
        if missing:
            raise ValueError(f"baseline artifact missing keys: {missing}")

        return BaselineBundle(
            direction_model=artifacts["direction_model"],
            direction_encoder=artifacts["direction_encoder"],
            event_model=artifacts["event_model"],
            event_encoder=artifacts["event_encoder"],
            score_model=artifacts["score_model"],
            tokenizer=tokenizer,
            model=model,
        )

    @staticmethod
    def _direction_from_rules(votes: Dict[str, int]) -> str:
        if not votes:
            return "neutral"
        best = max(votes.items(), key=lambda item: item[1])
        if best[1] == 0:
            return "neutral"
        return best[0]

    def _direction_from_sentiment(self, text: str) -> tuple[Optional[str], float]:
        if self._sentiment_pipeline is None:
            return None, 0.0

        scored = self._sentiment_pipeline(text[:512])  # type: ignore[operator]
        if not scored:
            return None, 0.0

        # pipeline returns a list of list of dicts when top_k=None
        candidates = scored[0] if isinstance(scored, list) and isinstance(scored[0], list) else scored
        best_label = None
        best_score = 0.0
        for item in candidates:
            label = item["label"].lower()
            mapped = SENTIMENT_TO_DIRECTION.get(label)
            score = float(item["score"])
            if mapped is None:
                continue
            if score > best_score:
                best_label = mapped
                best_score = score
        return best_label, best_score

    @staticmethod
    def _blend_directions(
        rule_direction: str,
        sentiment_direction: Optional[str],
        rule_votes: Dict[str, int],
        sentiment_strength: float,
    ) -> str:
        if sentiment_direction is None:
            return rule_direction

        if rule_direction == "neutral" and sentiment_direction != "neutral":
            if sentiment_strength > 0.35:
                return sentiment_direction
            return rule_direction

        if sentiment_direction == "neutral":
            if rule_direction != "neutral" and sentiment_strength < 0.45:
                return rule_direction
            return "neutral"

        if rule_direction != sentiment_direction:
            neutral_vote = rule_votes.get("neutral", 0)
            dominant_votes = rule_votes.get(rule_direction, 0)
            if sentiment_strength > 0.6 or dominant_votes == 0 or dominant_votes <= neutral_vote:
                return sentiment_direction

        return rule_direction

    @staticmethod
    def _merge_with_baseline_direction(
        current_direction: str,
        baseline_direction: Optional[str],
        sentiment_direction: Optional[str],
        sentiment_strength: float,
        rule_votes: Dict[str, int],
    ) -> str:
        if baseline_direction is None:
            return current_direction

        if current_direction == "neutral" and baseline_direction != "neutral":
            return baseline_direction

        if baseline_direction != current_direction:
            dominant_votes = rule_votes.get(current_direction, 0)
            if dominant_votes == 0 or sentiment_direction == baseline_direction:
                return baseline_direction
            if sentiment_direction == current_direction and sentiment_strength >= 0.6:
                return current_direction
            return baseline_direction

        return current_direction

    @staticmethod
    def _compute_score(
        direction: str,
        rule_bonus: float,
        sentiment_direction: Optional[str],
        sentiment_strength: float,
    ) -> float:
        base = 32.0
        if direction != "neutral":
            base = 45.0
        if sentiment_direction == direction and sentiment_direction != "neutral":
            base += sentiment_strength * 25
        if sentiment_direction == "neutral" and direction != "neutral":
            base -= (1 - sentiment_strength) * 10
        score = base + rule_bonus
        if direction == "neutral":
            score = min(score, 55.0)
        return round(max(5.0, min(score, 95.0)), 1)

    @staticmethod
    def _blend_scores(rule_score: float, baseline_score: float) -> float:
        blended = 0.4 * rule_score + 0.6 * baseline_score
        return round(max(5.0, min(blended, 95.0)), 1)

    @staticmethod
    def _event_from_rules(votes: Dict[str, int]) -> Optional[str]:
        if not votes:
            return None
        best = max(votes.items(), key=lambda item: item[1])
        return best[0]

    @staticmethod
    def _horizon_from_rules(votes: Dict[str, int]) -> Optional[str]:
        if not votes:
            return None
        best = max(votes.items(), key=lambda item: item[1])
        return best[0]

    @staticmethod
    def _estimate_horizon(score: float) -> str:
        if score >= 75:
            return "1-3d"
        if score >= 60:
            return "<1m"
        return "intraday"

    @staticmethod
    def _format_summary(direction: str, score: float, event_type: str, horizon: str | None, assets: List[str]) -> str:
        direction_map = {
            "bullish": "скорее позитивна",
            "bearish": "скорее негативна",
            "neutral": "скорее нейтральна",
        }
        parts = [
            f"Новость относится к категории {event_type}.",
            f"Оценка силы триггера {score} из 100 и она {direction_map.get(direction, direction)} для рынка.",
        ]
        if assets:
            parts.append(f"Затронутые активы: {', '.join(assets)}.")
        if horizon:
            parts.append(f"Ожидаемый горизонт реакции: {horizon}.")
        return " ".join(parts)


@dataclass
class BaselineBundle:
    direction_model: Any
    direction_encoder: Any
    event_model: Any
    event_encoder: Any
    score_model: Any
    tokenizer: Any
    model: Any
