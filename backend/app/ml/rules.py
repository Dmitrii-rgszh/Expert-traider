from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Set

POSITIVE_KEYWORDS: Dict[str, Sequence[str]] = {
    "bullish": (
        "рост прибыли",
        "увеличение дивидендов",
        "повышение прогноза",
        "расширение производства",
        "запуск новой продукции",
        "партнерство",
        "одобрил сделку",
        "снижение ставок",
        "buyback",
    ),
}

NEGATIVE_KEYWORDS: Dict[str, Sequence[str]] = {
    "bearish": (
        "санкц",
        "штраф",
        "снижение выручки",
        "падение прибыли",
        "сокращение производства",
        "банкрот",
        "дефолт",
        "прекращает поставки",
        "повышение налогов",
        "ограничение экспорта",
        "арест",
    ),
}

EVENT_KEYWORDS: Dict[str, Sequence[str]] = {
    "earnings": ("отчет", "отчёт", "финансовые результаты", "рост прибыли", "прибыль", "выручка"),
    "dividends": ("дивиденд", "выплата на акцию", "buyback"),
    "sanctions": ("санкц", "ограничен", "блокиров"),
    "macro": ("инфляци", "ставк", "ввп", "макро", "экономик", "экспорт", "пошлин", "пошлина"),
    "regulation": ("регулятор", "правительств", "минфин", "указ", "постановлен"),
    "mna": ("сделка", "слиян", "поглощен", "покупает", "приобретает"),
    "default": ("дефолт", "банкрот", "реструктуризац"),
}

HORIZON_KEYWORDS: Dict[str, Sequence[str]] = {
    "intraday": ("сегодня", "в ближайшие часы", "немедленно", "оперативно"),
    "1-3d": ("в течение дня", "в ближайшие дни", "на три дня"),
    "<1m": ("в течение месяца", "в ближайшие недели", "краткосрочн"),
}

ASSET_DICTIONARY: Dict[str, Sequence[str]] = {
    "SBER": ("сбербанк", "sber"),
    "GAZP": ("газпром", "gazprom"),
    "LKOH": ("лукойл", "lukoil"),
    "NVTK": ("новатэк", "novatek"),
    "MGNT": ("магнит", "magnit"),
    "VTBR": ("втб", "vtb"),
    "ALRS": ("алроса", "alrosa"),
    "PLZL": ("полюс", "polyus"),
    "GMKN": ("норникель", "nornickel", "gmkn"),
    "CHMF": ("северсталь", "severstal", "chmf"),
}


@dataclass
class RuleEvaluation:
    score_bonus: float
    direction_votes: Dict[str, int]
    event_votes: Dict[str, int]
    horizon_votes: Dict[str, int]
    assets: Set[str]


def _count_keyword_hits(text: str, keywords: Sequence[str]) -> int:
    hits = 0
    for keyword in keywords:
        if keyword in text:
            hits += 1
    return hits


def evaluate_rules(text: str) -> RuleEvaluation:
    lowered = text.lower()
    direction_votes: Dict[str, int] = {"bullish": 0, "bearish": 0}
    event_votes: Dict[str, int] = {}
    horizon_votes: Dict[str, int] = {}
    assets: Set[str] = set()

    score_bonus = 0.0

    for direction, keywords in POSITIVE_KEYWORDS.items():
        hits = _count_keyword_hits(lowered, keywords)
        if hits:
            direction_votes.setdefault(direction, 0)
            direction_votes[direction] += hits
            score_bonus += 4 * hits

    for direction, keywords in NEGATIVE_KEYWORDS.items():
        hits = _count_keyword_hits(lowered, keywords)
        if hits:
            direction_votes.setdefault(direction, 0)
            direction_votes[direction] += hits
            score_bonus += 5 * hits

    for event_type, keywords in EVENT_KEYWORDS.items():
        hits = _count_keyword_hits(lowered, keywords)
        if hits:
            event_votes[event_type] = event_votes.get(event_type, 0) + hits
            score_bonus += 2 * hits

    for horizon, keywords in HORIZON_KEYWORDS.items():
        hits = _count_keyword_hits(lowered, keywords)
        if hits:
            horizon_votes[horizon] = horizon_votes.get(horizon, 0) + hits

    for ticker, aliases in ASSET_DICTIONARY.items():
        for alias in aliases:
            if alias in lowered:
                assets.add(ticker)
                break

    score_bonus = min(score_bonus, 30.0)

    return RuleEvaluation(
        score_bonus=score_bonus,
        direction_votes=direction_votes,
        event_votes=event_votes,
        horizon_votes=horizon_votes,
        assets=assets,
    )
