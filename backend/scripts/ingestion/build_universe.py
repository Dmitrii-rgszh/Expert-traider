from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import requests

MOEX_HOST = "https://iss.moex.com"

# Board mappings for key instrument classes on MOEX cash market.
BOARD_MAP = {
    "equities": ("stock", "shares", "TQBR"),
    "bonds": ("stock", "bonds", "TQOB"),
    "etf": ("stock", "shares", "TQTF"),
}


def fetch_board_tickers(engine: str, market: str, board: str) -> List[str]:
    url = f"{MOEX_HOST}/iss/engines/{engine}/markets/{market}/boards/{board}/securities.json"
    params = {
        "iss.meta": "off",
        "iss.only": "securities",
        "securities.columns": "SECID",
    }
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    payload = resp.json()
    data = payload.get("securities", {}).get("data", []) or []
    tickers = [row[0] for row in data if row and row[0]]
    return sorted(set(tickers))


def build_universe() -> Dict[str, List[str]]:
    universe: Dict[str, List[str]] = {}
    for key, (engine, market, board) in BOARD_MAP.items():
        tickers = fetch_board_tickers(engine, market, board)
        universe[key] = tickers
    return universe


def main() -> None:
    universe = build_universe()
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "universe": universe,
        "metadata": {
            "boards": BOARD_MAP,
            "source": "iss.moex.com",
        },
    }
    out_path = Path("config/universe_full.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved universe with { {k: len(v) for k, v in universe.items()} } to {out_path}")


if __name__ == "__main__":
    main()
