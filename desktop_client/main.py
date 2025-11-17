from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import requests
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


@dataclass
class ReplayConfig:
    dataset_csv: Path
    api_url: str = "http://localhost:8000/api/trader/strong_q80/decision"
    initial_capital: float = 300_000.0


class ReplaySuperTraderWindow(QMainWindow):
    def __init__(self, config: ReplayConfig) -> None:
        super().__init__()
        self.config = config
        self.setWindowTitle("Strong_q80 Super-Trader (Replay)")

        central = QWidget(self)
        layout = QVBoxLayout(central)

        self.status_label = QLabel("Idle", self)
        self.start_button = QPushButton("Start replay", self)
        self.log_view = QTextEdit(self)
        self.log_view.setReadOnly(True)

        layout.addWidget(self.status_label)
        layout.addWidget(self.start_button)
        layout.addWidget(self.log_view)
        self.setCentralWidget(central)

        self.start_button.clicked.connect(self.start_replay)

        self._df: pd.DataFrame | None = None
        self._index: int = 0
        self._signals: int = 0
        self._wins: int = 0
        self._sum_pnl: float = 0.0
        self._equity_strategy: float = config.initial_capital
        self._equity_index: float = config.initial_capital

        self._timer = QTimer(self)
        self._timer.setInterval(200)
        self._timer.timeout.connect(self._step)

    def start_replay(self) -> None:
        try:
            self._df = pd.read_csv(self.config.dataset_csv)
        except Exception as exc:
            self.status_label.setText(f"Failed to load dataset: {exc}")
            return
        self._index = 0
        self._signals = 0
        self._wins = 0
        self._sum_pnl = 0.0
        self._equity_strategy = self.config.initial_capital
        self._equity_index = self.config.initial_capital
        self.log_view.clear()
        self.status_label.setText("Replaying...")
        self._timer.start()

    def _build_payload(self, row: pd.Series) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        for col, val in row.items():
            if pd.isna(val):
                payload[col] = None
            else:
                payload[col] = val
        return payload

    def _step(self) -> None:
        if self._df is None:
            self._timer.stop()
            return
        if self._index >= len(self._df):
            self._timer.stop()
            if self._signals:
                winrate = self._wins / self._signals
                self.status_label.setText(
                    f"Finished: signals={self._signals}, winrate={winrate:.3f}, "
                    f"equity_strategy={self._equity_strategy:.2f}, equity_imoex={self._equity_index:.2f}",
                )
            else:
                self.status_label.setText("Finished: no signals")
            return

        row = self._df.iloc[self._index]
        self._index += 1

        # Update benchmark index equity (buy & hold IMOEX on 300k)
        try:
            ret_1 = float(row.get("return_1", 0.0))
            ret_vs_imoex = float(row.get("return_vs_imoex", 0.0))
            index_ret = ret_1 - ret_vs_imoex
        except (TypeError, ValueError):
            index_ret = 0.0
        self._equity_index *= (1.0 + index_ret)

        payload = self._build_payload(row)
        try:
            resp = requests.post(self.config.api_url, json=payload, timeout=3)
            resp.raise_for_status()
        except Exception as exc:
            self.log_view.append(f"API error at idx={self._index}: {exc}")
            return

        decision = resp.json()
        action = decision.get("action", "HOLD")
        if action == "HOLD":
            return

        long_pnl = float(row.get("forward_return_pct", 0.0))
        short_pnl = float(row.get("short_pnl_pct", 0.0))
        if action == "OPEN_LONG":
            pnl = long_pnl
        elif action == "OPEN_SHORT":
            pnl = short_pnl
        else:
            pnl = 0.0

        self._signals += 1
        if pnl > 0:
            self._wins += 1
        self._sum_pnl += pnl

        # Treat pnl as fractional return on full capital for demo
        self._equity_strategy *= (1.0 + pnl)

        secid = row.get("secid")
        t = row.get("signal_time")
        p_long = decision.get("p_long")
        p_short = decision.get("p_short")
        self.log_view.append(
            f"{t} {secid}: {action} p_long={p_long:.4f} p_short={p_short:.4f} pnl={pnl:.5f}",
        )
        self.status_label.setText(
            f"Signals={self._signals}, winrate={(self._wins / self._signals):.3f}, "
            f"sum_pnl={self._sum_pnl:.4f}, "
            f"equity_strategy={self._equity_strategy:.2f}, equity_imoex={self._equity_index:.2f}",
        )


def main() -> None:
    dataset = Path("data/training/dataset_intraday_v2_1m_2025q4_enriched_strong_q80.csv")
    config = ReplayConfig(dataset_csv=dataset)

    app = QApplication(sys.argv)
    window = ReplaySuperTraderWindow(config)
    window.resize(900, 600)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
