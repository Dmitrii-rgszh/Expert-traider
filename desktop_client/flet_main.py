from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import flet as ft
import pandas as pd
import requests


@dataclass
class FletReplayConfig:
    dataset_csv: Path = Path("data/training/dataset_intraday_v2_1m_2025q4_enriched_strong_q80.csv")
    api_url: str = "http://localhost:8000/api/trader/strong_q80/decision"
    live_api_url: str = "http://localhost:8000/api/trader/strong_q80/live_decision"
    initial_capital: float = 300_000.0
    sleep_seconds: float = 0.1


def build_payload(row: pd.Series) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for col, val in row.items():
        if pd.isna(val):
            payload[col] = None
        else:
            payload[col] = val
    return payload


def run_replay(
    page: ft.Page,
    cfg: FletReplayConfig,
    status: ft.Text,
    log_view: ft.TextField,
    chart_ai: ft.LineChart,
    chart_index: ft.LineChart,
    settings: dict,
) -> None:
    try:
        df = pd.read_csv(cfg.dataset_csv)
    except Exception as exc:
        page.snack_bar = ft.SnackBar(ft.Text(f"Failed to load dataset: {exc}"), open=True)
        page.update()
        return

    equity_strategy = cfg.initial_capital
    equity_index = cfg.initial_capital
    signals = 0
    wins = 0
    sum_pnl = 0.0

    series_strategy: List[ft.LineChartDataPoint] = []
    series_index: List[ft.LineChartDataPoint] = []

    for idx, row in df.reset_index(drop=True).iterrows():
        payload = build_payload(row)
        try:
            resp = requests.post(cfg.api_url, json=payload, timeout=3)
            resp.raise_for_status()
        except Exception as exc:
            log_view.value += f"API error at idx={idx}: {exc}\n"
            page.update()
            break

        decision = resp.json()
        action = decision.get("action", "HOLD")

        # Benchmark (IMOEX buy & hold)
        try:
            ret_1 = float(row.get("return_1", 0.0))
            ret_vs_imoex = float(row.get("return_vs_imoex", 0.0))
            index_ret = ret_1 - ret_vs_imoex
        except (TypeError, ValueError):
            index_ret = 0.0
        equity_index *= (1.0 + index_ret)

        long_pnl = float(row.get("forward_return_pct", 0.0))
        short_pnl = float(row.get("short_pnl_pct", 0.0))
        if action == "OPEN_LONG":
            pnl = long_pnl
        elif action == "OPEN_SHORT":
            pnl = short_pnl
        else:
            pnl = 0.0

        if action != "HOLD":
            signals += 1
            if pnl > 0:
                wins += 1
            sum_pnl += pnl
            equity_strategy *= (1.0 + pnl)

            secid = row.get("secid")
            t = row.get("signal_time")
            p_long = decision.get("p_long")
            p_short = decision.get("p_short")
            log_view.value += (
                f"{t} {secid}: {action} "
                f"p_long={p_long:.4f} p_short={p_short:.4f} pnl={pnl:.5f}\n"
            )

        # Update chart every few steps
        series_strategy.append(ft.LineChartDataPoint(idx, equity_strategy))
        series_index.append(ft.LineChartDataPoint(idx, equity_index))

        window_points = settings.get("window_points")
        if window_points and window_points > 0:
            strat_points = series_strategy[-window_points:]
            idx_points = series_index[-window_points:]
        else:
            strat_points = series_strategy
            idx_points = series_index

        chart_ai.data_series[0].data_points = strat_points
        chart_index.data_series[0].data_points = idx_points

        if signals:
            winrate = wins / signals
            status.value = (
                f"Сделок: {signals}, винрейт: {winrate:.3f}, "
                f"суммарный P&L: {sum_pnl:.4f}, "
                f"ИИ-портфель: {equity_strategy:,.0f} ₽, "
                f"Индекс IMOEX: {equity_index:,.0f} ₽"
            )
        else:
            status.value = (
                f"Сделок: 0, "
                f"ИИ-портфель: {equity_strategy:,.0f} ₽, "
                f"Индекс IMOEX: {equity_index:,.0f} ₽"
            )

        page.update()
        time.sleep(cfg.sleep_seconds)


def main(page: ft.Page) -> None:
    page.title = "Нейросетевой трейдер Strong_q80 — тест на истории"
    page.horizontal_alignment = "stretch"
    page.scroll = "AUTO"

    status_label = ft.Text("Ожидание. Нажмите «Старт теста».")
    start_button = ft.ElevatedButton("Старт теста")
    log_view = ft.TextField(
        multiline=True,
        min_lines=10,
        max_lines=20,
        read_only=True,
        expand=True,
    )

    equity_chart_ai = ft.LineChart(
        data_series=[ft.LineChartData(data_points=[])],
        min_y=0,
        expand=True,
    )
    equity_chart_index = ft.LineChart(
        data_series=[ft.LineChartData(data_points=[])],
        min_y=0,
        expand=True,
    )

    cfg = FletReplayConfig()
    is_running = {"value": False}
    settings: dict = {"window_points": 0}

    time_window_dropdown = ft.Dropdown(
        label="Окно графика",
        options=[
            ft.dropdown.Option("all", "Всё время теста"),
            ft.dropdown.Option("1d", "Последний день"),
            ft.dropdown.Option("1w", "Последняя неделя"),
            ft.dropdown.Option("1m", "Последний месяц"),
            ft.dropdown.Option("1q", "Последний квартал"),
        ],
        value="all",
    )

    def on_window_change(e: ft.ControlEvent) -> None:
        value = e.control.value
        mapping = {
            "all": 0,
            "1d": 400,
            "1w": 400 * 5,
            "1m": 400 * 21,
            "1q": 400 * 63,
        }
        settings["window_points"] = mapping.get(value, 0)

    time_window_dropdown.on_change = on_window_change

    # режим данных: история / live MOEX
    data_mode_dropdown = ft.Dropdown(
        label="Режим данных",
        options=[
            ft.dropdown.Option("history", "Тест на истории"),
            ft.dropdown.Option("live", "Live MOEX (онлайн)"),
        ],
        value="history",
    )

    # режим сделок: авто / ручной
    trade_mode_dropdown = ft.Dropdown(
        label="Режим сделок",
        options=[
            ft.dropdown.Option("auto", "Авто (вход по сигналу)"),
            ft.dropdown.Option("manual", "Ручной (только рекомендации)"),
        ],
        value="auto",
    )

    lag_field = ft.TextField(label="Лаг исполнения, секунд", value="5", width=160)
    live_secid_field = ft.TextField(label="Тикер для Live (MOEX)", value="SBER", width=160)
    portfolio_text = ft.Text("Портфель пока пуст.", selectable=True)

    def on_start_click(e: ft.ControlEvent) -> None:
        if is_running["value"]:
            return
        is_running["value"] = True
        log_view.value = ""
        equity_chart_ai.data_series[0].data_points = []
        equity_chart_index.data_series[0].data_points = []
        page.update()

        mode = data_mode_dropdown.value
        if mode == "history":
            status_label.value = "Идёт тест на истории..."
            page.update()
            run_replay(page, cfg, status_label, log_view, equity_chart_ai, equity_chart_index, settings)
        else:
            # простой live: несколько шагов опроса MOEX для одного тикера
            import time as _time

            status_label.value = "Live MOEX режим: опрос тикера, сигналы и виртуальный портфель."
            page.update()
            secid = live_secid_field.value.strip() or "SBER"
            lag_sec = 0.0
            try:
                lag_sec = float(lag_field.value)
            except ValueError:
                lag_sec = 5.0

            equity_live = cfg.initial_capital
            positions: list[dict] = []
            max_positions = 5
            horizon_secs = 30 * 60

            for step in range(60):
                try:
                    resp = requests.get(cfg.live_api_url, params={"secid": secid}, timeout=5)
                    resp.raise_for_status()
                    payload = resp.json()
                except Exception as exc:
                    log_view.value += f"Live API error: {exc}\n"
                    page.update()
                    break

                if not payload.get("market_open", False):
                    status_label.value = payload.get("message", "Рынок закрыт")
                    page.update()
                    _time.sleep(lag_sec)
                    continue

                action = payload.get("action", "HOLD")
                p_long = payload.get("p_long")
                p_short = payload.get("p_short")
                feats = payload.get("features", {})
                regime = payload.get("regime", {})
                last_price = float(feats.get("last_price", 0.0) or 0.0)

                auto = trade_mode_dropdown.value == "auto"
                exec_flag = "ИСПОЛНЕНО" if auto and action != "HOLD" else "РЕКОМЕНДАЦИЯ"

                # обновление портфеля (авто-режим)
                if auto and action != "HOLD" and last_price > 0 and len(positions) < max_positions:
                    allocation = 0.05 * equity_live
                    lot_size = float(feats.get("lot_size") or 1.0)
                    if lot_size <= 0:
                        lot_size = 1.0
                    max_lots = int(allocation // (last_price * lot_size))
                    qty = max_lots * lot_size
                    if qty >= lot_size:
                        positions.append(
                            {
                                "secid": secid,
                                "side": "long" if action == "OPEN_LONG" else "short",
                                "qty": qty,
                                "entry_price": last_price,
                                "entry_time": _time.time(),
                            }
                        )

                # закрытие позиций по реальному времени (горизонт ~30 минут)
                still_open: list[dict] = []
                for pos in positions:
                    if _time.time() - pos["entry_time"] >= horizon_secs and last_price > 0:
                        if pos["side"] == "long":
                            pnl = (last_price - pos["entry_price"]) * pos["qty"]
                        else:
                            pnl = (pos["entry_price"] - last_price) * pos["qty"]
                        equity_live += pnl
                    else:
                        still_open.append(pos)
                positions = still_open

                log_view.value += (
                    f"[LIVE {step}] {secid} {feats.get('signal_time')} "
                    f"action={action} ({exec_flag}), "
                    f"p_long={p_long:.4f}, p_short={p_short:.4f}, "
                    f"ret1={feats.get('return_1')}, "
                    f"regime={json.dumps(regime, ensure_ascii=False)}\n"
                )
                status_label.value = (
                    f"Live шаг {step+1}, последний сигнал: {action} ({exec_flag}), "
                    f"капитал ИИ-портфеля: {equity_live:,.0f} ₽"
                )

                # пересчёт состава портфеля и live-графика капитала
                equity_chart_ai.data_series[0].data_points.append(
                    ft.LineChartDataPoint(step, equity_live)
                )
                if positions:
                    total_value = sum(
                        (last_price if pos["secid"] == secid else last_price) * pos["qty"]
                        for pos in positions
                    )
                    lines = ["Текущий ИИ-портфель:"]
                    for pos in positions:
                        mv = (last_price if pos["secid"] == secid else last_price) * pos["qty"]
                        weight = mv / equity_live if equity_live > 0 else 0.0
                        lines.append(
                            f"{pos['secid']} {pos['side']} qty={pos['qty']} "
                            f"средняя={pos['entry_price']:.2f}, вес={weight:.2%}"
                        )
                    portfolio_text.value = "\n".join(lines)
                else:
                    portfolio_text.value = "Портфель пока пуст."

                page.update()
                _time.sleep(lag_sec)

        is_running["value"] = False

    start_button.on_click = on_start_click

    page.add(
        ft.Column(
            [
                status_label,
                start_button,
                ft.Row(
                    [
                        data_mode_dropdown,
                        trade_mode_dropdown,
                        lag_field,
                        live_secid_field,
                    ],
                    spacing=20,
                ),
                ft.Text("Графики капитала от 300 000 ₽ (для режима истории):"),
                time_window_dropdown,
                ft.Row(
                    [
                        ft.Column(
                            [
                                ft.Text("ИИ‑трейдер (портфель нейросети)", weight="bold"),
                                equity_chart_ai,
                            ],
                            expand=1,
                        ),
                        ft.Column(
                            [
                                ft.Text("Индекс МОEX (куплен на 300 000 ₽)", weight="bold"),
                                equity_chart_index,
                            ],
                            expand=1,
                        ),
                    ],
                    expand=True,
                    alignment=ft.MainAxisAlignment.SPACE_AROUND,
                ),
                ft.Text("Лог сделок нейросетевого трейдера:"),
                log_view,
                ft.Text("Состав ИИ-портфеля (live):"),
                portfolio_text,
            ],
            expand=True,
        ),
    )


if __name__ == "__main__":
    ft.app(target=main)
