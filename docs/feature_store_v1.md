# Feature Store v1

## Архитектура

- **Fact слой (SQLite/PostgreSQL)** — таблицы `feature_windows`, `feature_numeric`, `feature_categorical`, `train_data_snapshots`.
- **Materialized слой** — parquet/feather снапшоты по пути `data/processed/features/{feature_set}/{timeframe}/YYYYMMDD.parquet`, которые формируются `export_snapshots.py`.
- **Версионность** — идентификатор `feature_set` (например, `tech_v1`, `tech_v2`) фиксирует состав признаков и окно. Описания/таймфреймы лежат в `config/feature_store.yaml`, история — в ML.md.

## Доступные feature_set

### `tech_v1`

| Группа | Признаки |
| --- | --- |
| Тренд | `return_1`, `sma_ratio_5_20`, `ema_ratio_12_26`, `macd_line`, `macd_signal`, `macd_hist` |
| Импульс | `rsi_14`, `volatility_20`, `atr_14` |
| Объём | `volume_zscore_20` |
| Диапазоны | `price_position_20`, `intraday_range_pct` |
| Свечи | `candle_body_pct`, `upper_shadow_pct`, `lower_shadow_pct`, `bullish_engulfing`, `bearish_engulfing` |

### `tech_v2`

Добавляет расширенный контекст:

- Bollinger Bands — `bollinger_band_pct`.
- Stochastic Oscillator — `stoch_k_14`, `stoch_d_3`.
- Увеличенные окна (`default_window_size=120`) и поддержка таймфреймов `1m/5m/15m/1h`.

## Пайплайн

1. **Сбор свечей** — `backend/scripts/ingestion/backfill_candles.py` или Prefect flow `candles-ingest-flow`.
2. **Построение фич** — `backend/scripts/features/build_window.py` читает yaml-конфиг, подставляет `window_size`/`features` и сразу чистит NaN.
3. **Экспорт снапшотов** — `backend/scripts/features/export_snapshots.py` пишет parquet/feather и регистрирует строки в `train_data_snapshots` (secid/timeframe/feature_set/rows_count).
4. **Контроль качества** — `docs/data_quality/*.json` (см. `data_quality_report.py`) + Prefect мониторинги.

## Версионность и раскатка

- Любое изменение списка признаков → новый `feature_set` + запись в `config/feature_store.yaml`/ML.md.
- Prefect деплой (`candles-and-features-flow`) получает параметры `feature_set`, `snapshot_output_dir`, `snapshot_file_format`, что позволяет катить параллельно v1/v2.
- `train_data_snapshots` выступает как audit-log: можно отследить какой снапшот (secid/timeframe/feature_set) использован в обучении (`train-temporal-model-flow` читает тот же config).
