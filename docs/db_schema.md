# AI Trader Data Schema (PostgreSQL)

This document defines the canonical storage model for the AI-трейдер platform. The goal is to support data ingestion, feature engineering, model training/inference, user analytics, and the Telegram Miniapp API from a single, well-governed database (PostgreSQL, extensions: `timescaledb` optional, `pgvector` optional for embeddings/logs).

> **Notation**
>
> - `PK` – primary key, `FK` – foreign key. Timestamp columns are stored as `TIMESTAMPTZ` in UTC. Monetary values are decimal with explicit currency codes. Partitioning is recommended for high-volume tables (candles, features, signals). All IDs use monotonically increasing `BIGSERIAL` unless stated otherwise.

## 0. Reference & Configuration Tables

| Table | Purpose | Key Columns |
| --- | --- | --- |
| `securities` | Master list of tickers/boards used across modules. | `id (PK)`, `secid (text, unique)`, `board`, `instrument_type`, `sector_code`, `lot_size`, `currency`, `is_active`, metadata JSON. |
| `boards` | Optional lookup for MOEX boards (TQBR etc.). | `code (PK)`, `name`, `description`. |
| `timeframes` | Supported bar sizes for ETL/feature factory. | `code (PK)` (`1m`, `5m`, ...), `seconds`. |
| `model_configs` | Versioned hyperparameters for global/strategy/adapter models. | `id`, `model_name`, `version`, `config_json`, `created_at`. |

## 1. Market Data Layer

| Table | Purpose | Key Columns / Notes |
| --- | --- | --- |
| `candles` | OHLCV bars per security/timeframe from MOEX ISS. | `id`, `secid (FK securities)`, `board`, `timeframe (FK)`, `datetime`, `open, high, low, close, volume`. Unique index on `(secid, timeframe, datetime)`. Partition by timeframe or month. |
| `index_candles` | OHLCV for indices (IMOEX, RTSI, sector indices). | same structure as `candles` with `index_code`. |
| `sector_snapshots` | Aggregated sector metrics for a given window. | `id`, `sector_code`, `datetime`, `avg_return`, `avg_volatility`, `breadth`, `notes`. |
| `liquidity_metrics` | Optional per-security liquidity snapshot (spread, depth). | `id`, `secid`, `datetime`, metrics JSON. |
| `order_book` | Level 2 order book snapshots. | `id`, `secid`, `datetime`, `side` (`bid/ask`), `level`, `price`, `size`, `provider`, `ingested_at`. |
| `trades` | Tick-by-tick trades. | `id`, `secid`, `datetime`, `price`, `size`, `side` (`buy/sell/na`), `trade_id_ext`, `provider`. |
| `price_limits` | Daily price limits per instrument. | `id`, `secid`, `date`, `upper_limit`, `lower_limit`, `reason`, `source`. |
| `trading_status` | Trading state and session information. | `id`, `secid`, `timestamp`, `status` (`open/auction/halt/closing`), `reason`, `session` (`T+`, `after-hours`). |
| `exchange_calendar` | Trading calendar and sessions. | `date (PK)`, `is_trading_day`, `session_open`, `session_close`, `notes`. |
| `schedule_changes` | Ad-hoc schedule changes. | `id`, `effective_date`, `change_type`, `details_json`, `source_ref`. |

### Historical coverage requirement

- ETL must backfill and keep at least **8 лет** истории по всем поддерживаемым тикерам и таймфреймам. 
- Используем `train_data_snapshots (id, secid, timeframe, snapshot_start, snapshot_end, rows_count, created_at)` для фиксации выгруженных обучающих выборок и контроля полноты.
- Старые партиции `candles`/`index_candles` архивируются, но остаются доступными для переобучений; ETL автоматически догружает пропущенные окна.

### Macro, FX, Commodities, Rates

| Table | Purpose | Key Columns / Notes |
| --- | --- | --- |
| `fx_rates` | FX pairs relevant for MOEX. | `id`, `pair` (`USDRUB`, `EURRUB`, `CNYRUB`, ...), `datetime`, `rate`, `provider`. |
| `commodity_prices` | Key commodities (Brent, Urals, gold, gas). | `id`, `symbol`, `datetime`, `price`, `currency`, `provider`. |
| `macro_series` | Macro time series. | `id`, `series_code` (`CPI_YOY`, `PMI`, `GDP_QOQ`, ...), `period_start`, `period_end`, `value`, `revision`, `source`. |
| `policy_rates` | Policy and money market rates. | `id`, `rate_code` (`CBR_KEY`, `RUONIA`, `REPO_1D`, ...), `date`, `value`, `announced_at`, `effective_from`. |
| `ofz_yields` | OFZ yields and risk metrics. | `id`, `isin`, `maturity_date`, `date`, `ytm`, `dirty_price`, `duration`, `convexity`, `coupon`, `next_coupon_date`. |
| `ofz_auctions` | OFZ primary auctions. | `id`, `auction_date`, `isin`, `offered`, `placed`, `yield_min`, `yield_avg`, `yield_max`, `bid_cover`, `notes`. |

## 2. Fundamental Layer

| Table | Purpose | Key Columns |
| --- | --- | --- |
| `fundamental_reports` | Raw quarterly/annual indicators. | `id`, `secid`, `report_date`, `period_type`, `currency`, `revenue`, `ebitda`, `net_income`, `assets`, `liabilities`, `dividends`, `pe`, `pb`, `ev_ebitda`, etc. |
| `fundamental_scores` | Aggregated compact factors per ML spec. | `id`, `secid`, `as_of_date`, `fund_value_score`, `fund_growth_score`, `fund_quality_score`, `fund_dividend_score`, `fund_risk_score`, `details_json`. |

## 3. News & Trigger Layer

| Table | Purpose | Key Columns / Notes |
| --- | --- | --- |
| `news_events` | Unified feed (could ingest from existing `news` table). | `id`, `source`, `external_id`, `secid (nullable, FK)`, `published_at`, `title`, `body`, `tags`, `raw_payload`. |
| `news_trigger_scores` | Output of news-trigger module per security & time. | `id`, `secid`, `event_id (FK news_events)`, `timestamp`, `news_trigger_score`, `news_direction`, `news_event_type`, `explanation`. Unique `(secid, timestamp, news_event_type)`. |
| `news_embeddings` (optional) | Vector store for Semantic search. | `id`, `event_id`, `provider`, `embedding vector`. Requires `pgvector`. |
| `news_sources` | Registry of feeds (МОEX disclosures, Reuters, Bloomberg, Telegram-каналы). | `id`, `code`, `provider`, `geo_scope (ru/global)`, `reliability_score`, `latency_profile`, `ingest_params`. |
| `global_risk_events` | Каталог внешних событий, влияющих на российский рынок. | `id`, `event_time`, `event_type` (`sanctions`, `war`, `epidemic`, ...), `geo_region`, `severity (0-100)`, `description`, `source_ref (FK news_events/news_sources)`, `affected_sectors`, `expected_impact_json`. |
| `risk_alerts` | Быстрые уведомления, сформированные после обработки глобальных новостей. | `id`, `secid (nullable)`, `sector_code (nullable)`, `timestamp`, `risk_level`, `trigger_reason`, `linked_global_event_id`, `decay_at`. |
| `sanction_entities` | Реестр санкционных субъектов. | `id`, `entity_name`, `list_code`, `listed_at`, `status`, `source`. |
| `sanction_links` | Связь санкционных субъектов с бумагами. | `id`, `entity_id (FK)`, `secid`, `confidence`, `notes`. |

> **Fast reaction requirement**: news ingestion пайплайн обрабатывает локальные и мировые новости в режиме близком к real-time, записывая их в `news_events`/`global_risk_events`, после чего обновляет `news_trigger_scores` и `risk_alerts` для ускоренной реакции моделей.

## 4. Feature Store

Two options: a wide table per timeframe or key-value store. Proposed hybrid for clarity.

| Table | Purpose | Key Columns / Notes |
| --- | --- | --- |
| `feature_windows` | Metadata about generated feature batches. | `id`, `secid`, `timeframe`, `window_start`, `window_end`, `feature_set` (e.g., `v1.tech+fund+news`), `generated_at`, `checksum`. |
| `features_numeric` | Tall table storing numeric feature values. | `feature_window_id (FK)`, `feature_name`, `value_numeric`. Composite PK `(feature_window_id, feature_name)`. |
| `features_categorical` | Same for categorical/encoded values. | `feature_window_id`, `feature_name`, `value_text`. |
| `features_vector` | Optional for embeddings/time-based vectors. | `feature_window_id`, `feature_name`, `vector`. |

## 5. Labeling & Backtesting

| Table | Purpose |
| --- | --- |
| `trade_labels` | Stores computed success/failure labels per security/timeframe/horizon. Columns: `id`, `secid`, `timeframe`, `signal_time`, `horizon_code` (`intraday`, `swing`, `position`), `entry_price`, `tp_percent`, `sl_percent`, `label_long`, `label_short`, `pnl_long`, `pnl_short`, `max_drawdown`, `max_runup`. |
| `backtest_runs` | Metadata per labeling/backtest job (`id`, `horizon_code`, `tp/sl schema`, `params_json`, `started_at`, `finished_at`, `status`). |
| `slippage_impact_models` | Specs and metrics for execution cost models. | `id`, `secid (nullable)`, `timeframe`, `as_of`, `spec_json`, `metrics_json`. |

## 6. Modeling & Strategy Council

| Table | Purpose | Notes |
| --- | --- | --- |
| `model_runs` | Versioned experiments for global temporal model. | `id`, `model_name`, `run_id`, `data_span`, `feature_set`, `metric_json`, `artifact_uri`, `created_by`. |
| `strategy_signals` | Output per strategy head before aggregation. | `id`, `strategy_code (trend/mean_reversion/news/vol)`, `model_run_id`, `secid`, `timeframe`, `signal_time`, `p_success_long`, `p_success_short`, `tp_percent`, `sl_percent`, `confidence`, `explanation`. |
| `aggregated_signals` | Meta-head final decision. | `id`, `model_run_id`, `secid`, `timeframe`, `signal_time`, `signal` (`BUY/SELL/HOLD/NO_TRADE`), `entry_range_low/high`, `tp_price`, `sl_price`, `confidence`, `market_regime`, `scenario_id (FK)`, `strategy_votes_json`, `explanation`. |
| `market_regimes` | Output of regime classifier. | `id`, `timestamp`, `scope` (`market`, `sector`, `secid`), `value` (`uptrend`, `panic`, etc.), `probabilities_json`, `features_snapshot`. |
| `scenario_forecasts` | Price-range probability scenarios for each horizon. | `id`, `secid`, `timeframe`, `signal_time`, `horizon_code`, `bins_json` (list of {range_low, range_high, probability}), `derived_from_run_id`. |
| `adapter_overrides` | Per-security adapter corrections. | `id`, `secid`, `adapter_version`, `source_strategy_signal_id`, `p_success_long_adj`, `p_success_short_adj`, `tp_adj`, `sl_adj`, `confidence_adj`, `trained_on_span`, `metrics_json`. |
| `model_monitoring` | Production inference metrics/time series. | `id`, `secid`, `timeframe`, `timestamp`, `signal_latency_ms`, `inference_status`, `drift_score`, `notes`. |
| `feature_registry` | Registry of features and their owners. | `id`, `feature_name`, `owner`, `version`, `definition_ref`, `tests_json`, `status`, `created_at`. |
| `feature_lineage` | Lineage for features. | `id`, `feature_name`, `version`, `upstream_tables`, `code_ref`, `checksum`. |
| `model_registry` | Canonical registry of trainable models. | `id`, `model_name`, `version`, `stage` (`staging/prod/archived`), `approval_by`, `approved_at`, `artifact_uri`, `signature`. |
| `deployments` | Model deployment events. | `id`, `model_name`, `version`, `deployed_at`, `traffic_pct`, `rollback_to (nullable)`, `status`. |
| `ab_tests` | A/B tests over models/strategies. | `id`, `scope` (`inference/portfolio`), `start_at`, `end_at`, `arms_json`, `primary_metric`, `result_json`. |
| `canary_runs` | Canary deployments of new models. | `id`, `deployment_id`, `start_at`, `end_at`, `issues_json`, `decision`. |
| `rollback_events` | Rollbacks for failed deployments. | `id`, `deployment_id`, `rolled_back_at`, `reason`, `trigger_ref`. |
| `market_regime_details` | Детализированные режимы рынка/ликвидности. | `id`, `timestamp`, `scope` (`market/sector/secid`), `base_regime` (link to `market_regimes`), `liquidity_regime`, `spread_regime`, `news_burst_level`, `derived_from`, `features_snapshot`. |
| `strategy_regime_policies` | Связь стратегий с режимами. | `id`, `strategy_code`, `regime_pattern`, `enabled`, `weight_adjustment`, `rules_json`. |
| `policy_runs` | Запуски политик contextual bandits/RL. | `id`, `policy_name`, `version`, `start_at`, `end_at (nullable)`, `config_json`, `status`, `notes`. |
| `policy_feedback` | Обратная связь по решениям policy. | `id`, `policy_run_id`, `timestamp`, `user_id (nullable)`, `secid (nullable)`, `context_hash`, `chosen_action`, `reward`, `reward_components_json`. |

## 7. Portfolio, Behaviour & Training Modules

| Table | Purpose |
| --- | --- |
| `user_portfolios` | Snapshot of user accounts (FK `users`). Columns: `id`, `user_id`, `as_of`, `currency`, `equity`, `cash`, `leverage`, `notes`. |
| `portfolio_positions` | Holdings per snapshot. | `id`, `portfolio_id`, `secid`, `side` (`long/short`), `quantity`, `avg_price`, `market_value`, `pnl`, `risk_bucket`. |
| `portfolio_recommendations` | Output from `/analyze_portfolio`. | `id`, `user_id`, `secid`, `generated_at`, `action` (`increase/hold/trim/exit`), `rationale`, `risk_flags`. |
| `user_behaviour_events` | Anti-tilt log of actions vs signals. | `id`, `user_id`, `event_time`, `event_type` (`stop_violation`, `overtrading`, ...), `related_signal_id`, `details_json`. |
| `user_weekly_reports` | Persisted reports for GET endpoint. | `id`, `user_id`, `week_start`, `summary`, `stats_json`, `recommendations_json`. |
| `training_scenarios` | Historical scenarios served to users. | `id`, `secid`, `timeframe`, `scenario_time`, `context_json`, `created_at`. |
| `training_attempts` | Results of "Ты vs AI" games. | `id`, `scenario_id`, `user_id`, `user_action`, `ai_action`, `outcome_json`, `score_delta`, `created_at`. |
| `user_risk_profiles` | Риск‑профиль пользователя. | `id`, `user_id`, `created_at`, `updated_at`, `horizon_preference`, `max_drawdown_tolerance`, `risk_score`, `instrument_restrictions_json`, `notes`. |
| `user_preferences` | Инвестиционные предпочтения. | `id`, `user_id`, `created_at`, `esg_preference`, `sector_blacklist`, `country_blacklist`, `style_preference` (`growth/value/dividend`), `ui_experience_level`, `extra_json`. |
| `risk_factors` | Market/sector/FX/oil/etc. factor time series. | `id`, `as_of`, `factor_code`, `value`. |
| `security_betas` | Security factor loadings. | `id`, `secid`, `as_of`, `factor_code`, `beta`, `method`. |
| `portfolio_risk_metrics` | Portfolio risk measures. | `id`, `portfolio_id`, `as_of`, `var_1d`, `es_1d`, `beta_market`, `sector_exposures_json`, `turnover_30d`. |
| `stress_tests` | Portfolio stress test results. | `id`, `portfolio_id`, `as_of`, `scenario_code`, `shocks_json`, `pnl_impact`, `risk_notes`. |
| `compliance_rules` | Trading and exposure rules. | `id`, `rule_code`, `description`, `severity`, `effective_from`, `params_json`. |
| `restricted_securities` | Restrictions per security/rule. | `id`, `secid`, `rule_id`, `effective_from`, `expires_at`, `reason`. |
| `pretrade_checks` | Logged pre-trade compliance checks. | `id`, `user_id`, `timestamp`, `secid`, `check_code`, `result`, `details_json`. |

## 8. API & Logging Support

| Table | Purpose |
| --- | --- |
| `inference_requests` | Audit log for POST `/analyze_ticker` etc. Columns: `id`, `user_id`, `endpoint`, `payload`, `response`, `response_time_ms`, `status_code`, `created_at`. |
| `etl_jobs` | Status of scheduled ETL/feature/model tasks. | `id`, `job_name`, `scheduled_at`, `started_at`, `finished_at`, `status`, `rows_processed`, `error`. |
| `data_quality_alerts` | Detected gaps/anomalies (missing candles, stale fundamentals). | `id`, `alert_type`, `secid`, `timeframe`, `detected_at`, `severity`, `status`, `details`. |
| `alerts_stream` | User-facing alerts feed. | `id`, `user_id (nullable)`, `secid (nullable)`, `timestamp`, `alert_type` (`risk`, `compliance`, `news`, ...), `payload_json`, `seen_at`. |
| `scenario_explanations` | Stored explanations for scenarios. | `id`, `scenario_id`, `explanation_json`, `created_at`. |
| `data_freshness` | Data latency and freshness metrics. | `id`, `table_name`, `partition_key`, `expected_lag_sec`, `last_seen_at`, `status`. |
| `drift_metrics` | Feature/label drift tracking. | `id`, `secid (nullable)`, `timeframe`, `timestamp`, `feature_drift_score`, `label_drift_score`, `method`, `details_json`. |

## 9. Relationships & Flows (summary)

1. **Reference**: `securities` is referenced by almost every domain table (candles, fundamentals, signals, portfolios).
2. **Data ingestion**: `candles` + `index_candles` feed the feature factory → `feature_windows`/`features_*` → `trade_labels` for supervised tasks.
3. **Models**: `model_runs` describe trained artifacts; `strategy_signals` store per-head outputs; `aggregated_signals` combine them; `adapter_overrides` adjust per ticker.
4. **Regime & scenarios** attach to aggregated signals for UI explanations.
5. **Portfolios & behaviour** reuse `users` from the existing backend; they connect to `aggregated_signals` via `related_signal_id` for diagnostics.
6. **API** endpoints log to `inference_requests` and surface `aggregated_signals`, `portfolio_recommendations`, `scenario_forecasts`, `user_weekly_reports`.

## 10. Implementation Notes

- **Partitioning**: Use monthly partitions for `candles`, `strategy_signals`, `aggregated_signals`, `feature_windows`, `trade_labels`. Keep recent partitions in hot storage, archive old ones.
- **Indexing**: Composite indexes on `(secid, timeframe, datetime)` for time-series tables; GIN indexes for JSON columns with query access (e.g., tags, votes).
- **Timescale compatibility**: `candles` and `strategy_signals` benefit from hypertables if TimescaleDB is available.
- **Vector storage**: `news_embeddings`, `features_vector` may use `pgvector` if semantic search/drift detection is needed.
- **Foreign keys vs. soft links**: For ultra-high-volume pipelines we can enforce referential integrity at ingestion time, but for ingestion bursts we may stage data in raw tables before merging.
- **Migration strategy**: Use Alembic migrations; start by extending existing schema (`users`, `news`, `analysis_results`) with FK links to new tables to avoid breaking current features.
- **Security**: separate schemas (`core`, `market_data`, `ml`, `analytics`) with grants per service (ETL, API, research, monitoring).
- **Historical training window**: pipelines и storage рассчитаны на минимум 8 лет ретроспективы; контроль полноты через `train_data_snapshots`, деградация качества → алерт в `data_quality_alerts`.
- **Global news latency**: ingest-связка `news_sources → news_events → global_risk_events → risk_alerts` должна отрабатывать в минутах; при задержках логировать в `etl_jobs` и эскалировать.

This schema provides the foundation for the staged implementation plan (data layer → feature factory → modeling → portfolio/behaviour → API). Once approved, we can scaffold Alembic migrations and service modules accordingly.
