# Feature factory quickstart

## Dev shell

```powershell
./scripts/start_dev_shell.ps1
```

## Data ingestion

```powershell
python -m backend.scripts.ingestion.candles SBER GAZP --timeframe 1m --start-date 2025-11-01 --end-date 2025-11-05
```

## Feature build

```powershell
python -m backend.scripts.features.build_window SBER GAZP --timeframe 1m --start-date 2025-11-01T07:00:00+03:00 --end-date 2025-11-05T23:59:59+03:00
```

## Label build

```powershell
python -m backend.scripts.labels.build_labels SBER GAZP --timeframe 1m --start-date 2025-11-01T07:00:00+03:00 --end-date 2025-11-05T23:59:59+03:00 --horizons 60 240 1440 --take-profit 0.02 --stop-loss 0.01 --label-set basic_v1
```

## Prefect automation

### Candles → Features → Labels

```powershell
python -m backend.scripts.scheduler.register_deployments `
	--work-pool local-agent-pool `
	--work-queue default `
	--storage-block local-repo-storage `
	--ignore-warnings `
	--apply

prefect deployment run 'candles-and-features-flow/candles-and-features-daily'
```

### Trade-label monitoring flow

```powershell
python -m backend.scripts.scheduler.register_deployments `
	--flow-name trade-label-monitoring-flow `
	--monitor-lookback-days 3 `
	--monitor-min-rows 1200 `
	--monitor-fail-on-empty `
	--work-pool local-agent-pool `
	--work-queue default `
	--storage-block local-repo-storage `
	--ignore-warnings `
	--apply

prefect deployment run 'trade-label-monitoring-flow/trade-label-monitoring-flow'
```

## Prefect worker lifecycle

```powershell
# Foreground worker (Ctrl+C to stop)
prefect worker start --pool local-agent-pool

# Background job
Start-Job -Name PrefectWorker -ScriptBlock {
	& E:/TRAIN/AUTO-TRADER/.venv/Scripts/prefect.exe worker start --pool local-agent-pool
}

# Monitor / stop background worker
Get-Job -Name PrefectWorker
Receive-Job -Name PrefectWorker -Keep
Stop-Job -Name PrefectWorker; Remove-Job -Name PrefectWorker
```

## Baseline dataset (features + labels)

```powershell
python -m backend.scripts.ml.prepare_baseline_dataset `
	--secids SBER GAZP `
	--timeframe 1m `
	--feature-set tech_v1 `
	--label-set basic_v1 `
	--start-date 2025-11-01T07:00:00+03:00 `
	--end-date 2025-11-05T23:59:59+03:00 `
	--output data/training/baseline_dataset.csv
```

Используйте CSV выше в `notebooks/trade_labels_eda.ipynb`, чтобы проверить совмещённые фичи и лейблы и подготовить baseline-модель.