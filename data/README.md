# Dataset Structure

This folder contains the curated dataset used for training and evaluating the baseline TriggerScore model.

```
data/
├── raw/           # raw JSONL / CSV exports collected from public sources
├── processed/     # cleaned dataset ready for training
└── labeling/      # helper spreadsheets or guidelines (optional)
```

Populate `raw/news_sample.jsonl` with semi-structured news items. Each line is expected to be a JSON object with:

```json
{
  "id": "2024-06-12-rbc-001",
  "source": "RBC",
  "url": "https://www.rbc.ru/some-news",
  "published_at": "2024-06-12T08:05:00Z",
  "title": "ЦБ сохранил ключевую ставку",
  "text": "... полный текст новости ..."
}
```

After manual labeling, create a `processed/labeled_news.csv` file with the following columns:

```
news_id,title,text,direction,event_type,trigger_score,horizon,assets
```

- `direction`: `bullish`, `bearish`, `neutral`
- `event_type`: one of `earnings`, `dividends`, `sanctions`, `macro`, `regulation`, `mna`, `default`, `other`
- `trigger_score`: integer 0–100 (10/30/60/90 baseline buckets)
- `horizon`: `intraday`, `1-3d`, `<1m`
- `assets`: semicolon separated tickers or sectors

The scripts in `backend/scripts/` assume this structure when training or benchmarking the baseline model.
