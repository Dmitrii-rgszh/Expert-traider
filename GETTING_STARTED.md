# Auto-Trader MVP — Руководство по запуску (Windows / PowerShell)

Это руководство помогает поднять минимальный рабочий продукт: backend (FastAPI), frontend (Vite + React), Postgres через Docker и минимальный Telegram-бот.

## Требования

- Установленный и запущенный Docker Desktop
- Node.js 18+ и pnpm или npm (любой менеджер)
- Python 3.11+ (для бота)

## 1) Backend + БД

- Скопируйте примерный `.env` и при необходимости поменяйте значения:

```powershell
Copy-Item .\backend\.env.example .\backend\.env
```

- Соберите сервис и поднимите Postgres вместе с backend:

```powershell
# собирает backend и запускает db + backend
docker compose up -d --build
```

- Проверьте здоровье:

```powershell
curl http://localhost:8000/health
```

Ответ должен быть `{ "status": "ok" }`.

## 2) Фронтенд (каркас Miniapp)

```powershell
cd .\frontend
# установка зависимостей
npm install
# или
# pnpm install

# при необходимости переопределите URL API
Copy-Item .\.env.example .\.env

# запуск dev-сервера
npm run dev
# или
# pnpm dev
```

Откройте http://localhost:5173, вставьте текст и нажмите «Анализировать».

## 3) Telegram-бот (опционально)

Создайте бота через BotFather и укажите токен:

```powershell
cd ..\bot
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:TELEGRAM_BOT_TOKEN = "<ваш-токен>"
$env:WEBAPP_URL = "https://<ваш-туннель-или-прод>"  # для десктопа можно http://localhost:5173
python .\main.py
```

Отправьте /start — бот покажет кнопку для открытия веб-приложения.

## Быстрый тест API

```powershell
curl -X POST http://localhost:8000/api/analyze -H "Content-Type: application/json" -d '{"text":"Company X reports record profits"}'
```

Вернётся гибридный JSON (правила + ruBERT) с trigger_score, direction и т. п., а результат попадёт в БД. Перед запуском backend можно выставить `DISABLE_TRANSFORMERS=true`, чтобы пропустить загрузку весов (полезно для CI/офлайна). Если нужен другой sentiment-модель ID или приватный репозиторий, обновите `SENTIMENT_MODEL_ID` и `HUGGINGFACE_TOKEN` в `.env`.

## 4) Туннель + запуск Telegram miniapp

Нужно открыть локальную сборку через HTTPS внутри Telegram? Поднимайте стек с Cloudflared.

1. Установите [Cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/) и выполните `cloudflared login`. Убедитесь, что `%USERPROFILE%\.cloudflared\config.yaml` проксирует фронтенд и `/api/*` через один хост. Пример:

	```yaml
	tunnel: auto-trader
	credentials-file: C:\Users\<you>\.cloudflared\<tunnel-id>.json

	ingress:
	  - hostname: app.glazok.site
	    path: /api/*
	    service: http://localhost:8000
	  - hostname: app.glazok.site
	    service: http://localhost:5173
	  - hostname: www.glazok.site
	    path: /api/*
	    service: http://localhost:8000
	  - hostname: www.glazok.site
	    service: http://localhost:5173
	  - hostname: api.glazok.site
	    service: http://localhost:8000
	  - service: http_status:404
	```

	Так `https://app.glazok.site` (и `https://www.glazok.site`) обслуживает фронтенд и проксирует `/api` на backend, поэтому Telegram Mini App не требует extra CORS.

2. Заполните `backend/.env` секретами (`TELEGRAM_BOT_TOKEN`, `HUGGINGFACE_TOKEN`, и т. д.) и оставьте `WEBAPP_URL=https://app.glazok.site`, чтобы кнопка бота совпадала с туннелем. Скрипт автоматически импортирует `.env` (если есть) и `backend/.env`.
3. Установите зависимости один раз (`python -m venv .venv && pip install -r backend/requirements.txt -r bot/requirements.txt`, `npm install` в `frontend`).
4. Запустите стек (Postgres стартует автоматически; чтобы пропустить, используйте `-SkipDb`):

	```powershell
	pwsh .\scripts\start_stack.ps1
	# без базы
	pwsh .\scripts\start_stack.ps1 -SkipDb
	```

	Скрипт запускает Cloudflared, uvicorn на `127.0.0.1:8000`, Vite dev server на `0.0.0.0:5173` и Telegram-бота (`bot/main.py`). В выводе будут PID процессов и публичные URL, чтобы открыть `https://app.glazok.site` с любого устройства или скормить его BotFather («Web App URL»).

Остановить службы можно через `Stop-Process -Id <PID>` или закрыв PowerShell. Пока туннель активен, Telegram Mini App общается с локальным фронтом и backend по `/api`.

## 5) ML-артефакты для baseline

1. Перенесите размеченный датасет в `data/processed/labeled_news.csv` (схема описана в `data/README.md`).
2. Обновите лёгкие классификаторы/регрессор:

	```powershell
	python .\backend\scripts\train_baseline.py --data data/processed/labeled_news.csv --out backend/models/baseline.joblib
	```

3. Укажите backend путь к артефакту (опционально, иначе будут только правила + sentiment):

	```powershell
	$env:BASELINE_MODEL_PATH = "backend/models/baseline.joblib"
	$env:PYTHONUTF8 = "1"  # для корректного вывода в Windows-консоли
	docker compose up -d --build
	```

## Примечания

- По умолчанию backend использует SQLite (`./backend/app.db`), если `DATABASE_URL` не задан. В Docker применяется Postgres.
- CORS разрешает `http://localhost:5173`, `http://127.0.0.1:5173`, а также туннельные хосты (см. `backend/.env`).
- Sprint 2 добавляет гибридный анализатор. Заполните датасет и обучите baseline, чтобы поднять качество поверх правил и sentiment. При необходимости поменяйте `SENTIMENT_MODEL_ID`.
