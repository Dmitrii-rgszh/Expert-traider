import hashlib
import hmac
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

os.environ.setdefault("DISABLE_TRANSFORMERS", "true")
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "123456:TESTTOKEN")
os.environ.setdefault("TELEGRAM_DAILY_REQUEST_CAP", "2")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from app.main import app
from app.db.session import Base, engine
from app.core.config import get_settings

Base.metadata.create_all(bind=engine)

BOT_TOKEN = get_settings().telegram_bot_token or ""


def make_init_data(telegram_id: int) -> str:
    payload = {
        "auth_date": str(int(time.time())),
        "query_id": "AAH-test",
        "user": json.dumps({"id": telegram_id, "first_name": "Test"}),
    }
    data_check_string = "\n".join(f"{k}={v}" for k, v in sorted(payload.items()))
    secret = hashlib.sha256(f"WebAppData{BOT_TOKEN}".encode("utf-8")).digest()
    payload["hash"] = hmac.new(secret, data_check_string.encode("utf-8"), hashlib.sha256).hexdigest()
    return urlencode(payload)


client = TestClient(app)


def test_health():
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json().get('status') == 'ok'

def test_analyze_baseline():
    user_id = 1001
    init_data = make_init_data(user_id)
    payload = {
        "text": "Сбербанк объявил о росте прибыли и увеличении дивидендов.",
        "title": "Сбербанк усиливает дивидендную программу",
        "telegram_id": str(user_id),
        "init_data": init_data,
    }
    r = client.post('/api/analyze', json=payload)
    assert r.status_code == 200
    data = r.json()
    assert 'trigger_score' in data
    assert data['direction'] in ['bullish', 'bearish', 'neutral']
    assert 'news_id' in data and 'analysis_id' in data
    assert data['summary']

def test_history_endpoint():
    user_id = 2001
    init_data = make_init_data(user_id)
    payload = {
        "text": "Газпром сообщил о снижении экспорта.",
        "title": "Газпром снижает экспорт",
        "telegram_id": str(user_id),
        "init_data": init_data,
    }
    for _ in range(2):
        payload["init_data"] = make_init_data(user_id)
        res = client.post('/api/analyze', json=payload)
        assert res.status_code == 200

    history_res = client.get('/api/history', params={
        "telegram_id": str(user_id),
        "limit": 5,
        "init_data": make_init_data(user_id),
    })
    assert history_res.status_code == 200
    history = history_res.json()
    assert len(history) >= 2
    first = history[0]
    assert first['analysis_id']
    assert first['trigger_score']
    assert first['direction'] in ['bullish', 'bearish', 'neutral']


def test_request_cap_enforced():
    user_id = 3001
    payload = {
        "text": "Тестовая новость о рынке.",
        "title": "Рынок тестируется",
        "telegram_id": str(user_id),
    }
    for _ in range(2):
        payload["init_data"] = make_init_data(user_id)
        res = client.post('/api/analyze', json=payload)
        assert res.status_code == 200

    payload["init_data"] = make_init_data(user_id)
    res = client.post('/api/analyze', json=payload)
    assert res.status_code == 429


def test_history_requires_init_data():
    user_id = 4001
    res = client.get('/api/history', params={"telegram_id": str(user_id), "limit": 5})
    assert res.status_code == 401
