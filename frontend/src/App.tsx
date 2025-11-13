import { useCallback, useEffect, useMemo, useState } from 'react'
import type { KeyboardEvent as ReactKeyboardEvent } from 'react'
import { analyze, getHistory } from './api'
import type { AnalyzeResponse, HistoryItem } from './types'
import './App.css'

const directionLabels: Record<'bullish' | 'bearish' | 'neutral', string> = {
  bullish: 'Позитивно',
  bearish: 'Негативно',
  neutral: 'Нейтрально',
}

const debug = (...args: unknown[]) => {
  console.info('[Miniapp]', ...args)
}

// Minimal Telegram WebApp init
declare global {
  interface Window {
    Telegram?: any
  }
}

export default function App() {
  const [text, setText] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<AnalyzeResponse | null>(null)
  const [activeTab, setActiveTab] = useState<'analyze' | 'history'>('analyze')
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [historyLoading, setHistoryLoading] = useState(false)
  const [historyError, setHistoryError] = useState<string | null>(null)
  const [selectedHistory, setSelectedHistory] = useState<HistoryItem | null>(null)

  const telegramId = useMemo(() => {
    try {
      const id = window?.Telegram?.WebApp?.initDataUnsafe?.user?.id?.toString?.()
      debug('Detected Telegram user id', id)
      return id ?? undefined
    } catch {
      return undefined
    }
  }, [])

  const initData = useMemo(() => {
    try {
      const raw = window?.Telegram?.WebApp?.initData ?? ''
      if (raw) {
        debug('Received initData string', { length: raw.length })
      } else {
        debug('initData string is empty')
      }
      return raw
    } catch {
      return ''
    }
  }, [])

  useEffect(() => {
    try {
      window?.Telegram?.WebApp?.ready?.()
      debug('Telegram WebApp ready invoked')
    } catch {
      // noop
    }
  }, [])

  useEffect(() => {
    debug('App mounted', {
      telegramId: telegramId ?? null,
      initDataLength: initData?.length ?? 0,
    })
  }, [initData, telegramId])

  const onAnalyze = async () => {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      debug('Sending analyze request', { hasText: text.trim().length > 0, telegramId })
      const res = await analyze({
        text,
        telegram_id: telegramId,
        init_data: initData || undefined,
      })
      debug('Analyze response received', res)
      setResult(res)
      if (activeTab === 'history') {
        await loadHistory()
      }
    } catch (e: any) {
      debug('Analyze request failed', e)
      setError(e?.message ?? 'Ошибка запроса')
    } finally {
      setLoading(false)
    }
  }

  const loadHistory = useCallback(async () => {
    if (!telegramId) {
      setHistory([])
      debug('History skipped: no telegramId')
      return
    }
    setHistoryLoading(true)
    setHistoryError(null)
    try {
      debug('Loading history', { telegramId, limit: 20 })
      const items = await getHistory(telegramId, 20, initData || undefined)
      debug('History loaded', { count: items.length })
      setHistory(items)
    } catch (e: any) {
      debug('History request failed', e)
      setHistoryError(e?.message ?? 'Не удалось загрузить историю')
    } finally {
      setHistoryLoading(false)
    }
  }, [initData, telegramId])

  const closeHistoryDetail = useCallback(() => {
    setSelectedHistory(null)
  }, [])

  const handleHistoryCardClick = useCallback((item: HistoryItem) => {
    setSelectedHistory(item)
  }, [])

  const handleHistoryCardKeyDown = useCallback((event: ReactKeyboardEvent<HTMLDivElement>, item: HistoryItem) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      setSelectedHistory(item)
    }
  }, [])

  useEffect(() => {
    if (activeTab === 'history') {
      debug('History tab activated')
      loadHistory()
    }
  }, [activeTab, loadHistory])

  useEffect(() => {
    if (activeTab !== 'history') {
      debug('History tab deactivated, closing modal')
      closeHistoryDetail()
    }
  }, [activeTab, closeHistoryDetail])

  useEffect(() => {
    if (!selectedHistory) {
      return
    }
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        debug('Escape pressed, closing history modal')
        event.preventDefault()
        closeHistoryDetail()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [closeHistoryDetail, selectedHistory])

  return (
    <div className="app-container">
      <h1 className="app-title">Триггер-аналитика</h1>
      <div className="tab-group">
        <button
          onClick={() => setActiveTab('analyze')}
          className={`tab-button ${activeTab === 'analyze' ? 'active' : ''}`}
        >
          Анализ
        </button>
        <button
          onClick={() => setActiveTab('history')}
          className={`tab-button ${activeTab === 'history' ? 'active' : ''}`}
        >
          История
        </button>
      </div>

      {activeTab === 'analyze' && (
        <div>
          <textarea
            placeholder="Вставьте ссылку или текст новости"
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={8}
            className="textarea"
          />
          <button
            disabled={loading || text.trim().length < 5}
            onClick={onAnalyze}
            className="primary-btn"
          >
            {loading ? 'Анализ...' : 'Анализировать'}
          </button>

          {error && (
            <div className="error-text">
              Ошибка: {error}
            </div>
          )}

          {result && (
            <div className="card">
              <div className="card-header">
                <div className="card-score-value">{Math.round(result.trigger_score)}</div>
                <div className="card-score-label">TriggerScore</div>
              </div>
              <div className="card-row">
                Направление:{' '}
                <span className={`direction-pill direction-${result.direction}`}>
                  {directionLabels[result.direction]}
                </span>
              </div>
              {result.event_type && (
                <div className="card-row">
                  Тип события: <b>{result.event_type}</b>
                </div>
              )}
              {result.summary && <div className="card-row">{result.summary}</div>}
              {result.assets?.length ? (
                <div className="card-row">Активы: {result.assets.join(', ')}</div>
              ) : null}
            </div>
          )}
        </div>
      )}

      {activeTab === 'history' && (
        <div>
          <div className="history-header">
            <h2 className="history-title">Последние запросы</h2>
            <button onClick={loadHistory} disabled={historyLoading} className="secondary-btn">
              {historyLoading ? 'Обновление...' : 'Обновить'}
            </button>
          </div>
          {!telegramId && (
            <div className="history-hint">
              История появится после авторизации через Telegram Miniapp. Проверьте, что Miniapp открыт внутри Telegram.
            </div>
          )}
          {historyError && <div className="error-text">{historyError}</div>}

          {history.length === 0 && !historyLoading && telegramId && !historyError && (
            <div className="history-empty">Запросов пока нет.</div>
          )}

          <div className="history-list">
            {history.map((item) => (
              <div
                key={item.analysis_id}
                className={`history-card direction-${item.direction}`}
                role="button"
                tabIndex={0}
                aria-label={`Детали запроса ${item.title ?? 'без заголовка'}`}
                onClick={() => handleHistoryCardClick(item)}
                onKeyDown={(event) => handleHistoryCardKeyDown(event, item)}
              >
                <div className="history-card-header">
                  <div className="history-card-title">{item.title ?? 'Без заголовка'}</div>
                  <div className="history-card-date">{new Date(item.created_at).toLocaleString()}</div>
                </div>
                <div className="history-card-text">
                  {item.text?.slice(0, 160) ?? 'Текст недоступен'}{item.text && item.text.length > 160 ? '…' : ''}
                </div>
                <div className="history-card-meta">
                  <span>Score: <b>{Math.round(item.trigger_score)}</b></span>
                  <span className="history-direction">
                    Направление:{' '}
                    <span className={`direction-pill direction-${item.direction}`}>
                      {directionLabels[item.direction]}
                    </span>
                  </span>
                  {item.event_type && <span>Событие: <b>{item.event_type}</b></span>}
                </div>
              </div>
            ))}
          </div>

          {selectedHistory && (
            <div className="modal-backdrop" role="presentation" onClick={closeHistoryDetail}>
              <div
                className="modal-content"
                role="dialog"
                aria-modal="true"
                aria-labelledby="history-modal-title"
                onClick={(event) => event.stopPropagation()}
              >
                <button type="button" className="modal-close" onClick={closeHistoryDetail}>
                  Закрыть
                </button>
                <h3 id="history-modal-title">{selectedHistory.title ?? 'Без заголовка'}</h3>
                <div className="modal-meta">
                  <span>{new Date(selectedHistory.created_at).toLocaleString()}</span>
                  <span className={`direction-pill direction-${selectedHistory.direction}`}>
                    {directionLabels[selectedHistory.direction]}
                  </span>
                  <span>
                    Score: <b>{Math.round(selectedHistory.trigger_score)}</b>
                  </span>
                  {selectedHistory.event_type && (
                    <span>
                      Событие: <b>{selectedHistory.event_type}</b>
                    </span>
                  )}
                  {selectedHistory.horizon && (
                    <span>
                      Горизонт: <b>{selectedHistory.horizon}</b>
                    </span>
                  )}
                </div>
                {selectedHistory.summary && (
                  <div className="modal-section">
                    <h4>Summary</h4>
                    <p>{selectedHistory.summary}</p>
                  </div>
                )}
                {selectedHistory.assets?.length ? (
                  <div className="modal-section">
                    <h4>Активы</h4>
                    <div className="modal-assets">
                      {selectedHistory.assets.map((asset) => (
                        <span key={asset} className="asset-chip">{asset}</span>
                      ))}
                    </div>
                  </div>
                ) : null}
                <div className="modal-section">
                  <h4>Текст новости</h4>
                  <p className="modal-text">{selectedHistory.text ?? 'Текст недоступен'}</p>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
