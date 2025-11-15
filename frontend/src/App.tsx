import { useCallback, useEffect, useMemo, useState } from 'react'
import type { KeyboardEvent as ReactKeyboardEvent } from 'react'
import { analyze, getHistory, submitFeedback } from './api'
import type { AnalyzeResponse, HistoryItem, FeedbackVerdict } from './types'
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
  const [feedbackSending, setFeedbackSending] = useState(false)
  const [feedbackVerdict, setFeedbackVerdict] = useState<FeedbackVerdict | null>(null)
  const [feedbackError, setFeedbackError] = useState<string | null>(null)
  const [feedbackMessage, setFeedbackMessage] = useState<string | null>(null)

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

  const handleFeedback = useCallback(
    async (verdict: FeedbackVerdict) => {
      if (!result) {
        return
      }
      if (!telegramId) {
        setFeedbackError('Нужно открыть миниапп внутри Telegram, чтобы оставить отклик.')
        return
      }
      if (feedbackSending) {
        return
      }
      if (feedbackVerdict === verdict && !feedbackError) {
        return
      }

      setFeedbackSending(true)
      setFeedbackError(null)
      try {
        const response = await submitFeedback({
          analysis_id: result.analysis_id,
          verdict,
          telegram_id: telegramId,
          init_data: initData || undefined,
        })
        setFeedbackVerdict(response.verdict)
        setFeedbackMessage(
          response.verdict === 'agree'
            ? 'Спасибо! Этот сигнал подтверждён.'
            : 'Приняли несогласие — использую для дообучения.'
        )
      } catch (err: any) {
        setFeedbackError(err?.message ?? 'Не удалось сохранить обратную связь')
      } finally {
        setFeedbackSending(false)
      }
    },
    [feedbackError, feedbackSending, feedbackVerdict, initData, result, telegramId]
  )

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

  useEffect(() => {
    setFeedbackVerdict(null)
    setFeedbackError(null)
    setFeedbackMessage(null)
    setFeedbackSending(false)
  }, [result?.analysis_id])

  return (
    <div className="app-shell">
      <div className="app-ambient" aria-hidden="true" />

      <header className="hero">
        <span className="hero-badge">TriggerSense</span>
        <h1 className="hero-title">Триггер-аналитика в два тапа</h1>
        <p className="hero-caption">
          Лаконичный AI-ассистент для трейдеров: вставьте новость — получите направление, резюме и активы прямо в миниаппе.
        </p>
      </header>

      <section className="panel">
        <div className="tab-bar" role="tablist" aria-label="Основные разделы">
          <button
            type="button"
            onClick={() => setActiveTab('analyze')}
            className={`tab-button ${activeTab === 'analyze' ? 'active' : ''}`}
            role="tab"
          >
            Анализ
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('history')}
            className={`tab-button ${activeTab === 'history' ? 'active' : ''}`}
            role="tab"
          >
            История
          </button>
        </div>

        <div className="panel-content">
          {activeTab === 'analyze' && (
            <div className="analyze-block">
              <label htmlFor="news-input" className="input-label">
                Вставьте ссылку или текст новости
              </label>
              <textarea
                id="news-input"
                placeholder="Например, твит Илона или краткое описание сделки"
                value={text}
                onChange={(e) => setText(e.target.value)}
                rows={6}
                className="textarea"
              />
              <button
                type="button"
                disabled={loading || text.trim().length < 5}
                onClick={onAnalyze}
                className="primary-btn"
              >
                {loading ? 'Анализ...' : 'Анализировать'}
              </button>

              {error && (
                <div className="feedback feedback-error" role="status">
                  Ошибка: {error}
                </div>
              )}

              {result && (
                <div className="result-card">
                  <div className="result-header">
                    <div className="score-orb">
                      <span>{Math.round(result.trigger_score)}</span>
                    </div>
                    <div className="result-headline">
                      <span className="result-label">TriggerScore</span>
                      <span className={`direction-pill direction-${result.direction}`}>
                        {directionLabels[result.direction]}
                      </span>
                    </div>
                  </div>

                  <div className="result-body">
                    {result.summary && <p className="result-summary">{result.summary}</p>}

                    <div className="result-grid">
                      {result.event_type && (
                        <div className="result-item">
                          <span className="result-item-label">Тип события</span>
                          <span className="result-item-value">{result.event_type}</span>
                        </div>
                      )}

                      {result.assets?.length ? (
                        <div className="result-item">
                          <span className="result-item-label">Активы</span>
                          <span className="result-item-value">{result.assets.join(', ')}</span>
                        </div>
                      ) : null}
                    </div>
                  </div>

                  <div className="feedback-actions">
                    <span className="feedback-actions-label">Как оцените предсказание?</span>
                    <div className="feedback-buttons">
                      <button
                        type="button"
                        className={`feedback-btn ${feedbackVerdict === 'agree' ? 'active' : ''}`}
                        onClick={() => handleFeedback('agree')}
                        disabled={!telegramId || feedbackSending}
                      >
                        Триггер оценён верно
                      </button>
                      <button
                        type="button"
                        className={`feedback-btn ${feedbackVerdict === 'disagree' ? 'active' : ''}`}
                        onClick={() => handleFeedback('disagree')}
                        disabled={!telegramId || feedbackSending}
                      >
                        Не согласен с оценкой
                      </button>
                    </div>
                    {!telegramId && (
                      <div className="feedback-hint">
                        Чтобы делиться откликом, откройте миниапп внутри Telegram и авторизуйтесь.
                      </div>
                    )}
                    {feedbackError && <div className="feedback feedback-error">{feedbackError}</div>}
                    {feedbackVerdict && !feedbackError && feedbackMessage && (
                      <div className="feedback feedback-success">{feedbackMessage}</div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'history' && (
            <div className="history-block">
              <div className="history-header">
                <div>
                  <h2 className="history-title">Последние запросы</h2>
                  <p className="history-subtitle">Эти сессии синхронизированы с вашим Telegram ID</p>
                </div>
                <button onClick={loadHistory} disabled={historyLoading} className="secondary-btn" type="button">
                  {historyLoading ? 'Обновление...' : 'Обновить'}
                </button>
              </div>
              {!telegramId && (
                <div className="history-hint">
                  История появится после авторизации через Telegram Miniapp. Проверьте, что Miniapp открыт внутри Telegram.
                </div>
              )}
              {historyError && <div className="feedback feedback-error">{historyError}</div>}

              {history.length === 0 && !historyLoading && telegramId && !historyError && (
                <div className="history-empty">Запросов пока нет — как только заанализируете новость, она появится здесь.</div>
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
      </section>
    </div>
  )
}
