import type { AnalyzeRequest, AnalyzeResponse, HistoryItem } from './types'

const LOCAL_API_FALLBACK = 'http://localhost:8000/api'

const getDefaultApiUrl = (): string => {
  if (typeof window === 'undefined') {
    return LOCAL_API_FALLBACK
  }
  try {
    const url = new URL(window.location.href)
    const hostname = url.hostname
    const isLocalHost =
      hostname === 'localhost' ||
      hostname === '127.0.0.1' ||
      hostname === '[::1]' ||
      hostname.endsWith('.localhost')

    if (isLocalHost) {
      return LOCAL_API_FALLBACK
    }

    return `${url.origin}/api`
  } catch {
    return LOCAL_API_FALLBACK
  }
}

const API_URL = import.meta.env.VITE_API_URL || getDefaultApiUrl()

export async function analyze(req: AnalyzeRequest): Promise<AnalyzeResponse> {
  const res = await fetch(`${API_URL}/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
  if (!res.ok) {
    throw new Error(`Request failed: ${res.status}`)
  }
  return res.json()
}

export async function getHistory(telegramId: string, limit = 20, initData?: string): Promise<HistoryItem[]> {
  const url = new URL(`${API_URL}/history`)
  url.searchParams.set('telegram_id', telegramId)
  url.searchParams.set('limit', String(limit))
  if (initData) {
    url.searchParams.set('init_data', initData)
  }
  const res = await fetch(url)
  if (!res.ok) {
    throw new Error(`History request failed: ${res.status}`)
  }
  return res.json()
}
