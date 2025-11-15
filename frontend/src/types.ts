export type AnalyzeRequest = {
  text: string
  telegram_id?: string
  source_url?: string
  title?: string
  init_data?: string
}

export type AnalyzeResponse = {
  news_id: number
  analysis_id: number
  trigger_score: number
  direction: 'bullish' | 'bearish' | 'neutral'
  event_type?: string
  horizon?: string | null
  assets: string[]
  summary?: string
}

export type HistoryItem = {
  news_id: number
  analysis_id: number
  created_at: string
  title?: string | null
  text?: string | null
  trigger_score: number
  direction: 'bullish' | 'bearish' | 'neutral'
  event_type?: string | null
  horizon?: string | null
  assets: string[]
  summary?: string | null
}

export type FeedbackVerdict = 'agree' | 'disagree'

export type FeedbackRequest = {
  analysis_id: number
  verdict: FeedbackVerdict
  telegram_id?: string
  init_data?: string
  comment?: string
}

export type FeedbackResponse = {
  analysis_id: number
  verdict: FeedbackVerdict
  comment?: string | null
  created_at: string
  updated_at: string
}
