import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import { ErrorBoundary } from './ErrorBoundary'

console.info('[Miniapp] Bootstrap start', new Date().toISOString())

window.addEventListener('error', (event) => {
  console.error('[Miniapp] Global error event', event.error ?? event.message)
})

window.addEventListener('unhandledrejection', (event) => {
  console.error('[Miniapp] Unhandled rejection', event.reason)
})

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ErrorBoundary onError={(error) => console.error('[Miniapp] Error boundary caught', error)}>
      <App />
    </ErrorBoundary>
  </React.StrictMode>,
)

console.info('[Miniapp] Bootstrap end', new Date().toISOString())
