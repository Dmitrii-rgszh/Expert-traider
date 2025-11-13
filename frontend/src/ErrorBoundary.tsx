import type { PropsWithChildren } from 'react'
import { Component } from 'react'
import './ErrorBoundary.css'

type ErrorBoundaryProps = PropsWithChildren<{
  onError?: (error: Error, info: string) => void
}>

type ErrorBoundaryState = {
  hasError: boolean
  message?: string
  stack?: string
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  override state: ErrorBoundaryState = { hasError: false }

  override componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error('[Miniapp] Unhandled render error', error, info)
    this.setState({ hasError: true, message: error.message, stack: error.stack })
    this.props.onError?.(error, info.componentStack ?? '')
  }

  override render() {
    if (this.state.hasError) {
      return (
        <div className="error-boundary">
          <h1 className="error-boundary__title">Что-то пошло не так</h1>
          <p className="error-boundary__message">{this.state.message ?? 'Неизвестная ошибка UI'}</p>
          {this.state.stack ? <pre className="error-boundary__stack">{this.state.stack}</pre> : null}
          <p className="error-boundary__hint">
            Проверьте консоль браузера (F12 → Console) и перезапустите страницу. Если проблема повторяется, пришлите скриншоты логов.
          </p>
        </div>
      )
    }

    return this.props.children
  }
}
