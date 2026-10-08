import { useEffect, useRef, useCallback } from 'react'
import { WS_URL } from '../api'
import { useStore } from '../store'

export function useWebSocket() {
  const wsRef = useRef(null)
  const reconnectTimer = useRef(null)
  const { setWsConnected, setLastAnalysis, setLastTradeEvent, updateOverviewFromWs } = useStore()

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    const ws = new WebSocket(`${WS_URL}/dashboard`)
    wsRef.current = ws

    ws.onopen = () => {
      setWsConnected(true)
      console.log('WebSocket connected')
    }

    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data)
        switch (msg.type) {
          case 'portfolio_update':
            updateOverviewFromWs(msg.data)
            break
          case 'trade_event':
            setLastTradeEvent(msg.data)
            break
          case 'analysis_update':
            setLastAnalysis(msg.data)
            break
        }
      } catch (e) {
        console.warn('WS parse error', e)
      }
    }

    ws.onclose = () => {
      setWsConnected(false)
      // Auto-reconnect after 5s
      reconnectTimer.current = setTimeout(connect, 5000)
    }

    ws.onerror = (e) => {
      console.error('WS error', e)
      ws.close()
    }
  }, [setWsConnected, setLastAnalysis, setLastTradeEvent, updateOverviewFromWs])

  useEffect(() => {
    connect()
    return () => {
      clearTimeout(reconnectTimer.current)
      wsRef.current?.close()
    }
  }, [connect])
}

export function formatINR(amount, compact = false) {
  if (amount == null) return '—'
  if (compact && Math.abs(amount) >= 100000) {
    return `₹${(amount / 100000).toFixed(2)}L`
  }
  return new Intl.NumberFormat('en-IN', {
    style: 'currency',
    currency: 'INR',
    maximumFractionDigits: 0,
  }).format(amount)
}

export function formatR(r) {
  if (r == null) return '—'
  const sign = r >= 0 ? '+' : ''
  return `${sign}${r.toFixed(2)}R`
}

export function formatPct(pct) {
  if (pct == null) return '—'
  const sign = pct >= 0 ? '+' : ''
  return `${sign}${pct.toFixed(2)}%`
}

export function pnlClass(val) {
  if (val == null) return ''
  return val > 0 ? 'positive' : val < 0 ? 'negative' : ''
}
