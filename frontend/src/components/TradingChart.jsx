import { useEffect, useRef } from 'react'
import { createChart, ColorType } from 'lightweight-charts'

export default function TradingChart({ data, title = 'Chart', height = 380 }) {
  const containerRef = useRef(null)
  const chartRef = useRef(null)
  const seriesRef = useRef(null)

  useEffect(() => {
    if (!containerRef.current) return

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height,
      layout: {
        background: { type: ColorType.Solid, color: '#0f1a26' },
        textColor: '#7a9ab8',
      },
      grid: {
        vertLines: { color: 'rgba(32, 58, 90, 0.3)' },
        horzLines: { color: 'rgba(32, 58, 90, 0.3)' },
      },
      crosshair: {
        mode: 1,
        vertLine: { color: 'rgba(45, 156, 219, 0.5)', width: 1, style: 3 },
        horzLine: { color: 'rgba(45, 156, 219, 0.5)', width: 1, style: 3 },
      },
      timeScale: {
        borderColor: 'rgba(32, 58, 90, 0.5)',
        timeVisible: true,
        secondsVisible: false,
      },
      rightPriceScale: {
        borderColor: 'rgba(32, 58, 90, 0.5)',
      },
    })

    chartRef.current = chart

    // Candlestick series
    const candleSeries = chart.addCandlestickSeries({
      upColor: '#00e676',
      downColor: '#ff4757',
      borderUpColor: '#00e676',
      borderDownColor: '#ff4757',
      wickUpColor: '#00e676',
      wickDownColor: '#ff4757',
    })
    seriesRef.current = candleSeries

    if (data?.length) {
      candleSeries.setData(data)
    }

    // Resize handler
    const handleResize = () => {
      if (containerRef.current) {
        chart.applyOptions({ width: containerRef.current.clientWidth })
      }
    }
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      chart.remove()
    }
  }, [height])

  // Update data
  useEffect(() => {
    if (seriesRef.current && data?.length) {
      seriesRef.current.setData(data)
    }
  }, [data])

  return (
    <div className="card" style={{ padding: 0 }}>
      {title && (
        <div style={{
          padding: '12px 16px',
          borderBottom: '1px solid var(--border)',
          fontSize: '0.72rem',
          fontWeight: 700,
          letterSpacing: '0.1em',
          color: 'var(--text-muted)',
          textTransform: 'uppercase',
        }}>
          {title}
        </div>
      )}
      <div ref={containerRef} style={{ height, borderRadius: '0 0 16px 16px', overflow: 'hidden' }} />
    </div>
  )
}

// ── Equity Curve (Area chart) ─────────────────────────────────────────────────
export function EquityCurveChart({ data, height = 200 }) {
  const containerRef = useRef(null)
  const chartRef = useRef(null)

  useEffect(() => {
    if (!containerRef.current) return

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height,
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#7a9ab8',
      },
      grid: {
        vertLines: { color: 'rgba(32, 58, 90, 0.2)' },
        horzLines: { color: 'rgba(32, 58, 90, 0.2)' },
      },
      crosshair: { mode: 1 },
      timeScale: { borderColor: 'rgba(32, 58, 90, 0.3)' },
      rightPriceScale: { borderColor: 'rgba(32, 58, 90, 0.3)' },
      handleScroll: false,
      handleScale: false,
    })

    chartRef.current = chart

    const areaSeries = chart.addAreaSeries({
      lineColor: '#2d9cdb',
      topColor: 'rgba(45, 156, 219, 0.25)',
      bottomColor: 'rgba(45, 156, 219, 0.0)',
      lineWidth: 2,
    })

    if (data?.length) {
      areaSeries.setData(data.map(d => ({
        time: d.time,
        value: d.equity,
      })))
    }

    const handleResize = () => {
      if (containerRef.current) {
        chart.applyOptions({ width: containerRef.current.clientWidth })
      }
    }
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      chart.remove()
    }
  }, [data, height])

  return <div ref={containerRef} style={{ height }} />
}
