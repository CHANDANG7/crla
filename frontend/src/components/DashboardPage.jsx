import { useEffect, useRef } from 'react'
import { useStore } from '../store'
import { formatINR, formatR, formatPct, pnlClass } from '../hooks/useWebSocket'

function MetricCard({ label, value, sub, className = '', valueClass = '' }) {
  return (
    <div className={`metric-card ${className}`}>
      <div className="metric-label">{label}</div>
      <div className={`metric-value ${valueClass}`}>{value}</div>
      {sub && <div className="metric-sub">{sub}</div>}
    </div>
  )
}

function Badge({ type }) {
  const cls = type?.toLowerCase()
  return <span className={`badge ${cls}`}>{type}</span>
}

function RegimeBadge({ regime }) {
  if (!regime) return null
  return (
    <span className={`regime-badge regime-${regime}`}>
      {regime?.replace(/_/g, ' ')}
    </span>
  )
}

function EquityHeader({ overview }) {
  const p = overview?.portfolio || {}
  const pnlPositive = (p.total_pnl_inr || 0) >= 0
  return (
    <div className="equity-header fade-in">
      <div className="equity-main">
        <div className="equity-label">Paper Capital — ₹1 Crore Account</div>
        <div className="equity-amount">
          <span className="currency">₹</span>
          {(p.current_equity_inr || 10000000).toLocaleString('en-IN')}
        </div>
      </div>
      <div className="equity-stats">
        <div className="equity-stat">
          <div className="stat-label">Total P&L</div>
          <div className={`stat-value ${pnlClass(p.total_pnl_inr)}`}>
            {formatINR(p.total_pnl_inr || 0)}
          </div>
        </div>
        <div className="equity-stat">
          <div className="stat-label">Return</div>
          <div className={`stat-value ${pnlClass(p.return_pct)}`}>
            {formatPct(p.return_pct || 0)}
          </div>
        </div>
        <div className="equity-stat">
          <div className="stat-label">Max DD</div>
          <div className="stat-value negative">
            -{formatPct(p.current_drawdown_pct || 0)}
          </div>
        </div>
        <div className="equity-stat">
          <div className="stat-label">Daily P&L</div>
          <div className={`stat-value ${pnlClass(p.daily_pnl_inr)}`}>
            {formatINR(p.daily_pnl_inr || 0, true)}
          </div>
        </div>
        <div className="equity-stat">
          <div className="stat-label">Available</div>
          <div className="stat-value neutral">
            {formatINR(p.available_inr || 0, true)}
          </div>
        </div>
      </div>
    </div>
  )
}

function OpenPositions({ trades }) {
  if (!trades?.length) {
    return (
      <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem', padding: '12px 0' }}>
        No open positions
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {trades.map(t => (
        <div key={t.trade_id} className={`position-card ${t.action?.toLowerCase()}-pos`}>
          <div className="position-header">
            <span className="position-symbol">{t.symbol}</span>
            <Badge type={t.action} />
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8, marginTop: 8 }}>
            <div>
              <div className="metric-label">Entry</div>
              <div className="metric-value" style={{ fontSize: '0.9rem' }}>
                ${t.entry_price?.toLocaleString()}
              </div>
            </div>
            <div>
              <div className="metric-label">Stop Loss</div>
              <div className="metric-value negative" style={{ fontSize: '0.9rem' }}>
                ${t.stop_loss?.toLocaleString()}
              </div>
            </div>
            <div>
              <div className="metric-label">Risk</div>
              <div className="metric-value" style={{ fontSize: '0.9rem' }}>
                {formatINR(t.risk_amount_inr, true)}
              </div>
            </div>
          </div>
          <div style={{ display: 'flex', gap: 6, marginTop: 8, flexWrap: 'wrap' }}>
            {t.tp1_filled && <span className="badge long" style={{ fontSize: '0.6rem' }}>TP1 ✓</span>}
            {t.tp2_filled && <span className="badge long" style={{ fontSize: '0.6rem' }}>TP2 ✓</span>}
            {t.tsl_active && <span className="badge neutral" style={{ fontSize: '0.6rem' }}>TSL Active</span>}
            {t.sl_moved_to_be && <span className="badge neutral" style={{ fontSize: '0.6rem' }}>BE</span>}
          </div>
        </div>
      ))}
    </div>
  )
}

function TradeHistoryTable({ trades }) {
  if (!trades?.length) {
    return <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem', padding: '12px 0' }}>No trade history</div>
  }

  return (
    <div style={{ overflowX: 'auto' }}>
      <table className="trade-table">
        <thead>
          <tr>
            <th>Symbol</th>
            <th>Dir</th>
            <th>Entry</th>
            <th>Exit</th>
            <th>Reason</th>
            <th>R</th>
            <th>Net P&L</th>
            <th>Regime</th>
            <th>Policy</th>
          </tr>
        </thead>
        <tbody>
          {trades.map(t => (
            <tr key={t.trade_id}>
              <td className="mono" style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{t.symbol}</td>
              <td><Badge type={t.action} /></td>
              <td className="mono">${t.entry_price?.toLocaleString()}</td>
              <td className="mono">{t.exit_price ? `$${t.exit_price.toLocaleString()}` : '—'}</td>
              <td>
                <span style={{
                  color: t.exit_reason === 'SL' ? 'var(--short)' :
                         t.exit_reason?.startsWith('TP') ? 'var(--long)' : 'var(--text-muted)',
                  fontWeight: 600, fontSize: '0.72rem',
                }}>
                  {t.exit_reason || 'OPEN'}
                </span>
              </td>
              <td className={`mono ${t.realized_r >= 0 ? 'positive-cell' : 'negative-cell'}`}>
                {t.realized_r != null ? formatR(t.realized_r) : '—'}
              </td>
              <td className={t.net_pnl_inr >= 0 ? 'positive-cell' : 'negative-cell'}>
                {formatINR(t.net_pnl_inr, true)}
              </td>
              <td><RegimeBadge regime={t.market_regime} /></td>
              <td style={{ color: 'var(--text-muted)', fontSize: '0.7rem' }}>
                {t.policy_version || '—'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function PerformanceMetrics({ metrics, overview }) {
  const perf = overview?.performance || {}
  return (
    <div className="grid-4">
      <MetricCard
        label="Win Rate"
        value={`${(perf.win_rate || 0).toFixed(1)}%`}
        sub={`${perf.wins || 0}W / ${perf.losses || 0}L`}
        className={perf.win_rate >= 50 ? 'positive' : 'negative'}
      />
      <MetricCard
        label="Profit Factor"
        value={(metrics?.profit_factor || 0).toFixed(2)}
        sub="Gross Profit / Gross Loss"
        className={metrics?.profit_factor >= 1.5 ? 'positive' : ''}
      />
      <MetricCard
        label="Sharpe Ratio"
        value={(metrics?.sharpe || 0).toFixed(2)}
        sub="Risk-adjusted return"
        className={metrics?.sharpe >= 1 ? 'positive' : ''}
      />
      <MetricCard
        label="Expectancy"
        value={metrics?.expectancy_r != null ? formatR(metrics.expectancy_r) : '—'}
        sub="Per trade avg R"
        valueClass={metrics?.expectancy_r >= 0 ? 'positive' : 'negative'}
      />
      <MetricCard
        label="Total Trades"
        value={perf.total_trades || 0}
        sub="Completed paper trades"
      />
      <MetricCard
        label="Avg Win"
        value={metrics?.avg_win_r != null ? formatR(metrics.avg_win_r) : '—'}
        sub="Average winning R"
        valueClass="positive"
      />
      <MetricCard
        label="Avg Loss"
        value={metrics?.avg_loss_r != null ? `-${formatR(metrics.avg_loss_r)}` : '—'}
        sub="Average losing R"
        valueClass="negative"
      />
      <MetricCard
        label="Total R"
        value={metrics?.total_r != null ? formatR(metrics.total_r) : '—'}
        sub="Cumulative R gained"
        valueClass={pnlClass(metrics?.total_r)}
        className={metrics?.total_r >= 0 ? 'positive' : 'negative'}
      />
    </div>
  )
}

function RLPanel({ activePolicy, expStats }) {
  return (
    <div className="card">
      <div className="card-title">RL Policy</div>
      <div className="rl-panel">
        <div className="rl-stat-row">
          <span className="rl-stat-label">Active Version</span>
          <span className="rl-stat-value" style={{ color: 'var(--accent-cyan)' }}>
            {activePolicy?.version || 'None'}
          </span>
        </div>
        <div className="rl-stat-row">
          <span className="rl-stat-label">Training Episodes</span>
          <span className="rl-stat-value">
            {(activePolicy?.training_episodes || 0).toLocaleString()}
          </span>
        </div>
        <div className="rl-stat-row">
          <span className="rl-stat-label">Experiences</span>
          <span className="rl-stat-value">
            {(expStats?.total_experiences || 0).toLocaleString()}
          </span>
        </div>
        <div className="rl-stat-row">
          <span className="rl-stat-label">Avg Reward</span>
          <span className={`rl-stat-value ${pnlClass(expStats?.avg_reward)}`}>
            {expStats?.avg_reward != null ? expStats.avg_reward.toFixed(3) : '—'}
          </span>
        </div>
        <div className="rl-stat-row">
          <span className="rl-stat-label">Backtest Sharpe</span>
          <span className="rl-stat-value">
            {activePolicy?.backtest_sharpe?.toFixed(2) || '—'}
          </span>
        </div>
        <div className="rl-stat-row">
          <span className="rl-stat-label">Backtest Win Rate</span>
          <span className="rl-stat-value">
            {activePolicy?.backtest_win_rate != null
              ? `${(activePolicy.backtest_win_rate * 100).toFixed(1)}%`
              : '—'}
          </span>
        </div>
        <div className="rl-stat-row">
          <span className="rl-stat-label">WF Sharpe</span>
          <span className="rl-stat-value">
            {activePolicy?.wf_sharpe?.toFixed(2) || '—'}
          </span>
        </div>
      </div>
    </div>
  )
}

function RiskPanel({ overview }) {
  const risk = overview?.risk || {}
  const portfolio = overview?.portfolio || {}

  const dailyLossUsedPct = Math.min(
    100,
    Math.abs(portfolio.daily_pnl_inr || 0) / (risk.max_daily_loss_inr || 75000) * 100
  )

  return (
    <div className="card">
      <div className="card-title">Risk Status</div>
      <div className="rl-panel">
        <div className="rl-stat-row">
          <span className="rl-stat-label">Risk / Trade</span>
          <span className="rl-stat-value">{risk.risk_per_trade_pct?.toFixed(2) || 0.25}%</span>
        </div>
        <div className="rl-stat-row">
          <span className="rl-stat-label">Risk Amount</span>
          <span className="rl-stat-value">{formatINR(risk.risk_per_trade_inr, true)}</span>
        </div>
        <div className="rl-stat-row">
          <span className="rl-stat-label">Open Positions</span>
          <span className="rl-stat-value">
            {risk.open_positions || 0} / {risk.max_positions || 3}
          </span>
        </div>
        <div className="rl-stat-row">
          <span className="rl-stat-label">Daily Loss Limit</span>
          <span className="rl-stat-value">{formatINR(risk.max_daily_loss_inr, true)}</span>
        </div>
        <div style={{ marginTop: 8 }}>
          <div className="metric-label" style={{ marginBottom: 4 }}>
            Daily Loss Used: {dailyLossUsedPct.toFixed(0)}%
          </div>
          <div className="confidence-bar">
            <div
              className="confidence-fill short"
              style={{ width: `${dailyLossUsedPct}%`, background: 'var(--grad-red)' }}
            />
          </div>
        </div>
        <div style={{ marginTop: 8 }}>
          <div className="metric-label" style={{ marginBottom: 4 }}>
            Drawdown: {portfolio.current_drawdown_pct?.toFixed(2) || 0}% / 10%
          </div>
          <div className="confidence-bar">
            <div
              className="confidence-fill short"
              style={{
                width: `${Math.min(100, (portfolio.current_drawdown_pct || 0) * 10)}%`,
                background: 'var(--grad-red)',
              }}
            />
          </div>
        </div>
      </div>
    </div>
  )
}

export default function DashboardPage() {
  const { overview, equityCurve, metrics, openTrades, activePolicy, experienceStats,
    fetchOverview, fetchEquityCurve, fetchMetrics, fetchOpenTrades, fetchRLData, fetchTradeHistory,
    tradeHistory } = useStore()

  useEffect(() => {
    fetchOverview()
    fetchEquityCurve(30)
    fetchMetrics()
    fetchOpenTrades()
    fetchRLData()
    fetchTradeHistory(20)

    // Refresh every 30s
    const interval = setInterval(() => {
      fetchOverview()
      fetchOpenTrades()
    }, 30000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="fade-in">
      <EquityHeader overview={overview} />

      {/* ── Performance Metrics ─────────────────────────────────────── */}
      <div className="section-title">Performance</div>
      <PerformanceMetrics metrics={metrics} overview={overview} />

      {/* ── Main Dashboard Grid ─────────────────────────────────────── */}
      <div className="dashboard-grid" style={{ marginTop: 16 }}>
        {/* Open Positions */}
        <div className="card">
          <div className="card-title">Open Positions</div>
          <OpenPositions trades={openTrades} />
        </div>

        {/* RL + Risk */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <RLPanel activePolicy={activePolicy} expStats={experienceStats} />
          <RiskPanel overview={overview} />
        </div>

        {/* Sidebar: Quick stats */}
        <div className="sidebar">
          <div className="card">
            <div className="card-title">Portfolio</div>
            <div className="rl-panel">
              <div className="rl-stat-row">
                <span className="rl-stat-label">Peak Equity</span>
                <span className="rl-stat-value">
                  {formatINR(overview?.portfolio?.peak_equity_inr, true)}
                </span>
              </div>
              <div className="rl-stat-row">
                <span className="rl-stat-label">Sortino</span>
                <span className="rl-stat-value">{metrics?.sortino?.toFixed(2) || '—'}</span>
              </div>
              <div className="rl-stat-row">
                <span className="rl-stat-label">Best Trade</span>
                <span className="rl-stat-value positive">
                  {metrics?.best_trade_r != null ? formatR(metrics.best_trade_r) : '—'}
                </span>
              </div>
              <div className="rl-stat-row">
                <span className="rl-stat-label">Worst Trade</span>
                <span className="rl-stat-value negative">
                  {metrics?.worst_trade_r != null ? formatR(metrics.worst_trade_r) : '—'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ── Recent Trades ───────────────────────────────────────────── */}
      <div style={{ marginTop: 16 }}>
        <div className="card">
          <div className="card-title">Recent Trades</div>
          <TradeHistoryTable trades={tradeHistory} />
        </div>
      </div>
    </div>
  )
}
