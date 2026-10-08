import { useState, useEffect } from 'react'
import './index.css'
import { useWebSocket } from './hooks/useWebSocket'
import { useStore } from './store'
import DashboardPage from './components/DashboardPage'

function App() {
  const { page, setPage, wsConnected } = useStore()
  useWebSocket()

  const navItems = [
    { id: 'dashboard', label: 'Dashboard' },
    { id: 'trades',    label: 'Trades' },
    { id: 'analysis',  label: 'Analysis' },
    { id: 'rl',        label: 'RL Policy' },
    { id: 'knowledge', label: 'TA Knowledge' },
  ]

  return (
    <div className="app">
      {/* ── Top Bar ───────────────────────────────────────────────────── */}
      <header className="topbar">
        <div className="topbar-logo">
          <span className="logo-z">Z</span>
          <span className="logo-ecro">ECRO</span>
          <span className="logo-rl">RL</span>
        </div>

        <nav className="topbar-nav">
          {navItems.map(item => (
            <button
              key={item.id}
              className={`nav-btn ${page === item.id ? 'active' : ''}`}
              onClick={() => setPage(item.id)}
            >
              {item.label}
            </button>
          ))}
        </nav>

        <div className="topbar-status">
          <div className={`status-pill paper`}>
            <div className={`status-pulse`} />
            PAPER MODE
          </div>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            fontSize: '0.72rem',
            color: wsConnected ? 'var(--accent-green)' : 'var(--text-muted)',
          }}>
            <div style={{
              width: 7, height: 7,
              borderRadius: '50%',
              background: wsConnected ? 'var(--accent-green)' : 'var(--text-muted)',
            }} />
            {wsConnected ? 'LIVE' : 'OFFLINE'}
          </div>
        </div>
      </header>

      {/* ── Page Content ──────────────────────────────────────────────── */}
      <main className="main-content">
        {page === 'dashboard' && <DashboardPage />}

        {page === 'trades' && (
          <TradesPage />
        )}

        {page === 'analysis' && (
          <AnalysisPage />
        )}

        {page === 'rl' && (
          <RLPage />
        )}

        {page === 'knowledge' && (
          <KnowledgePage />
        )}
      </main>
    </div>
  )
}

// ── Placeholder pages ──────────────────────────────────────────────────────────

function ComingSoonCard({ title, description }) {
  return (
    <div className="card fade-in" style={{ textAlign: 'center', padding: 60 }}>
      <div style={{
        fontSize: '3rem',
        marginBottom: 16,
        filter: 'drop-shadow(0 0 20px rgba(45,156,219,0.4))',
      }}>⚡</div>
      <div style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: 8 }}>
        {title}
      </div>
      <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', maxWidth: 400, margin: '0 auto' }}>
        {description}
      </div>
    </div>
  )
}

function TradesPage() {
  const { tradeHistory, fetchTradeHistory, openTrades, fetchOpenTrades } = useStore()
  useEffect(() => {
    fetchTradeHistory(100)
    fetchOpenTrades()
  }, [])

  return (
    <div className="fade-in">
      <div className="section-title">Open Positions</div>
      <div style={{ marginBottom: 16 }}>
        {openTrades?.length === 0
          ? <div className="card" style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>No open positions</div>
          : openTrades?.map(t => (
            <div key={t.trade_id} className="card" style={{ marginBottom: 8, padding: '12px 16px' }}>
              <span style={{ fontWeight: 700 }}>{t.symbol}</span>{' '}
              <span className={`badge ${t.action?.toLowerCase()}`}>{t.action}</span>{' '}
              Entry: ${t.entry_price?.toLocaleString()} | SL: ${t.stop_loss?.toLocaleString()}
            </div>
          ))
        }
      </div>
      <div className="section-title">Trade History</div>
      <div className="card">
        {tradeHistory?.length > 0 ? (
          <div style={{ overflowX: 'auto' }}>
            <table className="trade-table">
              <thead><tr>
                <th>Symbol</th><th>Dir</th><th>Entry</th><th>Exit</th>
                <th>Reason</th><th>R</th><th>P&L</th><th>Policy</th>
              </tr></thead>
              <tbody>
                {tradeHistory.map(t => (
                  <tr key={t.trade_id}>
                    <td style={{ fontWeight: 700, color: 'var(--text-primary)' }}>{t.symbol}</td>
                    <td><span className={`badge ${t.action?.toLowerCase()}`}>{t.action}</span></td>
                    <td className="mono">${t.entry_price?.toLocaleString()}</td>
                    <td className="mono">{t.exit_price ? `$${t.exit_price.toLocaleString()}` : '—'}</td>
                    <td style={{ color: t.exit_reason === 'SL' ? 'var(--short)' : 'var(--long)', fontWeight: 600 }}>
                      {t.exit_reason || 'OPEN'}
                    </td>
                    <td className={`mono ${t.realized_r >= 0 ? 'positive-cell' : 'negative-cell'}`}>
                      {t.realized_r != null ? `${t.realized_r >= 0 ? '+' : ''}${t.realized_r?.toFixed(2)}R` : '—'}
                    </td>
                    <td className={t.net_pnl_inr >= 0 ? 'positive-cell' : 'negative-cell'}>
                      {t.net_pnl_inr != null ? `₹${Math.abs(t.net_pnl_inr).toLocaleString('en-IN')}` : '—'}
                    </td>
                    <td style={{ color: 'var(--text-muted)', fontSize: '0.7rem' }}>{t.policy_version || '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>No trade history yet.</div>
        )}
      </div>
    </div>
  )
}

function AnalysisPage() {
  const { lastAnalysis } = useStore()
  return (
    <div className="fade-in">
      <div className="section-title">1H Market Analysis</div>
      {lastAnalysis ? (
        <div className="card">
          <pre style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', whiteSpace: 'pre-wrap' }}>
            {JSON.stringify(lastAnalysis, null, 2)}
          </pre>
        </div>
      ) : (
        <ComingSoonCard
          title="Analysis Engine"
          description="The 1H + 15M Groq analysis results will appear here in real-time as the agent processes market data."
        />
      )}
    </div>
  )
}

function RLPage() {
  const { policies, activePolicy, experienceStats, fetchRLData } = useStore()
  useEffect(() => { fetchRLData() }, [])

  return (
    <div className="fade-in">
      <div className="section-title">RL Policy Versions</div>
      {policies?.length > 0 ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {policies.map(p => (
            <div key={p.version} className="card" style={{ padding: '12px 16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontWeight: 700, color: p.is_active ? 'var(--accent-cyan)' : 'var(--text-primary)' }}>
                  {p.version} {p.is_active && '(Active)'}
                </span>
                <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                  {p.training_episodes?.toLocaleString()} episodes
                </span>
              </div>
              <div style={{ display: 'flex', gap: 16, marginTop: 8, flexWrap: 'wrap' }}>
                {p.backtest_sharpe && (
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                    Sharpe: <span style={{ color: 'var(--text-value)' }}>{p.backtest_sharpe.toFixed(2)}</span>
                  </span>
                )}
                {p.backtest_win_rate && (
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                    Win: <span style={{ color: 'var(--long)' }}>{(p.backtest_win_rate * 100).toFixed(1)}%</span>
                  </span>
                )}
                {p.backtest_max_dd && (
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                    DD: <span style={{ color: 'var(--short)' }}>-{(p.backtest_max_dd * 100).toFixed(1)}%</span>
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      ) : (
        <ComingSoonCard
          title="RL Training"
          description="Policy versions appear here once PPO training begins. Run `python scripts/train_rl.py` to start training."
        />
      )}

      {experienceStats && (
        <>
          <div className="section-title" style={{ marginTop: 16 }}>Experience Buffer</div>
          <div className="grid-4">
            <div className="metric-card">
              <div className="metric-label">Total Experiences</div>
              <div className="metric-value">{experienceStats.total_experiences?.toLocaleString()}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Avg Reward</div>
              <div className={`metric-value ${experienceStats.avg_reward >= 0 ? 'positive' : 'negative'}`}>
                {experienceStats.avg_reward?.toFixed(3)}
              </div>
            </div>
            <div className="metric-card positive">
              <div className="metric-label">Max Reward</div>
              <div className="metric-value positive">{experienceStats.max_reward?.toFixed(3)}</div>
            </div>
            <div className="metric-card negative">
              <div className="metric-label">Min Reward</div>
              <div className="metric-value negative">{experienceStats.min_reward?.toFixed(3)}</div>
            </div>
          </div>
        </>
      )}
    </div>
  )
}

function KnowledgePage() {
  return (
    <div className="fade-in">
      <ComingSoonCard
        title="TA Knowledge Base"
        description="Upload your TA books (PDFs) using the backend script: python scripts/ingest_books.py --path /books. Then query the knowledge base here."
      />
    </div>
  )
}

export default App
