import axios from 'axios'

// Dynamically determine host and protocol for zero-config production deployments
const protocol = typeof window !== 'undefined' && window.location.protocol === 'https:' ? 'https:' : 'http:'
const wsProtocol = typeof window !== 'undefined' && window.location.protocol === 'https:' ? 'wss:' : 'ws:'
const host = typeof window !== 'undefined' ? window.location.host : 'localhost:8000'

const BASE = import.meta.env.VITE_API_URL || `${protocol}//${host}/api/v1`

const client = axios.create({ baseURL: BASE, timeout: 10000 })

const api = {
  // Dashboard
  getOverview:    () => client.get('/dashboard/overview').then(r => r.data),
  getEquityCurve: (days) => client.get(`/dashboard/equity-curve?days=${days}`).then(r => r.data),
  getMetrics:     () => client.get('/dashboard/metrics').then(r => r.data),

  // Trades
  getOpenTrades:   () => client.get('/trades/open').then(r => r.data),
  getTradeHistory: (limit) => client.get(`/trades/history?limit=${limit}`).then(r => r.data),
  getTrade:        (id) => client.get(`/trades/${id}`).then(r => r.data),

  // RL
  getPolicies:       () => client.get('/rl/policies').then(r => r.data),
  getActivePolicy:   () => client.get('/rl/active-policy').then(r => r.data),
  getExperienceStats:() => client.get('/rl/experience-stats').then(r => r.data),

  // Knowledge (RAG)
  getKnowledgeStats: () => client.get('/knowledge/stats').then(r => r.data),
  searchKnowledge:   (q) => client.get(`/knowledge/search?q=${encodeURIComponent(q)}`).then(r => r.data),
}

export const WS_URL = import.meta.env.VITE_WS_URL || `${wsProtocol}//${host}/ws`

export default api
