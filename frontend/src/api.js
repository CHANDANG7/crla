import axios from 'axios'

const BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1'

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
}

export const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws'

export default api
