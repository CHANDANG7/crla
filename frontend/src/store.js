import { create } from 'zustand'
import api from './api'

export const useStore = create((set, get) => ({
  // ── Active page ──────────────────────────────────────────────────────────
  page: 'dashboard',
  setPage: (page) => set({ page }),

  // ── Portfolio ────────────────────────────────────────────────────────────
  overview: null,
  equityCurve: [],
  metrics: null,
  isLoadingPortfolio: false,

  fetchOverview: async () => {
    set({ isLoadingPortfolio: true })
    try {
      const data = await api.getOverview()
      set({ overview: data })
    } catch (e) {
      console.error('Overview fetch failed', e)
    } finally {
      set({ isLoadingPortfolio: false })
    }
  },

  fetchEquityCurve: async (days = 30) => {
    try {
      const data = await api.getEquityCurve(days)
      set({ equityCurve: data })
    } catch (e) {
      console.error('Equity curve fetch failed', e)
    }
  },

  fetchMetrics: async () => {
    try {
      const data = await api.getMetrics()
      set({ metrics: data })
    } catch (e) {
      console.error('Metrics fetch failed', e)
    }
  },

  // ── Trades ───────────────────────────────────────────────────────────────
  openTrades: [],
  tradeHistory: [],
  isLoadingTrades: false,

  fetchOpenTrades: async () => {
    try {
      const data = await api.getOpenTrades()
      set({ openTrades: data })
    } catch (e) {
      console.error('Open trades fetch failed', e)
    }
  },

  fetchTradeHistory: async (limit = 50) => {
    set({ isLoadingTrades: true })
    try {
      const data = await api.getTradeHistory(limit)
      set({ tradeHistory: data })
    } catch (e) {
      console.error('Trade history fetch failed', e)
    } finally {
      set({ isLoadingTrades: false })
    }
  },

  // ── RL ───────────────────────────────────────────────────────────────────
  policies: [],
  activePolicy: null,
  experienceStats: null,

  fetchRLData: async () => {
    try {
      const [policies, activePolicy, expStats] = await Promise.all([
        api.getPolicies(),
        api.getActivePolicy(),
        api.getExperienceStats(),
      ])
      set({ policies, activePolicy, experienceStats: expStats })
    } catch (e) {
      console.error('RL data fetch failed', e)
    }
  },

  // ── WebSocket state ───────────────────────────────────────────────────────
  wsConnected: false,
  lastAnalysis: null,
  lastTradeEvent: null,

  setWsConnected: (v) => set({ wsConnected: v }),
  setLastAnalysis: (data) => set({ lastAnalysis: data }),
  setLastTradeEvent: (data) => set({ lastTradeEvent: data }),
  updateOverviewFromWs: (data) =>
    set((state) => ({ overview: state.overview ? { ...state.overview, portfolio: data } : { portfolio: data } })),
}))
