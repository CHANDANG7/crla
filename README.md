# ⚡ ZECRO-RL — Reinforcement Technical Analysis Trading Agent

**ZECRO-RL** is an autonomous cryptocurrency perpetual futures trading system built on a hybrid architecture:
- **Groq LLM (`llama-3.3-70b-versatile`)**: Structured Technical Analysis reasoning, market context, and RAG over technical literature.
- **PPO Reinforcement Learning Policy**: Action decision-making over a 160-dimensional deterministic TA feature vector.
- **Delta Exchange**: High-speed REST + WebSocket market data feeds.
- **Institutional Risk Gate**: Hard deterministic risk constraints (0.25% risk/trade, ₹75,000 daily loss, 10% max DD, ₹1,00,00,000 simulated capital).
- **React + Vite + TradingView Lightweight Charts**: Real-time glassmorphic trading dashboard with live WebSocket feeds.

---

## 🏛️ System Architecture

```
┌────────────────────────────────────────────────────────┐
│                   DELTA EXCHANGE                      │
│            REST (OHLCV) + WebSocket (Live)            │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│               DETERMINISTIC TA ENGINE                  │
│  - Market Structure (HH/HL, BOS, CHoCH, Swings)        │
│  - Price Action (Pin bars, Engulfing, Rejections)      │
│  - Liquidity (BSL/SSL Sweeps, Equal Highs/Lows)        │
│  - Momentum (EMAs 20/50/200, RSI, MACD, ADX)           │
│  - Volatility (ATR Percentile, Bollinger Bands)        │
│  - Volume (RVOL, OBV, Spikes, Divergences)             │
│        ▼                                               │
│  State Vector: EXACTLY 160 DIMENSIONS                  │
└────────────┬─────────────────────────────┬─────────────┘
             │                             │
             ▼                             ▼
┌─────────────────────────┐   ┌──────────────────────────┐
│     GROQ LLM (RAG)      │   │     PPO RL POLICY        │
│  TA Reasoning & Context │   │  Input (160) → Dense(256)│
│  Structured JSON Output │   │   → Dense(128) → (64)    │
│  (Books RAG via pgvector)│  │   → Policy (4) + Value(1)│
└────────────┬────────────┘   └────────────┬─────────────┘
             │                             │
             └──────────────┬──────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│             HARD RISK GATE (IMMUTABLE)                 │
│  - Max Risk per Trade: 0.25% of Equity (₹25,000)       │
│  - Max Daily Loss: ₹75,000                             │
│  - Max Portfolio Drawdown: 10%                         │
│  - Max Open Positions: 3                               │
│  - Minimum Risk:Reward: 1:1                            │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│              PAPER TRADING ENGINE                      │
│  - Partial TP1 (50% closed, SL moved to Breakeven)     │
│  - Partial TP2 (25% closed, Trailing Stop activated)   │
│  - Runner (25% closed via Trailing Stop Loss)          │
│  - Taker fee (0.05%) & Slippage model                  │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│              FASTAPI + REACT DASHBOARD                 │
│  - Real-time WebSockets, TradingView Candlesticks,     │
│    Live Equity Curve, RL Metrics, Position Manager     │
└────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Prerequisites
- **Docker & Docker Compose**
- **Python 3.10+** (PyTorch, FastAPI)
- **Node.js 18+ & npm** (React frontend)
- **Groq API Key** (for TA reasoning layer)
- **Delta Exchange API Key** (optional for public feeds, required for private orders)

### 2. Environment Setup
Copy `.env.example` to `.env` and fill in your API credentials:
```bash
cp .env.example .env
```

### 3. Start Database & Infrastructure
```bash
docker-compose up -d timescaledb redis
```

### 4. Ingest TA Books into pgvector (RAG)
Place your Technical Analysis PDF books in the `./books/` folder and run:
```bash
cd backend
python scripts/ingest_books.py --path ../books
```

### 5. Bootstrap Historical Market Data
Download historical 1H and 15M candles from Delta Exchange:
```bash
python scripts/download_history.py --symbol BTCUSD --days 60
# Or download all configured pairs:
python scripts/download_history.py --all --days 90
```

### 6. Train the PPO Policy
Train the reinforcement learning policy on historical market data using walk-forward validation:
```bash
python scripts/train_rl.py --symbol BTCUSD --episodes 100 --auto-promote
```

### 7. Launch the Backend
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation will be live at `http://localhost:8000/docs`.

### 8. Launch the Frontend Dashboard
```bash
cd ../frontend
npm install
npm run dev
```
Open `http://localhost:5173` to view the interactive dashboard.

---

## 📊 Dashboard Overview

The React + Vite frontend features:
- **Header Overview**: Current Equity (₹1,00,00,000 base), Total P&L, Daily P&L, Return %, Drawdown %.
- **TradingView Chart**: 1H & 15M candlesticks with dynamic overlay.
- **Performance Metrics**: Sharpe Ratio, Sortino Ratio, Win Rate, Profit Factor, Expectancy.
- **Live Positions**: Real-time trade monitor with partial TP indicators and breakeven stop status.
- **RL Panel**: Active policy version, episode count, experience replay stats.
- **Risk Gate Monitor**: Real-time limit meters for Daily Loss and Drawdown.
- **Trade History Table**: Full log of historical paper trades with R-multiples.

---

## 🛡️ Risk Management Rules (Hardcoded)

The Risk Manager rules are **hard-coded in pure Python** and cannot be bypassed by Groq LLM or the RL Agent:
1. **Position Sizing**: Maximum 0.25% of current equity at risk per trade ($R \le ₹25,000$ initially).
2. **Daily Loss Gate**: Trading automatically halts if daily loss reaches ₹75,000.
3. **Max Drawdown Gate**: System locks if peak-to-trough drawdown reaches 10%.
4. **Position Concurrency**: At most 3 concurrent open positions across all symbols.
5. **Mandatory Stop Loss**: Every trade must define an analytical stop loss with $RR \ge 1.0$.

---

## 🧪 RL Formulation

- **State Space**: Exactly 160 continuous normalized features:
  - 1H Market Structure & Swing Points (20)
  - 1H Momentum & Moving Averages (25)
  - 1H Volatility & Volume (20)
  - 15M Entry Setup & Liquidity Sweeps (35)
  - Multi-Timeframe Alignment & Regimes (20)
  - Position State & Risk Parameters (20)
  - Time & Market Session Encodings (20)
- **Action Space**: Discrete(4): `0=HOLD`, `1=LONG`, `2=SHORT`, `3=EXIT`.
- **Reward Function**: $R = \text{Realized\_R} + 0.1 \cdot \text{Peak\_R} - \text{Cost\_R} - \text{Drawdown\_Penalty} - \text{Overtrading\_Penalty}$ (clipped to $[-3.0, +3.0]$).

---

## 📜 License
MIT License. Built for autonomous research and paper trading.
