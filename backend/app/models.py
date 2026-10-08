"""
ZECRO-RL — SQLAlchemy ORM Models
Complete database schema for all ZECRO-RL tables.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON, BigInteger, Boolean, DateTime, Enum, Float,
    ForeignKey, Index, Integer, String, Text, UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


# ── Enums ──────────────────────────────────────────────────────────────────────

class TimeframeEnum(str, enum.Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"


class TrendDirection(str, enum.Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class MarketRegimeEnum(str, enum.Enum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    BREAKOUT = "BREAKOUT"
    BREAKDOWN = "BREAKDOWN"


class BiasEnum(str, enum.Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class ActionEnum(str, enum.Enum):
    HOLD = "HOLD"
    LONG = "LONG"
    SHORT = "SHORT"
    EXIT = "EXIT"


class TradeResultEnum(str, enum.Enum):
    SL = "SL"
    TP1 = "TP1"
    TP2 = "TP2"
    TSL = "TSL"
    MANUAL = "MANUAL"
    TIMEOUT = "TIMEOUT"
    OPEN = "OPEN"


# ── Market Candles (TimescaleDB hypertable) ────────────────────────────────────

class MarketCandle(Base):
    __tablename__ = "market_candles"
    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "timestamp", name="uq_candle"),
        Index("ix_candle_symbol_tf_ts", "symbol", "timeframe", "timestamp"),
        {"timescaledb_hypertable": {"time_column_name": "timestamp"}},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(5), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    funding_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    open_interest: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# ── Technical Features (TimescaleDB hypertable) ────────────────────────────────

class TechnicalFeature(Base):
    __tablename__ = "technical_features"
    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "timestamp", name="uq_feature"),
        Index("ix_feature_symbol_tf_ts", "symbol", "timeframe", "timestamp"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(5), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Market Structure
    trend_direction: Mapped[Optional[str]] = mapped_column(String(10))
    bos_bullish: Mapped[bool] = mapped_column(Boolean, default=False)
    bos_bearish: Mapped[bool] = mapped_column(Boolean, default=False)
    choch_bullish: Mapped[bool] = mapped_column(Boolean, default=False)
    choch_bearish: Mapped[bool] = mapped_column(Boolean, default=False)
    higher_high: Mapped[bool] = mapped_column(Boolean, default=False)
    higher_low: Mapped[bool] = mapped_column(Boolean, default=False)
    lower_high: Mapped[bool] = mapped_column(Boolean, default=False)
    lower_low: Mapped[bool] = mapped_column(Boolean, default=False)
    in_range: Mapped[bool] = mapped_column(Boolean, default=False)
    breakout: Mapped[bool] = mapped_column(Boolean, default=False)
    breakdown: Mapped[bool] = mapped_column(Boolean, default=False)
    swing_high: Mapped[Optional[float]] = mapped_column(Float)
    swing_low: Mapped[Optional[float]] = mapped_column(Float)

    # Price Action
    candle_type: Mapped[Optional[str]] = mapped_column(String(30))
    engulfing_bull: Mapped[bool] = mapped_column(Boolean, default=False)
    engulfing_bear: Mapped[bool] = mapped_column(Boolean, default=False)
    pin_bar_bull: Mapped[bool] = mapped_column(Boolean, default=False)
    pin_bar_bear: Mapped[bool] = mapped_column(Boolean, default=False)
    inside_bar: Mapped[bool] = mapped_column(Boolean, default=False)
    doji: Mapped[bool] = mapped_column(Boolean, default=False)
    rejection_bull: Mapped[bool] = mapped_column(Boolean, default=False)
    rejection_bear: Mapped[bool] = mapped_column(Boolean, default=False)
    momentum_bull: Mapped[bool] = mapped_column(Boolean, default=False)
    momentum_bear: Mapped[bool] = mapped_column(Boolean, default=False)
    candle_body_pct: Mapped[Optional[float]] = mapped_column(Float)
    upper_wick_pct: Mapped[Optional[float]] = mapped_column(Float)
    lower_wick_pct: Mapped[Optional[float]] = mapped_column(Float)

    # Liquidity
    prev_high_swept: Mapped[bool] = mapped_column(Boolean, default=False)
    prev_low_swept: Mapped[bool] = mapped_column(Boolean, default=False)
    equal_highs: Mapped[bool] = mapped_column(Boolean, default=False)
    equal_lows: Mapped[bool] = mapped_column(Boolean, default=False)
    ssl_level: Mapped[Optional[float]] = mapped_column(Float)  # sell-side liq
    bsl_level: Mapped[Optional[float]] = mapped_column(Float)  # buy-side liq
    dist_to_ssl_pct: Mapped[Optional[float]] = mapped_column(Float)
    dist_to_bsl_pct: Mapped[Optional[float]] = mapped_column(Float)
    liquidity_grab_bull: Mapped[bool] = mapped_column(Boolean, default=False)
    liquidity_grab_bear: Mapped[bool] = mapped_column(Boolean, default=False)

    # Momentum indicators
    ema20: Mapped[Optional[float]] = mapped_column(Float)
    ema50: Mapped[Optional[float]] = mapped_column(Float)
    ema200: Mapped[Optional[float]] = mapped_column(Float)
    ema20_slope: Mapped[Optional[float]] = mapped_column(Float)
    ema50_slope: Mapped[Optional[float]] = mapped_column(Float)
    price_vs_ema20: Mapped[Optional[float]] = mapped_column(Float)
    price_vs_ema50: Mapped[Optional[float]] = mapped_column(Float)
    price_vs_ema200: Mapped[Optional[float]] = mapped_column(Float)
    rsi: Mapped[Optional[float]] = mapped_column(Float)
    rsi_divergence: Mapped[Optional[int]] = mapped_column(Integer)  # -1/0/1
    macd: Mapped[Optional[float]] = mapped_column(Float)
    macd_signal: Mapped[Optional[float]] = mapped_column(Float)
    macd_hist: Mapped[Optional[float]] = mapped_column(Float)
    macd_cross: Mapped[Optional[int]] = mapped_column(Integer)  # -1/0/1
    adx: Mapped[Optional[float]] = mapped_column(Float)
    di_plus: Mapped[Optional[float]] = mapped_column(Float)
    di_minus: Mapped[Optional[float]] = mapped_column(Float)

    # Volatility
    atr: Mapped[Optional[float]] = mapped_column(Float)
    atr_normalized: Mapped[Optional[float]] = mapped_column(Float)
    atr_percentile: Mapped[Optional[float]] = mapped_column(Float)
    bb_upper: Mapped[Optional[float]] = mapped_column(Float)
    bb_lower: Mapped[Optional[float]] = mapped_column(Float)
    bb_width: Mapped[Optional[float]] = mapped_column(Float)
    bb_position: Mapped[Optional[float]] = mapped_column(Float)  # 0-1
    range_expansion: Mapped[bool] = mapped_column(Boolean, default=False)
    range_compression: Mapped[bool] = mapped_column(Boolean, default=False)

    # Volume
    volume_sma20: Mapped[Optional[float]] = mapped_column(Float)
    relative_volume: Mapped[Optional[float]] = mapped_column(Float)
    volume_spike: Mapped[bool] = mapped_column(Boolean, default=False)
    volume_divergence_bull: Mapped[bool] = mapped_column(Boolean, default=False)
    volume_divergence_bear: Mapped[bool] = mapped_column(Boolean, default=False)
    obv: Mapped[Optional[float]] = mapped_column(Float)
    obv_slope: Mapped[Optional[float]] = mapped_column(Float)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# ── Market Regime ──────────────────────────────────────────────────────────────

class MarketRegime(Base):
    __tablename__ = "market_regimes"
    __table_args__ = (
        Index("ix_regime_symbol_ts", "symbol", "timestamp"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    regime: Mapped[str] = mapped_column(String(20), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    regime_data: Mapped[Optional[Dict]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# ── Trade Setup ────────────────────────────────────────────────────────────────

class TradeSetup(Base):
    __tablename__ = "trade_setups"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True,
        default=lambda: str(uuid.uuid4())
    )
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    timeframe_analysis: Mapped[str] = mapped_column(String(5), nullable=False)
    timeframe_entry: Mapped[str] = mapped_column(String(5), nullable=False)
    setup_type: Mapped[Optional[str]] = mapped_column(String(50))
    bias: Mapped[str] = mapped_column(String(10), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    regime: Mapped[Optional[str]] = mapped_column(String(20))
    state_1h: Mapped[Optional[Dict]] = mapped_column(JSONB)
    state_15m: Mapped[Optional[Dict]] = mapped_column(JSONB)
    llm_analysis: Mapped[Optional[Dict]] = mapped_column(JSONB)
    ta_knowledge_used: Mapped[Optional[List]] = mapped_column(JSONB)
    invalidation_condition: Mapped[Optional[str]] = mapped_column(Text)
    is_valid: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    executions: Mapped[List["TradeExecution"]] = relationship(
        "TradeExecution", back_populates="setup"
    )


# ── Trade Execution ────────────────────────────────────────────────────────────

class TradeExecution(Base):
    __tablename__ = "trade_executions"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True,
        default=lambda: str(uuid.uuid4())
    )
    setup_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False), ForeignKey("trade_setups.id"), nullable=True
    )
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    action: Mapped[str] = mapped_column(String(10), nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    stop_loss: Mapped[float] = mapped_column(Float, nullable=False)
    tp1: Mapped[float] = mapped_column(Float, nullable=False)
    tp2: Mapped[Optional[float]] = mapped_column(Float)
    trailing_stop_pct: Mapped[float] = mapped_column(Float, default=0.5)
    position_size: Mapped[float] = mapped_column(Float, nullable=False)
    position_size_inr: Mapped[float] = mapped_column(Float, nullable=False)
    risk_amount_inr: Mapped[float] = mapped_column(Float, nullable=False)
    risk_r: Mapped[float] = mapped_column(Float, default=1.0)
    leverage: Mapped[int] = mapped_column(Integer, default=1)
    entry_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    exit_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    exit_price: Mapped[Optional[float]] = mapped_column(Float)
    exit_reason: Mapped[Optional[str]] = mapped_column(String(20))

    # Partial TP tracking
    tp1_filled: Mapped[bool] = mapped_column(Boolean, default=False)
    tp1_fill_price: Mapped[Optional[float]] = mapped_column(Float)
    tp1_fill_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    tp1_quantity: Mapped[Optional[float]] = mapped_column(Float)
    tp2_filled: Mapped[bool] = mapped_column(Boolean, default=False)
    tp2_fill_price: Mapped[Optional[float]] = mapped_column(Float)
    tp2_fill_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    tp2_quantity: Mapped[Optional[float]] = mapped_column(Float)
    tsl_active: Mapped[bool] = mapped_column(Boolean, default=False)
    tsl_price: Mapped[Optional[float]] = mapped_column(Float)
    sl_moved_to_be: Mapped[bool] = mapped_column(Boolean, default=False)

    # P&L
    realized_pnl_inr: Mapped[Optional[float]] = mapped_column(Float)
    realized_r: Mapped[Optional[float]] = mapped_column(Float)
    fees_inr: Mapped[float] = mapped_column(Float, default=0.0)
    slippage_inr: Mapped[float] = mapped_column(Float, default=0.0)
    net_pnl_inr: Mapped[Optional[float]] = mapped_column(Float)

    # Meta
    policy_version: Mapped[Optional[str]] = mapped_column(String(20))
    market_regime: Mapped[Optional[str]] = mapped_column(String(20))
    is_paper: Mapped[bool] = mapped_column(Boolean, default=True)
    is_open: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    setup: Mapped[Optional[TradeSetup]] = relationship(
        "TradeSetup", back_populates="executions"
    )
    reward: Mapped[Optional["TradeReward"]] = relationship(
        "TradeReward", back_populates="execution", uselist=False
    )


# ── Trade Reward ───────────────────────────────────────────────────────────────

class TradeReward(Base):
    __tablename__ = "trade_rewards"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    trade_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("trade_executions.id"), nullable=False, unique=True
    )
    realized_r: Mapped[float] = mapped_column(Float, default=0.0)
    unrealized_r_peak: Mapped[float] = mapped_column(Float, default=0.0)
    transaction_cost_r: Mapped[float] = mapped_column(Float, default=0.0)
    slippage_r: Mapped[float] = mapped_column(Float, default=0.0)
    drawdown_penalty: Mapped[float] = mapped_column(Float, default=0.0)
    overtrading_penalty: Mapped[float] = mapped_column(Float, default=0.0)
    invalid_entry_penalty: Mapped[float] = mapped_column(Float, default=0.0)
    final_reward: Mapped[float] = mapped_column(Float, nullable=False)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    execution: Mapped[TradeExecution] = relationship(
        "TradeExecution", back_populates="reward"
    )


# ── Agent Experience ───────────────────────────────────────────────────────────

class AgentExperience(Base):
    __tablename__ = "agent_experiences"
    __table_args__ = (
        Index("ix_exp_policy_ts", "policy_version", "timestamp"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    episode_id: Mapped[str] = mapped_column(String(50), nullable=False)
    step: Mapped[int] = mapped_column(Integer, nullable=False)
    state: Mapped[List[float]] = mapped_column(JSON, nullable=False)
    action: Mapped[int] = mapped_column(Integer, nullable=False)
    reward: Mapped[float] = mapped_column(Float, nullable=False)
    next_state: Mapped[List[float]] = mapped_column(JSON, nullable=False)
    done: Mapped[bool] = mapped_column(Boolean, default=False)
    policy_version: Mapped[Optional[str]] = mapped_column(String(20))
    symbol: Mapped[Optional[str]] = mapped_column(String(20))
    timestamp: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# ── Policy Version ─────────────────────────────────────────────────────────────

class PolicyVersion(Base):
    __tablename__ = "policy_versions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    version_name: Mapped[str] = mapped_column(String(20), nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    model_path: Mapped[Optional[str]] = mapped_column(String(255))
    training_start: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    training_end: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    training_episodes: Mapped[int] = mapped_column(Integer, default=0)
    hyperparams: Mapped[Optional[Dict]] = mapped_column(JSONB)
    # Backtest metrics
    backtest_sharpe: Mapped[Optional[float]] = mapped_column(Float)
    backtest_max_dd: Mapped[Optional[float]] = mapped_column(Float)
    backtest_win_rate: Mapped[Optional[float]] = mapped_column(Float)
    backtest_profit_factor: Mapped[Optional[float]] = mapped_column(Float)
    backtest_total_trades: Mapped[int] = mapped_column(Integer, default=0)
    # Walk-forward metrics
    wf_sharpe: Mapped[Optional[float]] = mapped_column(Float)
    wf_max_dd: Mapped[Optional[float]] = mapped_column(Float)
    wf_win_rate: Mapped[Optional[float]] = mapped_column(Float)
    # Paper metrics (live)
    paper_sharpe: Mapped[Optional[float]] = mapped_column(Float)
    paper_win_rate: Mapped[Optional[float]] = mapped_column(Float)
    paper_total_trades: Mapped[int] = mapped_column(Integer, default=0)
    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    is_promoted: Mapped[bool] = mapped_column(Boolean, default=False)
    notes: Mapped[Optional[str]] = mapped_column(Text)


# ── Portfolio State ────────────────────────────────────────────────────────────

class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"
    __table_args__ = (
        Index("ix_portfolio_ts", "timestamp"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    equity_inr: Mapped[float] = mapped_column(Float, nullable=False)
    available_inr: Mapped[float] = mapped_column(Float, nullable=False)
    unrealized_pnl_inr: Mapped[float] = mapped_column(Float, default=0.0)
    realized_pnl_inr: Mapped[float] = mapped_column(Float, default=0.0)
    total_pnl_inr: Mapped[float] = mapped_column(Float, default=0.0)
    return_pct: Mapped[float] = mapped_column(Float, default=0.0)
    drawdown_pct: Mapped[float] = mapped_column(Float, default=0.0)
    peak_equity: Mapped[float] = mapped_column(Float, nullable=False)
    open_positions: Mapped[int] = mapped_column(Integer, default=0)
    daily_pnl_inr: Mapped[float] = mapped_column(Float, default=0.0)
    policy_version: Mapped[Optional[str]] = mapped_column(String(20))


# ── TA Knowledge ───────────────────────────────────────────────────────────────

class TAKnowledge(Base):
    __tablename__ = "ta_knowledge"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    concept: Mapped[str] = mapped_column(String(100), nullable=False)
    chunk_type: Mapped[str] = mapped_column(String(30), default="text")
    content: Mapped[str] = mapped_column(Text, nullable=False)
    conditions: Mapped[Optional[List]] = mapped_column(JSONB)
    confirmation: Mapped[Optional[List]] = mapped_column(JSONB)
    invalidation: Mapped[Optional[List]] = mapped_column(JSONB)
    related_concepts: Mapped[Optional[List]] = mapped_column(JSONB)
    source_book: Mapped[Optional[str]] = mapped_column(String(200))
    source_page: Mapped[Optional[int]] = mapped_column(Integer)
    embedding: Mapped[Optional[List[float]]] = mapped_column(Vector(384))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
