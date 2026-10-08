-- ZECRO-RL: Initial TimescaleDB & PostgreSQL Extensions Setup
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Hypertables will be automatically created once SQLAlchemy creates tables or through below DDL:
-- market_candles table
CREATE TABLE IF NOT EXISTS market_candles (
    id BIGSERIAL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(5) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    funding_rate DOUBLE PRECISION,
    open_interest DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT pk_market_candles PRIMARY KEY (id, timestamp),
    CONSTRAINT uq_market_candle UNIQUE (symbol, timeframe, timestamp)
);

-- Convert market_candles to TimescaleDB hypertable
SELECT create_hypertable('market_candles', 'timestamp', if_not_exists => TRUE);

-- technical_features table
CREATE TABLE IF NOT EXISTS technical_features (
    id BIGSERIAL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(5) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    trend_direction VARCHAR(10),
    bos_bullish BOOLEAN DEFAULT FALSE,
    bos_bearish BOOLEAN DEFAULT FALSE,
    choch_bullish BOOLEAN DEFAULT FALSE,
    choch_bearish BOOLEAN DEFAULT FALSE,
    higher_high BOOLEAN DEFAULT FALSE,
    higher_low BOOLEAN DEFAULT FALSE,
    lower_high BOOLEAN DEFAULT FALSE,
    lower_low BOOLEAN DEFAULT FALSE,
    in_range BOOLEAN DEFAULT FALSE,
    breakout BOOLEAN DEFAULT FALSE,
    breakdown BOOLEAN DEFAULT FALSE,
    swing_high DOUBLE PRECISION,
    swing_low DOUBLE PRECISION,
    candle_type VARCHAR(30),
    engulfing_bull BOOLEAN DEFAULT FALSE,
    engulfing_bear BOOLEAN DEFAULT FALSE,
    pin_bar_bull BOOLEAN DEFAULT FALSE,
    pin_bar_bear BOOLEAN DEFAULT FALSE,
    inside_bar BOOLEAN DEFAULT FALSE,
    doji BOOLEAN DEFAULT FALSE,
    rejection_bull BOOLEAN DEFAULT FALSE,
    rejection_bear BOOLEAN DEFAULT FALSE,
    momentum_bull BOOLEAN DEFAULT FALSE,
    momentum_bear BOOLEAN DEFAULT FALSE,
    candle_body_pct DOUBLE PRECISION,
    upper_wick_pct DOUBLE PRECISION,
    lower_wick_pct DOUBLE PRECISION,
    prev_high_swept BOOLEAN DEFAULT FALSE,
    prev_low_swept BOOLEAN DEFAULT FALSE,
    equal_highs BOOLEAN DEFAULT FALSE,
    equal_lows BOOLEAN DEFAULT FALSE,
    ssl_level DOUBLE PRECISION,
    bsl_level DOUBLE PRECISION,
    dist_to_ssl_pct DOUBLE PRECISION,
    dist_to_bsl_pct DOUBLE PRECISION,
    liquidity_grab_bull BOOLEAN DEFAULT FALSE,
    liquidity_grab_bear BOOLEAN DEFAULT FALSE,
    ema20 DOUBLE PRECISION,
    ema50 DOUBLE PRECISION,
    ema200 DOUBLE PRECISION,
    ema20_slope DOUBLE PRECISION,
    ema50_slope DOUBLE PRECISION,
    price_vs_ema20 DOUBLE PRECISION,
    price_vs_ema50 DOUBLE PRECISION,
    price_vs_ema200 DOUBLE PRECISION,
    rsi DOUBLE PRECISION,
    rsi_divergence INTEGER,
    macd DOUBLE PRECISION,
    macd_signal DOUBLE PRECISION,
    macd_hist DOUBLE PRECISION,
    macd_cross INTEGER,
    adx DOUBLE PRECISION,
    di_plus DOUBLE PRECISION,
    di_minus DOUBLE PRECISION,
    atr DOUBLE PRECISION,
    atr_normalized DOUBLE PRECISION,
    atr_percentile DOUBLE PRECISION,
    bb_upper DOUBLE PRECISION,
    bb_lower DOUBLE PRECISION,
    bb_width DOUBLE PRECISION,
    bb_position DOUBLE PRECISION,
    range_expansion BOOLEAN DEFAULT FALSE,
    range_compression BOOLEAN DEFAULT FALSE,
    volume_sma20 DOUBLE PRECISION,
    relative_volume DOUBLE PRECISION,
    volume_spike BOOLEAN DEFAULT FALSE,
    volume_divergence_bull BOOLEAN DEFAULT FALSE,
    volume_divergence_bear BOOLEAN DEFAULT FALSE,
    obv DOUBLE PRECISION,
    obv_slope DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT pk_technical_features PRIMARY KEY (id, timestamp),
    CONSTRAINT uq_technical_feature UNIQUE (symbol, timeframe, timestamp)
);

-- Convert technical_features to TimescaleDB hypertable
SELECT create_hypertable('technical_features', 'timestamp', if_not_exists => TRUE);

-- Create standard indexes for high-speed queries
CREATE INDEX IF NOT EXISTS ix_candles_sym_tf_ts ON market_candles (symbol, timeframe, timestamp DESC);
CREATE INDEX IF NOT EXISTS ix_features_sym_tf_ts ON technical_features (symbol, timeframe, timestamp DESC);
