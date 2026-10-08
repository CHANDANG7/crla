"""
ZECRO-RL — Feature Engine (Unified State Builder)
Combines all TA modules into a single state dict / numpy vector for the RL agent.
This is the primary interface — all other modules feed into this.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import structlog

from app.ta.market_structure import MarketStructure, analyze_market_structure
from app.ta.price_action import PriceActionResult, analyze_price_action
from app.ta.liquidity import LiquidityResult, analyze_liquidity
from app.ta.momentum import MomentumResult, analyze_momentum
from app.ta.volatility import VolatilityResult, analyze_volatility
from app.ta.volume import VolumeResult, analyze_volume

logger = structlog.get_logger(__name__)

# Asset ID mapping for embedding
ASSET_IDS: Dict[str, int] = {
    "BTCUSD": 0, "ETHUSD": 1, "SOLUSD": 2, "XRPUSD": 3,
    "DOGEUSDT": 4, "AVAXUSD": 5, "LINKUSD": 6, "MATICUSD": 7,
    "BNBUSD": 8, "ADAUSD": 9,
}

REGIME_IDS: Dict[str, int] = {
    "TRENDING_UP": 0, "TRENDING_DOWN": 1, "RANGING": 2,
    "HIGH_VOLATILITY": 3, "LOW_VOLATILITY": 4, "BREAKOUT": 5, "BREAKDOWN": 6,
}


@dataclass
class MarketState:
    """
    Complete market state used by:
    1. Groq LLM for reasoning
    2. RL agent as numpy state vector
    """
    symbol: str
    timeframe: str
    timestamp: datetime

    # ── Market Structure ───────────────────────────────────────────────────────
    trend_direction: str = "NEUTRAL"
    higher_high: bool = False
    higher_low: bool = False
    lower_high: bool = False
    lower_low: bool = False
    bos_bullish: bool = False
    bos_bearish: bool = False
    choch_bullish: bool = False
    choch_bearish: bool = False
    in_range: bool = False
    breakout: bool = False
    breakdown: bool = False
    swing_high: Optional[float] = None
    swing_low: Optional[float] = None

    # ── Price Action ──────────────────────────────────────────────────────────
    candle_type: str = "NEUTRAL"
    engulfing_bull: bool = False
    engulfing_bear: bool = False
    pin_bar_bull: bool = False
    pin_bar_bear: bool = False
    inside_bar: bool = False
    doji: bool = False
    rejection_bull: bool = False
    rejection_bear: bool = False
    momentum_bull: bool = False
    momentum_bear: bool = False
    candle_body_pct: float = 0.0
    upper_wick_pct: float = 0.0
    lower_wick_pct: float = 0.0

    # ── Liquidity ─────────────────────────────────────────────────────────────
    prev_high_swept: bool = False
    prev_low_swept: bool = False
    equal_highs: bool = False
    equal_lows: bool = False
    liquidity_grab_bull: bool = False
    liquidity_grab_bear: bool = False
    ssl_level: Optional[float] = None
    bsl_level: Optional[float] = None
    dist_to_ssl_pct: float = 0.0
    dist_to_bsl_pct: float = 0.0

    # ── Momentum ──────────────────────────────────────────────────────────────
    ema20: Optional[float] = None
    ema50: Optional[float] = None
    ema200: Optional[float] = None
    ema20_slope: float = 0.0
    ema50_slope: float = 0.0
    price_vs_ema20: float = 0.0
    price_vs_ema50: float = 0.0
    price_vs_ema200: float = 0.0
    rsi: float = 50.0
    rsi_divergence: int = 0
    macd_hist: float = 0.0
    macd_cross: int = 0
    adx: float = 0.0
    di_plus: float = 0.0
    di_minus: float = 0.0
    trend_strong: bool = False

    # ── Volatility ────────────────────────────────────────────────────────────
    atr: Optional[float] = None
    atr_normalized: float = 0.0
    atr_percentile: float = 50.0
    bb_width: float = 0.0
    bb_position: float = 0.5
    range_expansion: bool = False
    range_compression: bool = False

    # ── Volume ────────────────────────────────────────────────────────────────
    relative_volume: float = 1.0
    volume_spike: bool = False
    volume_trend: float = 0.0
    obv_slope: float = 0.0
    volume_divergence_bull: bool = False
    volume_divergence_bear: bool = False
    high_volume_bull: bool = False
    high_volume_bear: bool = False
    volume_delta: float = 0.0

    # ── Market Regime ─────────────────────────────────────────────────────────
    market_regime: str = "NEUTRAL"

    # ── Current price ─────────────────────────────────────────────────────────
    current_price: float = 0.0
    current_volume: float = 0.0

    def to_rl_vector(self) -> np.ndarray:
        """
        Convert state to a flat numpy array for the RL policy network.
        Consistent ordering is critical — do NOT change order.
        Returns array of shape (160,)
        """
        def b(x): return float(bool(x))
        def f(x, default=0.0): return float(x) if x is not None else default
        def clip(x, lo, hi): return max(lo, min(hi, x))

        vec = [
            # Market Structure (14)
            float({"BULLISH": 1.0, "BEARISH": -1.0, "NEUTRAL": 0.0}.get(self.trend_direction, 0.0)),
            b(self.higher_high), b(self.higher_low),
            b(self.lower_high), b(self.lower_low),
            b(self.bos_bullish), b(self.bos_bearish),
            b(self.choch_bullish), b(self.choch_bearish),
            b(self.in_range), b(self.breakout), b(self.breakdown),
            # Swing distance (normalized)
            clip(f(self.dist_to_ssl_pct), -10, 10) / 10 if self.ssl_level else 0.0,
            clip(f(self.dist_to_bsl_pct), -10, 10) / 10 if self.bsl_level else 0.0,

            # Price Action (12)
            b(self.engulfing_bull), b(self.engulfing_bear),
            b(self.pin_bar_bull), b(self.pin_bar_bear),
            b(self.inside_bar), b(self.doji),
            b(self.rejection_bull), b(self.rejection_bear),
            b(self.momentum_bull), b(self.momentum_bear),
            clip(self.candle_body_pct, 0, 1),
            clip(self.lower_wick_pct - self.upper_wick_pct, -1, 1),

            # Liquidity (8)
            b(self.prev_high_swept), b(self.prev_low_swept),
            b(self.equal_highs), b(self.equal_lows),
            b(self.liquidity_grab_bull), b(self.liquidity_grab_bear),
            clip(self.dist_to_ssl_pct / 10, 0, 1),
            clip(self.dist_to_bsl_pct / 10, 0, 1),

            # Momentum (14)
            clip(self.ema20_slope / 2, -1, 1),
            clip(self.ema50_slope / 2, -1, 1),
            clip(self.price_vs_ema20 / 5, -1, 1),
            clip(self.price_vs_ema50 / 10, -1, 1),
            clip(self.price_vs_ema200 / 20, -1, 1),
            (self.rsi - 50) / 50,  # -1 to 1
            float(self.rsi_divergence),
            clip(self.macd_hist / max(abs(self.macd_hist), 1e-10), -1, 1),
            float(self.macd_cross),
            clip(self.adx / 60, 0, 1),
            clip(self.di_plus / 60, 0, 1),
            clip(self.di_minus / 60, 0, 1),
            1.0 if self.di_plus > self.di_minus else -1.0,
            b(self.trend_strong),

            # Volatility (8)
            clip(self.atr_normalized / 5, 0, 1),
            self.atr_percentile / 100,
            clip(self.bb_width / 20, 0, 1),
            clip(self.bb_position, 0, 1),
            b(self.range_expansion),
            b(self.range_compression),
            0.0, 0.0,  # reserved

            # Volume (8)
            clip(self.relative_volume / 5, 0, 1),
            b(self.volume_spike),
            clip(self.volume_trend, -1, 1),
            clip(self.obv_slope / max(abs(self.obv_slope), 1e-10), -1, 1),
            b(self.volume_divergence_bull),
            b(self.volume_divergence_bear),
            b(self.high_volume_bull),
            b(self.high_volume_bear),

            # Regime (7 one-hot)
            b(self.market_regime == "TRENDING_UP"),
            b(self.market_regime == "TRENDING_DOWN"),
            b(self.market_regime == "RANGING"),
            b(self.market_regime == "HIGH_VOLATILITY"),
            b(self.market_regime == "LOW_VOLATILITY"),
            b(self.market_regime == "BREAKOUT"),
            b(self.market_regime == "BREAKDOWN"),

            # Time features (4) — cyclical encoding
            *self._time_features(),

            # Asset embedding (10 one-hot)
            *self._asset_features(),
        ]

        # Pad to exactly 160
        vec_arr = np.array(vec, dtype=np.float32)
        if len(vec_arr) < 160:
            vec_arr = np.pad(vec_arr, (0, 160 - len(vec_arr)))
        return vec_arr[:160]

    def _time_features(self) -> List[float]:
        """Cyclical hour + day-of-week encoding."""
        hour = self.timestamp.hour
        dow = self.timestamp.weekday()
        return [
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * dow / 7),
            np.cos(2 * np.pi * dow / 7),
        ]

    def _asset_features(self) -> List[float]:
        """One-hot asset embedding."""
        asset_id = ASSET_IDS.get(self.symbol, 9)
        vec = [0.0] * 10
        if asset_id < 10:
            vec[asset_id] = 1.0
        return vec

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable dict for API / storage."""
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


class FeatureEngine:
    """
    Unified feature computation from raw OHLCV DataFrames.
    Primary entry point for all TA computation.
    """

    def compute_state(
        self,
        df_1h: pd.DataFrame,
        df_15m: Optional[pd.DataFrame] = None,
        symbol: str = "BTCUSD",
        market_regime: str = "NEUTRAL",
    ) -> MarketState:
        """
        Compute complete MarketState from OHLCV data.

        Args:
            df_1h: 1-hour OHLCV DataFrame (used for primary analysis)
            df_15m: 15-minute OHLCV DataFrame (used for entry features)
            symbol: Trading symbol
            market_regime: Current detected regime

        Returns:
            Complete MarketState
        """
        if len(df_1h) < 5:
            raise ValueError(f"Insufficient data: {len(df_1h)} bars")

        primary_df = df_1h

        # Run all TA modules
        ms: MarketStructure = analyze_market_structure(primary_df)
        pa: PriceActionResult = analyze_price_action(primary_df)
        liq: LiquidityResult = analyze_liquidity(primary_df)
        mom: MomentumResult = analyze_momentum(primary_df)
        vol_result: VolatilityResult = analyze_volatility(primary_df)
        vol_anal: VolumeResult = analyze_volume(primary_df)

        state = MarketState(
            symbol=symbol,
            timeframe="1h",
            timestamp=primary_df.index[-1] if isinstance(primary_df.index[-1], datetime)
                else datetime.utcnow(),
            current_price=float(primary_df["close"].iloc[-1]),
            current_volume=float(primary_df["volume"].iloc[-1]),
            market_regime=market_regime,

            # Market structure
            trend_direction=ms.trend_direction,
            higher_high=ms.higher_high,
            higher_low=ms.higher_low,
            lower_high=ms.lower_high,
            lower_low=ms.lower_low,
            bos_bullish=ms.bos_bullish,
            bos_bearish=ms.bos_bearish,
            choch_bullish=ms.choch_bullish,
            choch_bearish=ms.choch_bearish,
            in_range=ms.in_range,
            breakout=ms.breakout,
            breakdown=ms.breakdown,
            swing_high=ms.swing_high,
            swing_low=ms.swing_low,

            # Price action
            candle_type=pa.candle_type,
            engulfing_bull=pa.engulfing_bull,
            engulfing_bear=pa.engulfing_bear,
            pin_bar_bull=pa.pin_bar_bull,
            pin_bar_bear=pa.pin_bar_bear,
            inside_bar=pa.inside_bar,
            doji=pa.doji,
            rejection_bull=pa.rejection_bull,
            rejection_bear=pa.rejection_bear,
            momentum_bull=pa.momentum_bull,
            momentum_bear=pa.momentum_bear,
            candle_body_pct=pa.candle_body_pct,
            upper_wick_pct=pa.upper_wick_pct,
            lower_wick_pct=pa.lower_wick_pct,

            # Liquidity
            prev_high_swept=liq.prev_high_swept,
            prev_low_swept=liq.prev_low_swept,
            equal_highs=liq.equal_highs,
            equal_lows=liq.equal_lows,
            liquidity_grab_bull=liq.liquidity_grab_bull,
            liquidity_grab_bear=liq.liquidity_grab_bear,
            ssl_level=liq.ssl_level,
            bsl_level=liq.bsl_level,
            dist_to_ssl_pct=liq.dist_to_ssl_pct or 0.0,
            dist_to_bsl_pct=liq.dist_to_bsl_pct or 0.0,

            # Momentum
            ema20=mom.ema20,
            ema50=mom.ema50,
            ema200=mom.ema200,
            ema20_slope=mom.ema20_slope or 0.0,
            ema50_slope=mom.ema50_slope or 0.0,
            price_vs_ema20=mom.price_vs_ema20 or 0.0,
            price_vs_ema50=mom.price_vs_ema50 or 0.0,
            price_vs_ema200=mom.price_vs_ema200 or 0.0,
            rsi=mom.rsi or 50.0,
            rsi_divergence=mom.rsi_divergence,
            macd_hist=mom.macd_hist or 0.0,
            macd_cross=mom.macd_cross,
            adx=mom.adx or 0.0,
            di_plus=mom.di_plus or 0.0,
            di_minus=mom.di_minus or 0.0,
            trend_strong=mom.trend_strong,

            # Volatility
            atr=vol_result.atr,
            atr_normalized=vol_result.atr_normalized or 0.0,
            atr_percentile=vol_result.atr_percentile or 50.0,
            bb_width=vol_result.bb_width or 0.0,
            bb_position=vol_result.bb_position or 0.5,
            range_expansion=vol_result.range_expansion,
            range_compression=vol_result.range_compression,

            # Volume
            relative_volume=vol_anal.relative_volume or 1.0,
            volume_spike=vol_anal.volume_spike,
            volume_trend=vol_anal.volume_trend or 0.0,
            obv_slope=vol_anal.obv_slope or 0.0,
            volume_divergence_bull=vol_anal.volume_divergence_bull,
            volume_divergence_bear=vol_anal.volume_divergence_bear,
            high_volume_bull=vol_anal.high_volume_bull,
            high_volume_bear=vol_anal.high_volume_bear,
            volume_delta=vol_anal.volume_delta or 0.0,
        )

        logger.debug(
            "State computed",
            symbol=symbol,
            trend=state.trend_direction,
            regime=market_regime,
            rsi=state.rsi,
            adx=state.adx,
        )

        return state


# Singleton
feature_engine = FeatureEngine()
