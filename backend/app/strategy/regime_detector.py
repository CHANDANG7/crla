"""
ZECRO-RL — Market Regime Detector
Classifies market conditions into deterministic regimes:
- TRENDING_UP
- TRENDING_DOWN
- RANGING
- HIGH_VOLATILITY
- LOW_VOLATILITY
- BREAKOUT
- BREAKDOWN
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import structlog

from app.models import MarketRegimeEnum
from app.ta.feature_engine import MarketState

logger = structlog.get_logger(__name__)


@dataclass
class RegimeResult:
    regime: MarketRegimeEnum
    confidence: float
    details: Dict[str, Any]


class RegimeDetector:
    """
    Deterministic rule-based market regime detector.
    Analyzes trend strength (ADX, EMAs), volatility (ATR percentile, BB width),
    and structure (BOS, Breakouts/Breakdowns).
    """

    def detect(self, state: MarketState) -> RegimeResult:
        """
        Detect regime from computed MarketState.
        """
        adx = state.adx or 0.0
        atr_pct = state.atr_percentile or 50.0
        bb_width = state.bb_width or 0.0
        trend = state.trend_direction or "NEUTRAL"

        # 1. Breakout / Breakdown priority
        if state.breakout and (state.volume_spike or state.relative_volume > 1.5):
            return RegimeResult(
                regime=MarketRegimeEnum.BREAKOUT,
                confidence=0.85,
                details={
                    "reason": "Swing high breakout with high volume",
                    "rvol": state.relative_volume,
                    "trend": trend,
                },
            )

        if state.breakdown and (state.volume_spike or state.relative_volume > 1.5):
            return RegimeResult(
                regime=MarketRegimeEnum.BREAKDOWN,
                confidence=0.85,
                details={
                    "reason": "Swing low breakdown with high volume",
                    "rvol": state.relative_volume,
                    "trend": trend,
                },
            )

        # 2. Extreme Volatility regimes
        if atr_pct > 85.0 or (bb_width > 6.0 and state.range_expansion):
            return RegimeResult(
                regime=MarketRegimeEnum.HIGH_VOLATILITY,
                confidence=min(0.95, atr_pct / 100.0),
                details={
                    "atr_percentile": atr_pct,
                    "bb_width": bb_width,
                    "expansion": state.range_expansion,
                },
            )

        if atr_pct < 20.0 and state.range_compression:
            return RegimeResult(
                regime=MarketRegimeEnum.LOW_VOLATILITY,
                confidence=min(0.90, (100.0 - atr_pct) / 100.0),
                details={
                    "atr_percentile": atr_pct,
                    "bb_width": bb_width,
                    "compression": state.range_compression,
                },
            )

        # 3. Trending regimes (ADX >= 25 + EMA alignment)
        if adx >= 25.0:
            if (
                trend == "BULLISH"
                or (state.price_vs_ema20 > 0 and state.price_vs_ema50 > 0 and state.price_vs_ema200 > 0)
                or state.bos_bullish
            ):
                confidence = min(0.95, 0.5 + (adx / 100.0))
                return RegimeResult(
                    regime=MarketRegimeEnum.TRENDING_UP,
                    confidence=round(confidence, 3),
                    details={
                        "adx": adx,
                        "di_plus": state.di_plus,
                        "di_minus": state.di_minus,
                        "trend": trend,
                    },
                )
            elif (
                trend == "BEARISH"
                or (state.price_vs_ema20 < 0 and state.price_vs_ema50 < 0 and state.price_vs_ema200 < 0)
                or state.bos_bearish
            ):
                confidence = min(0.95, 0.5 + (adx / 100.0))
                return RegimeResult(
                    regime=MarketRegimeEnum.TRENDING_DOWN,
                    confidence=round(confidence, 3),
                    details={
                        "adx": adx,
                        "di_plus": state.di_plus,
                        "di_minus": state.di_minus,
                        "trend": trend,
                    },
                )

        # 4. Ranging / Neutral
        return RegimeResult(
            regime=MarketRegimeEnum.RANGING,
            confidence=0.70,
            details={
                "adx": adx,
                "in_range": state.in_range,
                "bb_width": bb_width,
            },
        )


regime_detector = RegimeDetector()
