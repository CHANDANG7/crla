"""
ZECRO-RL — Lower Timeframe (15M) Entry Engine
Validates specific technical setups and computes precise Entry, SL, TP1, and TP2.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import structlog

from app.llm.analysis_agent import EntryAnalysisOutput, groq_agent
from app.models import ActionEnum, BiasEnum
from app.strategy.bias_engine import BiasResult
from app.ta.feature_engine import MarketState

logger = structlog.get_logger(__name__)


@dataclass
class EntrySignal:
    is_valid: bool
    symbol: str
    action: ActionEnum
    entry_price: float
    stop_loss: float
    tp1: float
    tp2: float
    risk_r: float
    confidence: float
    pattern: str
    reasons: List[str]
    invalidation: str


class EntryEngine:
    """
    15M Entry Engine.
    Enforces multi-timeframe alignment (1H bias + 15M confirmation).
    """

    def evaluate_entry(
        self,
        state_15m: MarketState,
        bias_1h: BiasResult,
        atr_buffer_multiplier: float = 0.5,
    ) -> EntrySignal:
        """
        Deterministic 15M entry validation aligned with 1H bias.
        """
        price = state_15m.current_price
        atr = state_15m.atr or (price * 0.005)
        buffer = atr * atr_buffer_multiplier

        # Default no-trade signal
        no_signal = EntrySignal(
            is_valid=False,
            symbol=state_15m.symbol,
            action=ActionEnum.HOLD,
            entry_price=price,
            stop_loss=price,
            tp1=price,
            tp2=price,
            risk_r=0.0,
            confidence=0.0,
            pattern="NO_SETUP",
            reasons=["No aligned setup"],
            invalidation="N/A",
        )

        if bias_1h.bias == BiasEnum.NEUTRAL:
            return no_signal

        # ── LONG SETUP EVALUATION ──────────────────────────────────────────────
        if bias_1h.bias == BiasEnum.LONG:
            reasons = []
            valid = False
            pattern = "LONG_MOMENTUM"

            # Check for liquidity sweeps & price action
            if state_15m.prev_low_swept or state_15m.liquidity_grab_bull:
                valid = True
                pattern = "SSL_SWEEP_REVERSAL"
                reasons.append("15M sell-side liquidity swept")
            elif state_15m.engulfing_bull or state_15m.pin_bar_bull or state_15m.rejection_bull:
                valid = True
                pattern = "PA_BULLISH_CANDLE"
                reasons.append("15M bullish reversal candlestick")
            elif state_15m.bos_bullish or state_15m.choch_bullish:
                valid = True
                pattern = "STRUCTURE_BULLISH_BREAK"
                reasons.append("15M bullish structure break/change")

            if not valid:
                return no_signal

            # Calculate SL (swing low or current low minus ATR buffer)
            ref_low = state_15m.swing_low or (price - atr)
            stop_loss = round(min(ref_low - buffer, price - buffer), 2)
            risk_dist = price - stop_loss

            if risk_dist <= 0 or risk_dist > (price * 0.05):
                # Invalid SL distance (must be positive and <= 5%)
                return no_signal

            tp1 = round(price + (1.5 * risk_dist), 2)
            tp2 = round(price + (3.0 * risk_dist), 2)

            return EntrySignal(
                is_valid=True,
                symbol=state_15m.symbol,
                action=ActionEnum.LONG,
                entry_price=price,
                stop_loss=stop_loss,
                tp1=tp1,
                tp2=tp2,
                risk_r=1.0,
                confidence=bias_1h.confidence,
                pattern=pattern,
                reasons=reasons,
                invalidation=f"Close below {stop_loss}",
            )

        # ── SHORT SETUP EVALUATION ─────────────────────────────────────────────
        if bias_1h.bias == BiasEnum.SHORT:
            reasons = []
            valid = False
            pattern = "SHORT_MOMENTUM"

            # Check for liquidity sweeps & price action
            if state_15m.prev_high_swept or state_15m.liquidity_grab_bear:
                valid = True
                pattern = "BSL_SWEEP_REVERSAL"
                reasons.append("15M buy-side liquidity swept")
            elif state_15m.engulfing_bear or state_15m.pin_bar_bear or state_15m.rejection_bear:
                valid = True
                pattern = "PA_BEARISH_CANDLE"
                reasons.append("15M bearish reversal candlestick")
            elif state_15m.bos_bearish or state_15m.choch_bearish:
                valid = True
                pattern = "STRUCTURE_BEARISH_BREAK"
                reasons.append("15M bearish structure break/change")

            if not valid:
                return no_signal

            # Calculate SL (swing high or current high plus ATR buffer)
            ref_high = state_15m.swing_high or (price + atr)
            stop_loss = round(max(ref_high + buffer, price + buffer), 2)
            risk_dist = stop_loss - price

            if risk_dist <= 0 or risk_dist > (price * 0.05):
                # Invalid SL distance
                return no_signal

            tp1 = round(price - (1.5 * risk_dist), 2)
            tp2 = round(price - (3.0 * risk_dist), 2)

            return EntrySignal(
                is_valid=True,
                symbol=state_15m.symbol,
                action=ActionEnum.SHORT,
                entry_price=price,
                stop_loss=stop_loss,
                tp1=tp1,
                tp2=tp2,
                risk_r=1.0,
                confidence=bias_1h.confidence,
                pattern=pattern,
                reasons=reasons,
                invalidation=f"Close above {stop_loss}",
            )

        return no_signal


entry_engine = EntryEngine()
