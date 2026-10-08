"""
ZECRO-RL — Price Action Pattern Detection
Detects: engulfing, pin bar, inside bar, doji, rejection, momentum candles.
All deterministic — no LLM involved.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class PriceActionResult:
    candle_type: str = "NEUTRAL"

    # Patterns
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
    harami_bull: bool = False
    harami_bear: bool = False

    # Candle metrics (useful as RL features)
    candle_body_pct: float = 0.0   # body / range
    upper_wick_pct: float = 0.0    # upper wick / range
    lower_wick_pct: float = 0.0    # lower wick / range
    is_bullish: bool = False
    is_bearish: bool = False


def analyze_price_action(
    df: pd.DataFrame,
    doji_threshold: float = 0.1,       # body < 10% of range
    pin_bar_wick_ratio: float = 2.5,   # wick must be 2.5x the body
    momentum_body_ratio: float = 0.70,  # body > 70% of range
) -> PriceActionResult:
    """
    Analyze price action patterns from last 2 candles.

    Args:
        df: OHLCV DataFrame, last row is current candle
        doji_threshold: body/range ratio below which = doji
        pin_bar_wick_ratio: wick/body ratio for pin bar
        momentum_body_ratio: body/range ratio for momentum candle
    """
    pa = PriceActionResult()

    if len(df) < 2:
        return pa

    # Current and previous candles
    curr = df.iloc[-1]
    prev = df.iloc[-2]

    c_open, c_high, c_low, c_close = curr.open, curr.high, curr.low, curr.close
    p_open, p_high, p_low, p_close = prev.open, prev.high, prev.low, prev.close

    c_range = c_high - c_low
    if c_range == 0:
        return pa

    c_body = abs(c_close - c_open)
    c_upper_wick = c_high - max(c_open, c_close)
    c_lower_wick = min(c_open, c_close) - c_low

    pa.candle_body_pct = c_body / c_range
    pa.upper_wick_pct = c_upper_wick / c_range
    pa.lower_wick_pct = c_lower_wick / c_range
    pa.is_bullish = c_close > c_open
    pa.is_bearish = c_close < c_open

    # ── Doji ──────────────────────────────────────────────────────────────────
    if pa.candle_body_pct < doji_threshold:
        pa.doji = True
        pa.candle_type = "DOJI"
        return pa  # doji overrides other patterns

    # ── Inside Bar ────────────────────────────────────────────────────────────
    p_range = p_high - p_low
    p_body = abs(p_close - p_open)
    if p_range > 0:
        if c_high <= p_high and c_low >= p_low:
            pa.inside_bar = True
            pa.candle_type = "INSIDE_BAR"

    # ── Engulfing ─────────────────────────────────────────────────────────────
    if not pa.inside_bar:
        if p_body > 0:
            # Bullish engulfing: prev was bearish, current bullish and larger
            if (p_close < p_open and c_close > c_open
                    and c_close > p_open and c_open < p_close):
                pa.engulfing_bull = True
                pa.candle_type = "ENGULFING_BULL"

            # Bearish engulfing: prev was bullish, current bearish and larger
            elif (p_close > p_open and c_close < c_open
                  and c_open > p_close and c_close < p_open):
                pa.engulfing_bear = True
                pa.candle_type = "ENGULFING_BEAR"

    # ── Pin Bar / Hammer ───────────────────────────────────────────────────────
    if c_body > 0:
        # Bullish pin bar: long lower wick, small body at top
        if (c_lower_wick >= c_body * pin_bar_wick_ratio
                and c_upper_wick <= c_body * 0.5):
            pa.pin_bar_bull = True
            pa.candle_type = "PIN_BAR_BULL"

        # Bearish pin bar: long upper wick, small body at bottom
        elif (c_upper_wick >= c_body * pin_bar_wick_ratio
              and c_lower_wick <= c_body * 0.5):
            pa.pin_bar_bear = True
            pa.candle_type = "PIN_BAR_BEAR"

    # ── Rejection Candles ─────────────────────────────────────────────────────
    # Bullish rejection: lower wick > 60% of range, candle bullish
    if pa.lower_wick_pct > 0.60 and pa.is_bullish and not pa.pin_bar_bull:
        pa.rejection_bull = True
        if not pa.candle_type or pa.candle_type == "NEUTRAL":
            pa.candle_type = "REJECTION_BULL"

    # Bearish rejection: upper wick > 60% of range, candle bearish
    if pa.upper_wick_pct > 0.60 and pa.is_bearish and not pa.pin_bar_bear:
        pa.rejection_bear = True
        if not pa.candle_type or pa.candle_type == "NEUTRAL":
            pa.candle_type = "REJECTION_BEAR"

    # ── Momentum Candles ───────────────────────────────────────────────────────
    if pa.candle_body_pct >= momentum_body_ratio:
        if pa.is_bullish:
            pa.momentum_bull = True
            if pa.candle_type == "NEUTRAL":
                pa.candle_type = "MOMENTUM_BULL"
        else:
            pa.momentum_bear = True
            if pa.candle_type == "NEUTRAL":
                pa.candle_type = "MOMENTUM_BEAR"

    # ── Harami ────────────────────────────────────────────────────────────────
    if p_body > 0 and c_body > 0:
        # Bullish harami: prev bearish, small bullish inside
        if (p_close < p_open and c_close > c_open
                and c_close < p_open and c_open > p_close
                and c_body < p_body * 0.5):
            pa.harami_bull = True
            if pa.candle_type == "NEUTRAL":
                pa.candle_type = "HARAMI_BULL"

        # Bearish harami: prev bullish, small bearish inside
        elif (p_close > p_open and c_close < c_open
              and c_open < p_close and c_close > p_open
              and c_body < p_body * 0.5):
            pa.harami_bear = True
            if pa.candle_type == "NEUTRAL":
                pa.candle_type = "HARAMI_BEAR"

    if pa.candle_type == "NEUTRAL":
        if pa.is_bullish:
            pa.candle_type = "BULLISH"
        elif pa.is_bearish:
            pa.candle_type = "BEARISH"

    return pa
