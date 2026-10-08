"""
ZECRO-RL — Market Structure Analysis
Deterministic detection of: HH/HL/LH/LL, BOS, CHoCH, swing points,
trend direction, ranges, breakouts, breakdowns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class SwingPoint:
    index: int
    price: float
    swing_type: str  # "high" or "low"
    confirmed: bool = False


@dataclass
class MarketStructure:
    # Trend
    trend_direction: str = "NEUTRAL"  # BULLISH / BEARISH / NEUTRAL

    # Swing labels
    higher_high: bool = False
    higher_low: bool = False
    lower_high: bool = False
    lower_low: bool = False

    # Structure breaks
    bos_bullish: bool = False   # Break of Structure to the upside
    bos_bearish: bool = False   # Break of Structure to the downside
    choch_bullish: bool = False  # Change of Character to bullish
    choch_bearish: bool = False  # Change of Character to bearish

    # State
    in_range: bool = False
    breakout: bool = False
    breakdown: bool = False

    # Key levels
    swing_high: Optional[float] = None
    swing_low: Optional[float] = None
    previous_swing_high: Optional[float] = None
    previous_swing_low: Optional[float] = None

    # Recent swings
    swing_highs: List[SwingPoint] = field(default_factory=list)
    swing_lows: List[SwingPoint] = field(default_factory=list)


def detect_swing_points(
    df: pd.DataFrame,
    pivot_window: int = 5,
) -> Tuple[List[SwingPoint], List[SwingPoint]]:
    """
    Detect swing highs and lows using pivot point method.
    A swing high: high[i] is the highest in window [i-n, i+n]
    A swing low:  low[i]  is the lowest  in window [i-n, i+n]

    Returns: (swing_highs, swing_lows)
    """
    highs = df["high"].values
    lows = df["low"].values
    n = len(highs)

    swing_highs = []
    swing_lows = []

    for i in range(pivot_window, n - pivot_window):
        window_h = highs[max(0, i - pivot_window): i + pivot_window + 1]
        window_l = lows[max(0, i - pivot_window): i + pivot_window + 1]

        if highs[i] == window_h.max():
            swing_highs.append(SwingPoint(
                index=i,
                price=highs[i],
                swing_type="high",
                confirmed=True,
            ))

        if lows[i] == window_l.min():
            swing_lows.append(SwingPoint(
                index=i,
                price=lows[i],
                swing_type="low",
                confirmed=True,
            ))

    return swing_highs, swing_lows


def analyze_market_structure(
    df: pd.DataFrame,
    pivot_window: int = 5,
    range_threshold: float = 0.02,
) -> MarketStructure:
    """
    Full market structure analysis on OHLCV DataFrame.

    Args:
        df: OHLCV DataFrame with columns: open, high, low, close, volume
        pivot_window: bars each side to confirm a swing
        range_threshold: price % range to consider "in range"

    Returns:
        MarketStructure dataclass
    """
    ms = MarketStructure()

    if len(df) < pivot_window * 2 + 5:
        return ms

    swing_highs, swing_lows = detect_swing_points(df, pivot_window)

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return ms

    # Store recent swings
    ms.swing_highs = swing_highs[-10:]
    ms.swing_lows = swing_lows[-10:]

    # Key levels: last confirmed swings
    last_sh = swing_highs[-1]
    last_sl = swing_lows[-1]
    ms.swing_high = last_sh.price
    ms.swing_low = last_sl.price

    # Previous swings
    if len(swing_highs) >= 2:
        ms.previous_swing_high = swing_highs[-2].price
    if len(swing_lows) >= 2:
        ms.previous_swing_low = swing_lows[-2].price

    # ── HH / HL / LH / LL ─────────────────────────────────────────────────────
    # Compare last 2 swing highs and lows
    if ms.previous_swing_high is not None:
        if last_sh.price > ms.previous_swing_high:
            ms.higher_high = True
        else:
            ms.lower_high = True

    if ms.previous_swing_low is not None:
        if last_sl.price > ms.previous_swing_low:
            ms.higher_low = True
        else:
            ms.lower_low = True

    # ── Trend Direction ────────────────────────────────────────────────────────
    if ms.higher_high and ms.higher_low:
        ms.trend_direction = "BULLISH"
    elif ms.lower_high and ms.lower_low:
        ms.trend_direction = "BEARISH"
    else:
        ms.trend_direction = "NEUTRAL"

    # ── BOS (Break of Structure) ───────────────────────────────────────────────
    # BOS Bullish: close breaks above previous swing high in a bullish trend
    # BOS Bearish: close breaks below previous swing low in a bearish trend
    current_close = df["close"].iloc[-1]

    if ms.previous_swing_high is not None:
        if current_close > ms.previous_swing_high and ms.trend_direction == "BULLISH":
            ms.bos_bullish = True

    if ms.previous_swing_low is not None:
        if current_close < ms.previous_swing_low and ms.trend_direction == "BEARISH":
            ms.bos_bearish = True

    # ── CHoCH (Change of Character) ───────────────────────────────────────────
    # CHoCH Bullish: was bearish (LH/LL), now breaks ABOVE previous swing high
    # CHoCH Bearish: was bullish (HH/HL), now breaks BELOW previous swing low
    if ms.trend_direction == "BEARISH" and ms.previous_swing_high is not None:
        if current_close > ms.previous_swing_high:
            ms.choch_bullish = True
            ms.bos_bullish = False  # CHoCH supersedes BOS

    if ms.trend_direction == "BULLISH" and ms.previous_swing_low is not None:
        if current_close < ms.previous_swing_low:
            ms.choch_bearish = True
            ms.bos_bearish = False

    # ── Range / Breakout / Breakdown ────────────────────────────────────────────
    if ms.swing_high is not None and ms.swing_low is not None:
        range_size = (ms.swing_high - ms.swing_low) / ms.swing_low
        if range_size < range_threshold:
            ms.in_range = True

        if current_close > ms.swing_high * (1 + 0.001):
            ms.breakout = True
            ms.in_range = False
        elif current_close < ms.swing_low * (1 - 0.001):
            ms.breakdown = True
            ms.in_range = False

    return ms
