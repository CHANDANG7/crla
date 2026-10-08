"""
ZECRO-RL — Liquidity Analysis
Detects: liquidity zones, sweeps, equal highs/lows, buy/sell-side liquidity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class LiquidityLevel:
    price: float
    level_type: str  # "high" or "low"
    strength: int    # how many times touched
    swept: bool = False


@dataclass
class LiquidityResult:
    # Sweeps on the current bar
    prev_high_swept: bool = False
    prev_low_swept: bool = False
    liquidity_grab_bull: bool = False  # swept lows + bullish close
    liquidity_grab_bear: bool = False  # swept highs + bearish close

    # Equal highs/lows (cluster of touches)
    equal_highs: bool = False
    equal_lows: bool = False

    # Key levels
    ssl_level: Optional[float] = None   # sell-side liquidity (below)
    bsl_level: Optional[float] = None   # buy-side liquidity (above)
    dist_to_ssl_pct: Optional[float] = None
    dist_to_bsl_pct: Optional[float] = None

    # All detected levels
    liquidity_levels: List[LiquidityLevel] = field(default_factory=list)


def analyze_liquidity(
    df: pd.DataFrame,
    lookback: int = 50,
    equal_threshold_pct: float = 0.002,  # 0.2% price tolerance for "equal"
    swing_window: int = 3,
) -> LiquidityResult:
    """
    Analyze liquidity in the market structure.

    Liquidity sits above swing highs (buy-side) and below swing lows (sell-side).
    When price sweeps these levels and reverses, it's a liquidity grab.

    Args:
        df: OHLCV DataFrame
        lookback: bars to look back for liquidity levels
        equal_threshold_pct: price % tolerance for equal highs/lows
        swing_window: bars each side for swing detection
    """
    result = LiquidityResult()

    if len(df) < lookback + 5:
        return result

    # Work on lookback window
    window = df.tail(lookback + 5).reset_index(drop=True)
    current = window.iloc[-1]
    prev = window.iloc[-2]

    # ── Detect recent swing highs and lows ─────────────────────────────────────
    swing_highs = []
    swing_lows = []
    n = len(window)

    for i in range(swing_window, n - swing_window - 1):
        h_slice = window["high"].iloc[i - swing_window: i + swing_window + 1]
        l_slice = window["low"].iloc[i - swing_window: i + swing_window + 1]

        if window["high"].iloc[i] == h_slice.max():
            swing_highs.append(window["high"].iloc[i])

        if window["low"].iloc[i] == l_slice.min():
            swing_lows.append(window["low"].iloc[i])

    # ── Previous high/low sweep ────────────────────────────────────────────────
    # Prev high: last 20 bar high
    recent = window.tail(21).head(20)
    prev_high = recent["high"].max()
    prev_low = recent["low"].min()

    # Check if current candle swept previous high (wick above) but closed below
    if current["high"] > prev_high and current["close"] < prev_high:
        result.prev_high_swept = True
        result.liquidity_grab_bear = True  # swept highs = bearish reversal possible

    # Check if current candle swept previous low (wick below) but closed above
    if current["low"] < prev_low and current["close"] > prev_low:
        result.prev_low_swept = True
        result.liquidity_grab_bull = True  # swept lows = bullish reversal possible

    # ── Equal highs / lows ─────────────────────────────────────────────────────
    if len(swing_highs) >= 2:
        highs_arr = np.array(swing_highs)
        for i in range(len(highs_arr)):
            for j in range(i + 1, len(highs_arr)):
                if abs(highs_arr[i] - highs_arr[j]) / highs_arr[i] < equal_threshold_pct:
                    result.equal_highs = True
                    break

    if len(swing_lows) >= 2:
        lows_arr = np.array(swing_lows)
        for i in range(len(lows_arr)):
            for j in range(i + 1, len(lows_arr)):
                if abs(lows_arr[i] - lows_arr[j]) / lows_arr[i] < equal_threshold_pct:
                    result.equal_lows = True
                    break

    # ── Buy-side / Sell-side liquidity levels ─────────────────────────────────
    current_price = current["close"]

    # BSL = highest swing high above current price
    above = [h for h in swing_highs if h > current_price]
    if above:
        result.bsl_level = min(above)  # nearest BSL above
        result.dist_to_bsl_pct = (result.bsl_level - current_price) / current_price * 100

    # SSL = lowest swing low below current price
    below = [l for l in swing_lows if l < current_price]
    if below:
        result.ssl_level = max(below)  # nearest SSL below
        result.dist_to_ssl_pct = (current_price - result.ssl_level) / current_price * 100

    # ── Build liquidity level objects ──────────────────────────────────────────
    for h in swing_highs[-5:]:
        result.liquidity_levels.append(LiquidityLevel(
            price=h,
            level_type="high",
            strength=1,
            swept=h < current_price,  # price went above it already
        ))

    for l in swing_lows[-5:]:
        result.liquidity_levels.append(LiquidityLevel(
            price=l,
            level_type="low",
            strength=1,
            swept=l > current_price,  # price went below it already
        ))

    return result
