"""
ZECRO-RL — Volatility Analysis
ATR, ATR percentile, Bollinger Bands, range expansion/compression.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class VolatilityResult:
    atr: Optional[float] = None
    atr_normalized: Optional[float] = None  # ATR / price * 100
    atr_percentile: Optional[float] = None  # 0-100 vs last 100 ATRs
    bb_upper: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_mid: Optional[float] = None
    bb_width: Optional[float] = None
    bb_position: Optional[float] = None   # 0=lower, 1=upper
    range_expansion: bool = False
    range_compression: bool = False
    historical_volatility: Optional[float] = None  # 20-day realized vol


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    return tr.ewm(alpha=1 / period, adjust=False).mean()


def calculate_bollinger_bands(
    close: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, sma, lower


def analyze_volatility(
    df: pd.DataFrame,
    atr_period: int = 14,
    bb_period: int = 20,
    percentile_lookback: int = 100,
    expansion_threshold: float = 1.3,   # ATR > 1.3x median = expansion
    compression_threshold: float = 0.7,  # ATR < 0.7x median = compression
) -> VolatilityResult:
    """
    Full volatility analysis.

    Args:
        df: OHLCV DataFrame
        atr_period: ATR calculation period
        bb_period: Bollinger Band period
        percentile_lookback: bars to compute ATR percentile
        expansion_threshold: multiplier above median for range expansion
        compression_threshold: multiplier below median for range compression
    """
    v = VolatilityResult()
    close = df["close"]
    n = len(df)

    if n < max(atr_period, bb_period) + 5:
        return v

    # ── ATR ────────────────────────────────────────────────────────────────────
    atr_series = calculate_atr(df, atr_period)
    v.atr = float(atr_series.iloc[-1])
    current_price = float(close.iloc[-1])
    v.atr_normalized = v.atr / current_price * 100  # as % of price

    # ATR percentile vs lookback window
    if n >= percentile_lookback:
        atr_window = atr_series.tail(percentile_lookback)
        v.atr_percentile = float(
            np.percentile(atr_window.dropna(), 100) -
            np.searchsorted(np.sort(atr_window.dropna()), v.atr) / len(atr_window) * 100
        )
        # Simpler percentile:
        v.atr_percentile = float(
            (atr_window.dropna() <= v.atr).sum() / len(atr_window.dropna()) * 100
        )

        # Range expansion / compression vs median ATR
        median_atr = float(atr_window.median())
        if median_atr > 0:
            ratio = v.atr / median_atr
            v.range_expansion = ratio >= expansion_threshold
            v.range_compression = ratio <= compression_threshold

    # ── Bollinger Bands ────────────────────────────────────────────────────────
    if n >= bb_period:
        bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(close, bb_period)
        v.bb_upper = float(bb_upper.iloc[-1])
        v.bb_mid = float(bb_mid.iloc[-1])
        v.bb_lower = float(bb_lower.iloc[-1])

        bb_range = v.bb_upper - v.bb_lower
        if bb_range > 0:
            v.bb_width = bb_range / v.bb_mid * 100  # as % of middle
            v.bb_position = (current_price - v.bb_lower) / bb_range  # 0-1

    # ── Historical Volatility (20-day realized vol) ────────────────────────────
    if n >= 21:
        log_returns = np.log(close / close.shift(1)).tail(20).dropna()
        v.historical_volatility = float(log_returns.std() * np.sqrt(365) * 100)  # annualized %

    return v
