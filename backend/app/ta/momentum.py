"""
ZECRO-RL — Momentum Indicators
Deterministic: EMA, RSI, MACD, ADX, DI+, DI-, divergence detection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class MomentumResult:
    # EMAs
    ema20: Optional[float] = None
    ema50: Optional[float] = None
    ema200: Optional[float] = None
    ema20_slope: Optional[float] = None  # % change over 3 bars
    ema50_slope: Optional[float] = None
    price_vs_ema20: Optional[float] = None  # % above/below
    price_vs_ema50: Optional[float] = None
    price_vs_ema200: Optional[float] = None

    # RSI
    rsi: Optional[float] = None
    rsi_overbought: bool = False
    rsi_oversold: bool = False
    rsi_divergence: int = 0  # +1 bullish, -1 bearish, 0 none

    # MACD
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_hist: Optional[float] = None
    macd_cross: int = 0  # +1 bullish cross, -1 bearish cross

    # ADX / DI
    adx: Optional[float] = None
    di_plus: Optional[float] = None
    di_minus: Optional[float] = None
    trend_strong: bool = False  # ADX > 25


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def calculate_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = calculate_ema(close, fast)
    ema_slow = calculate_ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_adx(
    df: pd.DataFrame,
    period: int = 14,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate ADX, DI+, DI-."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # True Range
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    dm_plus = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    dm_minus = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    # Smoothed
    atr = true_range.ewm(alpha=1 / period, adjust=False).mean()
    di_plus = 100 * dm_plus.ewm(alpha=1 / period, adjust=False).mean() / atr
    di_minus = 100 * dm_minus.ewm(alpha=1 / period, adjust=False).mean() / atr

    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    return adx, di_plus, di_minus


def detect_rsi_divergence(
    close: pd.Series,
    rsi: pd.Series,
    lookback: int = 10,
) -> int:
    """
    Detect RSI divergence.
    +1 = bullish divergence (price lower low, RSI higher low)
    -1 = bearish divergence (price higher high, RSI lower high)
    0  = no divergence
    """
    if len(close) < lookback:
        return 0

    price_slice = close.tail(lookback)
    rsi_slice = rsi.tail(lookback)

    price_min_idx = price_slice.idxmin()
    price_max_idx = price_slice.idxmax()

    # Bullish divergence: recent price low < previous low, RSI low > previous RSI low
    # Simplified: check if price made new low but RSI didn't
    if price_slice.iloc[-1] < price_slice.iloc[0]:
        if rsi_slice.iloc[-1] > rsi_slice.iloc[0]:
            return 1  # bullish divergence

    # Bearish divergence: recent price high > previous high, RSI high < previous RSI high
    if price_slice.iloc[-1] > price_slice.iloc[0]:
        if rsi_slice.iloc[-1] < rsi_slice.iloc[0]:
            return -1  # bearish divergence

    return 0


def analyze_momentum(
    df: pd.DataFrame,
    rsi_overbought: float = 70.0,
    rsi_oversold: float = 30.0,
    adx_strong: float = 25.0,
) -> MomentumResult:
    """
    Full momentum analysis.

    Args:
        df: OHLCV DataFrame (needs at least 200+ bars for EMA200)
        rsi_overbought: RSI threshold for overbought
        rsi_oversold: RSI threshold for oversold
        adx_strong: ADX threshold for strong trend
    """
    m = MomentumResult()
    close = df["close"]
    n = len(df)

    if n < 30:
        return m

    # ── EMAs ──────────────────────────────────────────────────────────────────
    ema20 = calculate_ema(close, 20)
    ema50 = calculate_ema(close, 50) if n >= 50 else None
    ema200 = calculate_ema(close, 200) if n >= 200 else None

    current_price = float(close.iloc[-1])
    m.ema20 = float(ema20.iloc[-1])

    if n >= 3:
        m.ema20_slope = (float(ema20.iloc[-1]) - float(ema20.iloc[-3])) / float(ema20.iloc[-3]) * 100
    m.price_vs_ema20 = (current_price - m.ema20) / m.ema20 * 100

    if ema50 is not None:
        m.ema50 = float(ema50.iloc[-1])
        if n >= 3:
            m.ema50_slope = (float(ema50.iloc[-1]) - float(ema50.iloc[-3])) / float(ema50.iloc[-3]) * 100
        m.price_vs_ema50 = (current_price - m.ema50) / m.ema50 * 100

    if ema200 is not None:
        m.ema200 = float(ema200.iloc[-1])
        m.price_vs_ema200 = (current_price - m.ema200) / m.ema200 * 100

    # ── RSI ────────────────────────────────────────────────────────────────────
    if n >= 15:
        rsi = calculate_rsi(close, 14)
        m.rsi = float(rsi.iloc[-1])
        m.rsi_overbought = m.rsi > rsi_overbought
        m.rsi_oversold = m.rsi < rsi_oversold
        m.rsi_divergence = detect_rsi_divergence(close, rsi)

    # ── MACD ───────────────────────────────────────────────────────────────────
    if n >= 35:
        macd_line, signal_line, histogram = calculate_macd(close)
        m.macd = float(macd_line.iloc[-1])
        m.macd_signal = float(signal_line.iloc[-1])
        m.macd_hist = float(histogram.iloc[-1])

        # Detect crossover
        if len(histogram) >= 2:
            prev_hist = float(histogram.iloc[-2])
            curr_hist = float(histogram.iloc[-1])
            if prev_hist <= 0 and curr_hist > 0:
                m.macd_cross = 1   # bullish cross
            elif prev_hist >= 0 and curr_hist < 0:
                m.macd_cross = -1  # bearish cross

    # ── ADX ────────────────────────────────────────────────────────────────────
    if n >= 20:
        adx, di_plus, di_minus = calculate_adx(df, 14)
        m.adx = float(adx.iloc[-1])
        m.di_plus = float(di_plus.iloc[-1])
        m.di_minus = float(di_minus.iloc[-1])
        m.trend_strong = m.adx > adx_strong

    return m
