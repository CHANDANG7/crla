"""
ZECRO-RL — Volume Analysis
Volume SMA, relative volume (RVOL), volume spikes, OBV, divergence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class VolumeResult:
    volume: float = 0.0
    volume_sma20: Optional[float] = None
    relative_volume: Optional[float] = None   # volume / SMA20
    volume_spike: bool = False                # RVOL > 2.0
    volume_trend: Optional[float] = None      # slope of last 5 bars
    obv: Optional[float] = None
    obv_slope: Optional[float] = None         # OBV trend direction
    volume_divergence_bull: bool = False      # price down, volume up = accumulation
    volume_divergence_bear: bool = False      # price up, volume down = distribution
    high_volume_bull: bool = False            # high volume bullish bar
    high_volume_bear: bool = False            # high volume bearish bar
    volume_delta: Optional[float] = None      # buying vs selling pressure (approx)


def calculate_obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume."""
    obv = [0.0]
    closes = df["close"].values
    volumes = df["volume"].values

    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            obv.append(obv[-1] + volumes[i])
        elif closes[i] < closes[i - 1]:
            obv.append(obv[-1] - volumes[i])
        else:
            obv.append(obv[-1])

    return pd.Series(obv, index=df.index)


def analyze_volume(
    df: pd.DataFrame,
    sma_period: int = 20,
    spike_threshold: float = 2.0,     # RVOL above this = spike
    divergence_lookback: int = 5,
) -> VolumeResult:
    """
    Full volume analysis.

    Args:
        df: OHLCV DataFrame
        sma_period: period for volume SMA
        spike_threshold: relative volume ratio for spike detection
        divergence_lookback: bars to check for divergence
    """
    v = VolumeResult()
    n = len(df)

    if n < 5:
        return v

    volume = df["volume"]
    close = df["close"]

    v.volume = float(volume.iloc[-1])

    # ── Volume SMA ─────────────────────────────────────────────────────────────
    if n >= sma_period:
        vol_sma = volume.rolling(sma_period).mean()
        v.volume_sma20 = float(vol_sma.iloc[-1])

        if v.volume_sma20 > 0:
            v.relative_volume = v.volume / v.volume_sma20
            v.volume_spike = v.relative_volume >= spike_threshold

    # ── Volume Trend (slope of last 5 bars) ─────────────────────────────────────
    if n >= 5:
        recent_vol = volume.tail(5).values.astype(float)
        if recent_vol.mean() > 0:
            # Simple linear regression slope normalized
            x = np.arange(len(recent_vol))
            slope = np.polyfit(x, recent_vol, 1)[0]
            v.volume_trend = slope / recent_vol.mean()  # normalized

    # ── OBV ────────────────────────────────────────────────────────────────────
    obv = calculate_obv(df)
    v.obv = float(obv.iloc[-1])

    if n >= 5:
        obv_recent = obv.tail(5)
        x = np.arange(len(obv_recent))
        v.obv_slope = float(np.polyfit(x, obv_recent.values, 1)[0])

    # ── Volume Divergence ─────────────────────────────────────────────────────
    if n >= divergence_lookback + 1:
        price_change = float(close.iloc[-1]) - float(close.iloc[-divergence_lookback])
        vol_change = float(volume.tail(divergence_lookback).mean()) - float(
            volume.iloc[-(divergence_lookback * 2):-divergence_lookback].mean()
        ) if n >= divergence_lookback * 2 else 0

        # Bullish divergence: price declining but volume increasing = accumulation
        if price_change < 0 and vol_change > 0:
            v.volume_divergence_bull = True

        # Bearish divergence: price rising but volume declining = distribution
        if price_change > 0 and vol_change < 0:
            v.volume_divergence_bear = True

    # ── High Volume Directional Bars ───────────────────────────────────────────
    if v.relative_volume is not None and v.relative_volume >= 1.5:
        current = df.iloc[-1]
        if current["close"] > current["open"]:
            v.high_volume_bull = True
        else:
            v.high_volume_bear = True

    # ── Volume Delta (approximate buying/selling pressure) ─────────────────────
    # Approximation: assume bullish bars = buying, bearish = selling
    if n >= 5:
        recent = df.tail(5)
        buy_vol = recent.loc[recent["close"] > recent["open"], "volume"].sum()
        sell_vol = recent.loc[recent["close"] <= recent["open"], "volume"].sum()
        total = buy_vol + sell_vol
        if total > 0:
            v.volume_delta = (buy_vol - sell_vol) / total  # -1 to +1

    return v
