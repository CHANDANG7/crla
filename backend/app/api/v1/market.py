"""
ZECRO-RL — Market Data API
Endpoints for candle data, technical features, symbols, and regimes.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models import MarketCandle, MarketRegime, TechnicalFeature
from app.ta.feature_engine import feature_engine

router = APIRouter()


@router.get("/symbols")
async def get_symbols() -> Dict[str, Any]:
    """List of active trading symbols and timeframes."""
    return {
        "symbols": settings.symbols,
        "default_symbol": settings.default_symbol,
        "analysis_timeframe": settings.timeframe_analysis,
        "entry_timeframe": settings.timeframe_entry,
        "all_timeframes": settings.timeframes_all,
    }


@router.get("/candles")
async def get_candles(
    symbol: str = "BTCUSD",
    timeframe: str = "1h",
    limit: int = Query(200, ge=10, le=1000),
    db: AsyncSession = Depends(get_db),
) -> List[Dict[str, Any]]:
    """Get candlestick data for TradingView chart."""
    result = await db.execute(
        select(MarketCandle)
        .where(
            MarketCandle.symbol == symbol,
            MarketCandle.timeframe == timeframe,
        )
        .order_by(desc(MarketCandle.timestamp))
        .limit(limit)
    )
    candles = result.scalars().all()
    # Return chronologically ascending
    candles_asc = sorted(candles, key=lambda c: c.timestamp)

    return [
        {
            "time": int(c.timestamp.timestamp()),
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume,
        }
        for c in candles_asc
    ]


@router.get("/features")
async def get_features(
    symbol: str = "BTCUSD",
    timeframe: str = "1h",
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Get the latest computed 160-dim TA features for a symbol."""
    result = await db.execute(
        select(MarketCandle)
        .where(
            MarketCandle.symbol == symbol,
            MarketCandle.timeframe == timeframe,
        )
        .order_by(desc(MarketCandle.timestamp))
        .limit(200)
    )
    candles = result.scalars().all()

    if len(candles) < 20:
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "status": "insufficient_data",
            "message": f"Found {len(candles)} candles (20+ needed).",
            "state": None,
        }

    candles_asc = sorted(candles, key=lambda c: c.timestamp)
    ohlcv_data = [
        {
            "timestamp": c.timestamp,
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume,
        }
        for c in candles_asc
    ]

    state = feature_engine.compute_state(symbol, timeframe, ohlcv_data)
    rl_vector = feature_engine.to_rl_vector(state)

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "status": "ok",
        "timestamp": candles_asc[-1].timestamp.isoformat(),
        "state": state.to_dict(),
        "rl_vector_dim": len(rl_vector),
        "rl_vector_sample": rl_vector[:10].tolist(),
    }


@router.get("/regimes")
async def get_regimes(
    symbol: Optional[str] = None,
    limit: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
) -> List[Dict[str, Any]]:
    """Get latest detected market regimes."""
    query = select(MarketRegime).order_by(desc(MarketRegime.timestamp)).limit(limit)
    if symbol:
        query = query.where(MarketRegime.symbol == symbol)

    result = await db.execute(query)
    regimes = result.scalars().all()

    return [
        {
            "id": r.id,
            "symbol": r.symbol,
            "timestamp": r.timestamp.isoformat(),
            "regime": r.regime,
            "confidence": r.confidence,
            "regime_data": r.regime_data,
        }
        for r in regimes
    ]
