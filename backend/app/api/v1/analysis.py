"""
ZECRO-RL — TA Analysis API
Endpoints for 1H & 15M market analyses and trade setups.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.llm.analysis_agent import groq_agent
from app.models import TradeSetup, MarketCandle
from app.ta.feature_engine import feature_engine

router = APIRouter()


class RunAnalysisRequest(BaseModel):
    symbol: str = "BTCUSD"
    timeframe: str = "1h"
    include_knowledge: bool = True


@router.get("/latest")
async def get_latest_analysis(
    symbol: Optional[str] = None,
    limit: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
) -> List[Dict[str, Any]]:
    """Get recent trade setups and LLM analyses."""
    query = select(TradeSetup).order_by(desc(TradeSetup.timestamp)).limit(limit)
    if symbol:
        query = query.where(TradeSetup.symbol == symbol)

    result = await db.execute(query)
    setups = result.scalars().all()

    return [
        {
            "id": s.id,
            "symbol": s.symbol,
            "timestamp": s.timestamp.isoformat() if s.timestamp else None,
            "bias": s.bias,
            "confidence": s.confidence,
            "setup_type": s.setup_type,
            "regime": s.regime,
            "llm_analysis": s.llm_analysis,
            "invalidation_condition": s.invalidation_condition,
            "is_valid": s.is_valid,
        }
        for s in setups
    ]


@router.get("/setups")
async def get_setups(
    symbol: Optional[str] = None,
    bias: Optional[str] = None,
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> List[Dict[str, Any]]:
    """Query recent trade setups with filtering."""
    query = select(TradeSetup).order_by(desc(TradeSetup.timestamp)).limit(limit)
    if symbol:
        query = query.where(TradeSetup.symbol == symbol)
    if bias:
        query = query.where(TradeSetup.bias == bias.upper())

    result = await db.execute(query)
    setups = result.scalars().all()

    return [
        {
            "id": s.id,
            "symbol": s.symbol,
            "timestamp": s.timestamp.isoformat() if s.timestamp else None,
            "timeframe_analysis": s.timeframe_analysis,
            "timeframe_entry": s.timeframe_entry,
            "bias": s.bias,
            "confidence": s.confidence,
            "setup_type": s.setup_type,
            "regime": s.regime,
            "llm_analysis": s.llm_analysis,
            "invalidation": s.invalidation_condition,
            "is_valid": s.is_valid,
        }
        for s in setups
    ]


@router.post("/run")
async def run_analysis(
    req: RunAnalysisRequest,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Trigger real-time Groq TA analysis for a symbol using recent candle data.
    """
    # Fetch recent candles for feature computation
    result = await db.execute(
        select(MarketCandle)
        .where(
            MarketCandle.symbol == req.symbol,
            MarketCandle.timeframe == req.timeframe,
        )
        .order_by(desc(MarketCandle.timestamp))
        .limit(200)
    )
    candles = result.scalars().all()

    if len(candles) < 20:
        # Fallback message if not enough stored candles
        return {
            "symbol": req.symbol,
            "timeframe": req.timeframe,
            "status": "warning",
            "message": f"Insufficient historical candles in database for {req.symbol} ({len(candles)} found, 20+ required). Run download_history.py first.",
            "analysis": {
                "bias": "NEUTRAL",
                "confidence": 0.0,
                "setup_type": "INSUFFICIENT_DATA",
                "market_regime": "UNKNOWN",
                "evidence": ["Waiting for historical candle data bootstrap"],
                "invalidation": "N/A",
                "risk_model": "0R",
            },
        }

    # Sort ascending for feature calculation
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

    # Compute deterministic market state
    state = feature_engine.compute_state(req.symbol, req.timeframe, ohlcv_data)

    # Run Groq 1H analysis
    analysis_output = await groq_agent.analyze_1h(state)

    # Save setup to DB
    setup = TradeSetup(
        symbol=req.symbol,
        timestamp=datetime.now(timezone.utc),
        timeframe_analysis=req.timeframe,
        timeframe_entry="15m",
        setup_type=analysis_output.setup_type,
        bias=analysis_output.bias,
        confidence=analysis_output.confidence,
        regime=analysis_output.market_regime,
        state_1h=state.to_dict(),
        llm_analysis=analysis_output.model_dump(),
        invalidation_condition=analysis_output.invalidation,
        is_valid=True,
    )
    db.add(setup)
    await db.commit()
    await db.refresh(setup)

    return {
        "setup_id": setup.id,
        "symbol": req.symbol,
        "timeframe": req.timeframe,
        "analysis": analysis_output.model_dump(),
        "market_state": state.to_dict(),
    }
