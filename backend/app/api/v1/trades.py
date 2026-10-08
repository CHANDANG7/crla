"""
ZECRO-RL — Trades API
Trade history, journal, open positions.
"""
from __future__ import annotations

from typing import List, Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import TradeExecution, TradeSetup

router = APIRouter()


@router.get("/open")
async def get_open_trades(db: AsyncSession = Depends(get_db)):
    """Get all currently open paper trades."""
    result = await db.execute(
        select(TradeExecution)
        .where(TradeExecution.is_open == True, TradeExecution.is_paper == True)
        .order_by(desc(TradeExecution.entry_time))
    )
    trades = result.scalars().all()
    return [_trade_to_dict(t) for t in trades]


@router.get("/history")
async def get_trade_history(
    limit: int = Query(50, le=500),
    offset: int = 0,
    symbol: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """Get closed trade history with pagination."""
    query = select(TradeExecution).where(
        TradeExecution.is_open == False,
        TradeExecution.is_paper == True,
    )
    if symbol:
        query = query.where(TradeExecution.symbol == symbol)

    query = query.order_by(desc(TradeExecution.exit_time)).limit(limit).offset(offset)
    result = await db.execute(query)
    trades = result.scalars().all()
    return [_trade_to_dict(t) for t in trades]


@router.get("/{trade_id}")
async def get_trade(trade_id: str, db: AsyncSession = Depends(get_db)):
    """Get a specific trade by ID (full journal record)."""
    result = await db.execute(
        select(TradeExecution).where(TradeExecution.id == trade_id)
    )
    trade = result.scalar_one_or_none()
    if not trade:
        from fastapi import HTTPException
        raise HTTPException(404, "Trade not found")

    trade_dict = _trade_to_dict(trade)

    # Get the setup
    if trade.setup_id:
        setup_result = await db.execute(
            select(TradeSetup).where(TradeSetup.id == trade.setup_id)
        )
        setup = setup_result.scalar_one_or_none()
        if setup:
            trade_dict["setup"] = {
                "setup_type": setup.setup_type,
                "bias": setup.bias,
                "confidence": setup.confidence,
                "llm_analysis": setup.llm_analysis,
                "state_1h": setup.state_1h,
            }

    return trade_dict


def _trade_to_dict(t: TradeExecution) -> dict:
    return {
        "trade_id": t.id,
        "symbol": t.symbol,
        "action": t.action,
        "entry_price": t.entry_price,
        "stop_loss": t.stop_loss,
        "tp1": t.tp1,
        "tp2": t.tp2,
        "position_size": t.position_size,
        "risk_amount_inr": t.risk_amount_inr,
        "entry_time": t.entry_time.isoformat() if t.entry_time else None,
        "exit_time": t.exit_time.isoformat() if t.exit_time else None,
        "exit_price": t.exit_price,
        "exit_reason": t.exit_reason,
        "tp1_filled": t.tp1_filled,
        "tp2_filled": t.tp2_filled,
        "tsl_active": t.tsl_active,
        "sl_moved_to_be": t.sl_moved_to_be,
        "realized_pnl_inr": t.realized_pnl_inr,
        "realized_r": t.realized_r,
        "net_pnl_inr": t.net_pnl_inr,
        "fees_inr": t.fees_inr,
        "market_regime": t.market_regime,
        "policy_version": t.policy_version,
        "is_open": t.is_open,
    }
