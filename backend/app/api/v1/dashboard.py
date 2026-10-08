"""
ZECRO-RL — Dashboard API
Portfolio overview, equity, P&L, metrics.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Depends
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import PortfolioSnapshot, TradeExecution, PolicyVersion
from app.risk.risk_manager import risk_manager

router = APIRouter()


@router.get("/overview")
async def get_overview(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Main dashboard overview — equity, P&L, metrics."""
    stats = risk_manager.get_dashboard_stats()

    # Active policy
    result = await db.execute(
        select(PolicyVersion).where(PolicyVersion.is_active == True).limit(1)
    )
    active_policy = result.scalar_one_or_none()

    # Trade stats
    trade_result = await db.execute(
        select(
            func.count(TradeExecution.id).label("total"),
            func.sum(TradeExecution.net_pnl_inr).label("total_pnl"),
        ).where(TradeExecution.is_paper == True, TradeExecution.is_open == False)
    )
    trade_stats = trade_result.one()

    # Win rate
    win_result = await db.execute(
        select(func.count(TradeExecution.id)).where(
            TradeExecution.is_paper == True,
            TradeExecution.is_open == False,
            TradeExecution.net_pnl_inr > 0,
        )
    )
    wins = win_result.scalar() or 0
    total_trades = trade_stats.total or 0
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0.0

    return {
        "portfolio": {
            "initial_capital_inr": stats["initial_capital_inr"],
            "current_equity_inr": stats["current_equity_inr"],
            "available_inr": stats["available_inr"],
            "total_pnl_inr": stats["total_pnl_inr"],
            "return_pct": round(stats["return_pct"], 4),
            "daily_pnl_inr": stats["daily_pnl_inr"],
            "current_drawdown_pct": round(stats["current_drawdown_pct"], 4),
            "peak_equity_inr": stats["peak_equity_inr"],
        },
        "risk": {
            "open_positions": stats["open_positions"],
            "max_positions": stats["max_positions"],
            "risk_per_trade_pct": stats["risk_per_trade_pct"],
            "risk_per_trade_inr": stats["risk_per_trade_inr"],
            "max_daily_loss_inr": stats["max_daily_loss_inr"],
            "daily_loss_remaining_inr": stats["daily_loss_remaining_inr"],
        },
        "performance": {
            "total_trades": total_trades,
            "win_rate": round(win_rate, 2),
            "wins": wins,
            "losses": total_trades - wins,
        },
        "rl_policy": {
            "version": active_policy.version_name if active_policy else "None",
            "episodes": active_policy.training_episodes if active_policy else 0,
            "backtest_sharpe": active_policy.backtest_sharpe if active_policy else None,
            "is_active": True if active_policy else False,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/equity-curve")
async def get_equity_curve(
    days: int = 30,
    db: AsyncSession = Depends(get_db)
) -> List[Dict]:
    """Equity curve data for chart."""
    since = datetime.now(timezone.utc) - timedelta(days=days)
    result = await db.execute(
        select(PortfolioSnapshot)
        .where(PortfolioSnapshot.timestamp >= since)
        .order_by(PortfolioSnapshot.timestamp)
    )
    snapshots = result.scalars().all()

    return [
        {
            "time": int(s.timestamp.timestamp()),
            "equity": s.equity_inr,
            "pnl": s.total_pnl_inr,
            "return_pct": s.return_pct,
            "drawdown": s.drawdown_pct,
        }
        for s in snapshots
    ]


@router.get("/metrics")
async def get_metrics(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Performance metrics: Sharpe, Sortino, profit factor, etc."""
    result = await db.execute(
        select(TradeExecution).where(
            TradeExecution.is_paper == True,
            TradeExecution.is_open == False,
        ).order_by(TradeExecution.exit_time)
    )
    trades = result.scalars().all()

    if not trades:
        return {"message": "No closed trades yet"}

    import numpy as np

    pnls = [t.net_pnl_inr or 0 for t in trades]
    returns = [t.realized_r or 0 for t in trades]

    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r < 0]

    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = abs(sum(losses) / len(losses)) if losses else 0

    # Sharpe (simplified daily returns)
    if len(returns) > 1:
        returns_arr = np.array(returns)
        sharpe = (returns_arr.mean() / returns_arr.std()) * np.sqrt(252) if returns_arr.std() > 0 else 0
        sortino_neg = returns_arr[returns_arr < 0]
        sortino = (returns_arr.mean() / sortino_neg.std()) * np.sqrt(252) if len(sortino_neg) > 1 else 0
    else:
        sharpe = sortino = 0

    return {
        "total_trades": len(trades),
        "win_rate": len(wins) / len(returns) * 100 if returns else 0,
        "profit_factor": sum(wins) / max(abs(sum(losses)), 0.001),
        "avg_win_r": avg_win,
        "avg_loss_r": avg_loss,
        "expectancy_r": (len(wins) / len(returns) * avg_win) - (len(losses) / len(returns) * avg_loss) if returns else 0,
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "total_r": sum(returns),
        "avg_r": sum(returns) / len(returns) if returns else 0,
        "best_trade_r": max(returns) if returns else 0,
        "worst_trade_r": min(returns) if returns else 0,
    }
