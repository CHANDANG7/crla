"""
ZECRO-RL — RL API
Policy versions, training stats, learning curves.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import PolicyVersion, AgentExperience

router = APIRouter()


@router.get("/policies")
async def get_policies(db: AsyncSession = Depends(get_db)):
    """Get all policy versions."""
    result = await db.execute(
        select(PolicyVersion).order_by(desc(PolicyVersion.created_at))
    )
    policies = result.scalars().all()
    return [
        {
            "version": p.version_name,
            "created_at": p.created_at.isoformat(),
            "training_episodes": p.training_episodes,
            "is_active": p.is_active,
            "backtest_sharpe": p.backtest_sharpe,
            "backtest_win_rate": p.backtest_win_rate,
            "backtest_max_dd": p.backtest_max_dd,
            "wf_sharpe": p.wf_sharpe,
            "paper_win_rate": p.paper_win_rate,
        }
        for p in policies
    ]


@router.get("/active-policy")
async def get_active_policy(db: AsyncSession = Depends(get_db)):
    """Get currently active policy."""
    result = await db.execute(
        select(PolicyVersion).where(PolicyVersion.is_active == True).limit(1)
    )
    policy = result.scalar_one_or_none()
    if not policy:
        return {"message": "No active policy"}
    return {
        "version": policy.version_name,
        "training_episodes": policy.training_episodes,
        "hyperparams": policy.hyperparams,
        "backtest_sharpe": policy.backtest_sharpe,
        "backtest_win_rate": policy.backtest_win_rate,
        "backtest_max_dd": policy.backtest_max_dd,
        "wf_sharpe": policy.wf_sharpe,
        "paper_win_rate": policy.paper_win_rate,
        "paper_total_trades": policy.paper_total_trades,
    }


@router.get("/experience-stats")
async def get_experience_stats(db: AsyncSession = Depends(get_db)):
    """Experience buffer statistics."""
    from sqlalchemy import func
    result = await db.execute(
        select(
            func.count(AgentExperience.id).label("total"),
            func.avg(AgentExperience.reward).label("avg_reward"),
            func.max(AgentExperience.reward).label("max_reward"),
            func.min(AgentExperience.reward).label("min_reward"),
        )
    )
    stats = result.one()
    return {
        "total_experiences": stats.total or 0,
        "avg_reward": float(stats.avg_reward or 0),
        "max_reward": float(stats.max_reward or 0),
        "min_reward": float(stats.min_reward or 0),
    }
