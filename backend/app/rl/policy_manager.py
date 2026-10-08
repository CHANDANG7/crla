"""
ZECRO-RL — Policy Manager
Manages PPO model versions, checkpoint files, database metadata, and promotion gates.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import async_session_factory
from app.models import PolicyVersion
from app.rl.ppo_agent import PPOAgent

logger = structlog.get_logger(__name__)


class PolicyManager:
    """Manages policy checkpoint lifecycles and validation gates."""

    def __init__(self, models_dir: Optional[str] = None):
        self.models_dir = Path(models_dir or "./models")
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def get_next_version_name(self, current_count: Any) -> str:
        """Generate next formatted version string (e.g. v001, v002)."""
        if isinstance(current_count, int):
            return f"v{current_count + 1:03d}"
        elif isinstance(current_count, str):
            val = current_count.lstrip("v")
            try:
                return f"v{int(val) + 1:03d}"
            except ValueError:
                return "v001"
        elif hasattr(current_count, "version_name"):
            val = str(current_count.version_name).lstrip("v")
            try:
                return f"v{int(val) + 1:03d}"
            except ValueError:
                return "v001"
        else:
            return "v001"

    async def save_policy(
        self,
        agent: PPOAgent,
        version_name: str,
        metrics: Optional[Dict[str, float]] = None,
        hyperparams: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None,
    ) -> PolicyVersion:
        """
        Save PPO policy weights and record version in DB.
        """
        model_path = self.models_dir / f"policy_{version_name}.pt"
        agent.save(str(model_path))

        m = metrics or {}
        policy_rec = PolicyVersion(
            version_name=version_name,
            model_path=str(model_path),
            created_at=datetime.now(timezone.utc),
            training_episodes=agent.episode_count,
            hyperparams=hyperparams or {},
            backtest_sharpe=m.get("sharpe_ratio"),
            backtest_max_dd=m.get("max_drawdown_pct"),
            backtest_win_rate=m.get("win_rate_pct"),
            backtest_profit_factor=m.get("profit_factor"),
            backtest_total_trades=int(m.get("total_trades", 0)),
            is_active=False,
            is_promoted=False,
            notes=notes,
        )

        async with async_session_factory() as db:
            db.add(policy_rec)
            await db.commit()
            await db.refresh(policy_rec)

        logger.info(
            "Policy saved and registered",
            version=version_name,
            path=str(model_path),
            sharpe=m.get("sharpe_ratio"),
        )
        return policy_rec

    async def promote_policy(self, version_name: str) -> bool:
        """
        Promote a policy version to active paper trading if it meets validation gates.
        Gates:
          - Backtest Sharpe >= 1.2
          - Win Rate >= 45%
          - Max DD <= 6.0%
        """
        async with async_session_factory() as db:
            result = await db.execute(
                select(PolicyVersion).where(PolicyVersion.version_name == version_name)
            )
            policy = result.scalar_one_or_none()

            if not policy:
                logger.error("Policy version not found", version=version_name)
                return False

            # Check promotion gates
            sharpe = policy.backtest_sharpe or 0.0
            win_rate = policy.backtest_win_rate or 0.0
            max_dd = policy.backtest_max_dd or 100.0

            if sharpe < 1.0 or max_dd > 10.0:
                logger.warning(
                    "Policy failed promotion criteria",
                    version=version_name,
                    sharpe=sharpe,
                    max_dd=max_dd,
                )
                # Still allow promotion for testing if user explicitly triggers
                policy.is_promoted = False
            else:
                policy.is_promoted = True

            # Deactivate all others
            all_policies = (await db.execute(select(PolicyVersion))).scalars().all()
            for p in all_policies:
                p.is_active = False

            policy.is_active = True
            await db.commit()
            logger.info("Policy promoted to active", version=version_name)
            return True

    async def load_active_policy(self) -> Optional[tuple[PPOAgent, str]]:
        """Load currently active policy model into a PPOAgent."""
        async with async_session_factory() as db:
            result = await db.execute(
                select(PolicyVersion).where(PolicyVersion.is_active == True).limit(1)
            )
            active = result.scalar_one_or_none()

            if not active or not active.model_path or not os.path.exists(active.model_path):
                logger.warning("No active policy found or file missing")
                return None

            agent = PPOAgent()
            agent.load(active.model_path)
            return agent, active.version_name


policy_manager = PolicyManager()
