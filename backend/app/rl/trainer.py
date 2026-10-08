"""
ZECRO-RL — PPO Policy Trainer
Walk-forward chronological training loop with out-of-sample validation and policy registration.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import structlog
import torch

from app.config import settings
from app.monitoring.performance import performance_calculator
from app.rl.environment import TradingEnvironment
from app.rl.policy_manager import policy_manager
from app.rl.ppo_agent import PPOAgent, RolloutBuffer

logger = structlog.get_logger(__name__)


class PPOTrainer:
    """
    PPO Training Runner.
    Trains policy networks on historical data with strict chronological train/val splits.
    """

    def __init__(
        self,
        df_1h: pd.DataFrame,
        symbol: str = "BTCUSD",
        train_split: float = 0.70,
        episode_length: int = 250,
        agent: Optional[PPOAgent] = None,
    ):
        self.df_1h = df_1h.copy().reset_index(drop=True)
        self.symbol = symbol
        self.episode_length = episode_length

        # Chronological train/validation split
        split_idx = int(len(self.df_1h) * train_split)
        self.train_df = self.df_1h.iloc[:split_idx].copy().reset_index(drop=True)
        self.val_df = self.df_1h.iloc[split_idx:].copy().reset_index(drop=True)

        logger.info(
            "Trainer initialized",
            symbol=symbol,
            train_candles=len(self.train_df),
            val_candles=len(self.val_df),
        )

        self.agent = agent or PPOAgent()
        self.train_env = TradingEnvironment(
            df_1h=self.train_df,
            symbol=symbol,
            episode_length=episode_length,
            random_start=True,
        )
        self.val_env = TradingEnvironment(
            df_1h=self.val_df,
            symbol=symbol,
            episode_length=len(self.val_df) - 250,
            random_start=False,
        )

    def train(
        self,
        total_episodes: int = 100,
        steps_per_update: int = 2048,
        eval_interval: int = 20,
    ) -> Dict[str, Any]:
        """
        Execute PPO training loop.
        """
        buffer = RolloutBuffer(state_dim=settings.rl_state_dim)
        obs, _ = self.train_env.reset()

        episode_rewards = []
        best_val_sharpe = -np.inf
        best_metrics: Dict[str, Any] = {}

        logger.info("Starting PPO training loop", total_episodes=total_episodes)

        step_counter = 0
        current_ep = 0

        while current_ep < total_episodes:
            for _ in range(steps_per_update):
                action, log_prob, value = self.agent.select_action(obs, deterministic=False)
                next_obs, reward, terminated, truncated, info = self.train_env.step(action)
                done = terminated or truncated

                buffer.add(obs, action, reward, value, log_prob, done)
                obs = next_obs
                step_counter += 1

                if done:
                    current_ep += 1
                    ep_pnl = info.get("total_pnl", 0)
                    episode_rewards.append(ep_pnl)
                    obs, _ = self.train_env.reset()
                    if current_ep >= total_episodes:
                        break

            # PPO Update step
            if len(buffer.states) > 0:
                self.agent.policy.eval()
                with torch.no_grad():
                    _, last_val = self.agent.policy(
                        torch.FloatTensor(obs).unsqueeze(0).to(self.agent.device)
                    )
                    last_value = last_val.item()
                self.agent.policy.train()

                train_metrics = self.agent.update(buffer, last_value)
                buffer.clear()

                logger.info(
                    "PPO Policy updated",
                    episode=current_ep,
                    policy_loss=f"{train_metrics.get('policy_loss', 0):.4f}",
                    value_loss=f"{train_metrics.get('value_loss', 0):.4f}",
                    entropy=f"{train_metrics.get('entropy', 0):.4f}",
                )

            # Periodic Evaluation
            if current_ep > 0 and current_ep % eval_interval == 0:
                val_metrics = self.evaluate()
                sharpe = val_metrics.get("sharpe_ratio", 0.0)
                logger.info(
                    "Validation check",
                    episode=current_ep,
                    sharpe=sharpe,
                    win_rate=f"{val_metrics.get('win_rate_pct', 0)}%",
                    max_dd=f"{val_metrics.get('max_drawdown_pct', 0)}%",
                    trades=val_metrics.get("total_trades", 0),
                )
                if sharpe > best_val_sharpe:
                    best_val_sharpe = sharpe
                    best_metrics = val_metrics

        # Final evaluation on validation set
        final_metrics = self.evaluate()

        return {
            "episodes_trained": current_ep,
            "final_metrics": final_metrics,
            "best_metrics": best_metrics if best_metrics else final_metrics,
        }

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate current policy deterministically on the out-of-sample validation slice.
        """
        obs, _ = self.val_env.reset()
        done = False
        trades_list = []

        while not done:
            action, _, _ = self.agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.val_env.step(action)
            done = terminated or truncated

        trades_list = self.val_env.trade_history
        metrics = performance_calculator.calculate_from_trades(
            trades=trades_list, initial_capital=self.val_env.initial_capital
        )

        return {
            "total_trades": metrics.total_trades,
            "win_rate_pct": metrics.win_rate_pct,
            "total_pnl_inr": metrics.total_pnl_inr,
            "return_pct": metrics.return_pct,
            "profit_factor": metrics.profit_factor,
            "sharpe_ratio": metrics.sharpe_ratio,
            "sortino_ratio": metrics.sortino_ratio,
            "max_drawdown_pct": metrics.max_drawdown_pct,
            "expectancy_r": metrics.expectancy_r,
        }
