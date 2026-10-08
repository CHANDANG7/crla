"""
ZECRO-RL — Reward Engine
Computes R-normalized, risk-adjusted rewards for RL policy optimization.
Formula:
  Reward = Realized_R + (0.1 * Peak_R) - (Cost_R) - Drawdown_Penalty - Overtrading_Penalty
  Clipped to [-3.0, +3.0]
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class RewardBreakdown:
    realized_r: float
    peak_r_bonus: float
    cost_penalty_r: float
    drawdown_penalty: float
    overtrading_penalty: float
    invalid_action_penalty: float
    total_reward: float


class RewardEngine:
    """Calculates granular RL step and episode rewards."""

    @staticmethod
    def compute_trade_reward(
        realized_r: float,
        peak_unrealized_r: float = 0.0,
        transaction_cost_r: float = 0.0,
        drawdown_pct: float = 0.0,
        max_drawdown_limit_pct: float = 10.0,
        trades_count: int = 1,
        max_trades_allowed: int = 50,
        invalid_action: bool = False,
    ) -> RewardBreakdown:
        """
        Compute institutional R-normalized reward.
        """
        # 1. Base Realized R multiple (-1.0 on full SL, +1.5 on TP1, etc.)
        base_r = realized_r

        # 2. Peak reward encouragement (reward letting winners run)
        peak_bonus = 0.10 * max(0.0, peak_unrealized_r)

        # 3. Transaction costs (taker fees + slippage normalized to R)
        cost_penalty = transaction_cost_r

        # 4. Drawdown penalty (exponential penalty if near DD limit)
        dd_ratio = drawdown_pct / max(max_drawdown_limit_pct, 1.0)
        dd_penalty = 0.5 * (dd_ratio ** 2) if dd_ratio > 0.5 else 0.0

        # 5. Overtrading penalty
        overtrading = 0.2 if trades_count > max_trades_allowed else 0.0

        # 6. Invalid action penalty (e.g. attempting to enter when position is already open)
        invalid_penalty = 0.5 if invalid_action else 0.0

        raw_total = (
            base_r + peak_bonus - cost_penalty - dd_penalty - overtrading - invalid_penalty
        )
        clipped_total = float(np.clip(raw_total, -3.0, 3.0))

        return RewardBreakdown(
            realized_r=round(base_r, 4),
            peak_r_bonus=round(peak_bonus, 4),
            cost_penalty_r=round(cost_penalty, 4),
            drawdown_penalty=round(dd_penalty, 4),
            overtrading_penalty=round(overtrading, 4),
            invalid_action_penalty=round(invalid_penalty, 4),
            total_reward=round(clipped_total, 4),
        )


reward_engine = RewardEngine()
