"""
ZECRO-RL — Trading Environment (Gym-compatible)
The heart of the RL system. Simulates realistic paper trading for PPO training.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
import structlog

from app.config import settings
from app.ta.feature_engine import FeatureEngine, MarketState

logger = structlog.get_logger(__name__)

# ── Action constants ───────────────────────────────────────────────────────────
ACTION_HOLD  = 0
ACTION_LONG  = 1
ACTION_SHORT = 2
ACTION_EXIT  = 3

ACTION_NAMES = {0: "HOLD", 1: "LONG", 2: "SHORT", 3: "EXIT"}


@dataclass
class Position:
    """Open position tracking."""
    symbol: str
    direction: str           # "LONG" or "SHORT"
    entry_price: float
    stop_loss: float
    tp1: float
    tp2: float
    size: float              # normalized (1.0 = full risk unit)
    risk_r: float            # 1.0 = 1R
    entry_step: int

    # Partial TP
    tp1_hit: bool = False
    tp2_hit: bool = False
    remaining_size: float = 1.0
    sl_moved_to_be: bool = False

    # Trailing stop
    tsl_active: bool = False
    tsl_price: Optional[float] = None
    tsl_pct: float = 0.005  # 0.5% trailing

    # P&L tracking
    unrealized_r: float = 0.0
    peak_unrealized_r: float = 0.0


class TradingEnvironment(gym.Env):
    """
    Gym-compatible trading environment for PPO training.

    State: 160-dimensional feature vector from FeatureEngine
    Action: HOLD (0), LONG (1), SHORT (2), EXIT (3)
    Reward: Risk-normalized R value

    Realistic simulation includes:
    - Maker/taker fees
    - Slippage model
    - Partial TP (50% at TP1, 25% at TP2, 25% trailing)
    - Position sizing via ATR / risk per trade
    - Daily loss limits
    - Max drawdown limits
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df_1h: pd.DataFrame,
        df_15m: Optional[pd.DataFrame] = None,
        symbol: str = "BTCUSD",
        initial_capital: float = settings.initial_capital_inr,
        risk_per_trade: float = settings.risk_per_trade,
        max_daily_loss: float = settings.max_daily_loss_inr,
        max_dd: float = settings.max_portfolio_dd,
        taker_fee: float = settings.taker_fee,
        slippage_bps: float = settings.slippage_bps,
        episode_length: int = 200,
        random_start: bool = True,
    ):
        super().__init__()

        self.df_1h = df_1h.copy().reset_index(drop=True)
        self.df_15m = df_15m
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_daily_loss = max_daily_loss
        self.max_dd = max_dd
        self.taker_fee = taker_fee
        self.slippage_bps = slippage_bps / 10000  # convert bps to decimal
        self.episode_length = episode_length
        self.random_start = random_start

        # Gym spaces
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(settings.rl_state_dim,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(settings.rl_action_dim)

        # Feature engine
        self.feature_engine = FeatureEngine()

        # Minimum bars required before first step
        self.min_bars = 220  # EMA200 needs 200 bars

        # State tracking
        self._reset_state()

    def _reset_state(self):
        """Reset all episode state."""
        self.current_step = 0
        self.episode_start = 0
        self.equity = self.initial_capital
        self.peak_equity = self.initial_capital
        self.available = self.initial_capital
        self.position: Optional[Position] = None
        self.trade_history: List[Dict] = []
        self.daily_pnl = 0.0
        self.daily_start_equity = self.initial_capital
        self.total_trades = 0
        self.winning_trades = 0
        self.step_count = 0
        self._done = False
        self._last_state: Optional[np.ndarray] = None
        self._last_market_state: Optional[MarketState] = None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self._reset_state()

        # Random start for diversity during training
        max_start = len(self.df_1h) - self.episode_length - self.min_bars
        if self.random_start and max_start > 0:
            self.episode_start = random.randint(self.min_bars, max(self.min_bars, max_start))
        else:
            self.episode_start = self.min_bars

        self.current_step = self.episode_start

        obs = self._get_observation()
        self._last_state = obs
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the trading environment.

        Args:
            action: 0=HOLD, 1=LONG, 2=SHORT, 3=EXIT

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        if self._done:
            raise RuntimeError("Environment is done. Call reset() first.")

        self.step_count += 1
        current_bar = self.df_1h.iloc[self.current_step]
        current_price = float(current_bar["close"])
        current_high = float(current_bar["high"])
        current_low = float(current_bar["low"])

        reward = 0.0
        info: Dict[str, Any] = {
            "action": ACTION_NAMES[action],
            "price": current_price,
            "equity": self.equity,
            "step": self.current_step,
        }

        # ── Manage existing position ─────────────────────────────────────────
        if self.position is not None:
            position_reward, closed = self._manage_position(
                current_high, current_low, current_price
            )
            reward += position_reward

            if closed:
                info["trade_closed"] = True

        # ── Execute new action ────────────────────────────────────────────────
        action_reward = 0.0

        if action == ACTION_LONG and self.position is None:
            action_reward = self._open_position("LONG", current_price, current_bar)

        elif action == ACTION_SHORT and self.position is None:
            action_reward = self._open_position("SHORT", current_price, current_bar)

        elif action == ACTION_EXIT and self.position is not None:
            action_reward = self._close_position(current_price, "MANUAL")

        elif action in (ACTION_LONG, ACTION_SHORT) and self.position is not None:
            # Penalize overtrading (trying to trade while in position)
            action_reward = -0.05

        reward += action_reward

        # ── Update step ────────────────────────────────────────────────────────
        self.current_step += 1

        # ── Check termination ──────────────────────────────────────────────────
        terminated = False
        truncated = False

        # Episode end
        episode_done = (self.current_step - self.episode_start) >= self.episode_length
        data_end = self.current_step >= len(self.df_1h) - 1

        # Risk limits
        dd = (self.peak_equity - self.equity) / self.peak_equity
        daily_loss = self.daily_pnl
        risk_limit_hit = (
            dd >= self.max_dd or
            daily_loss <= -self.max_daily_loss
        )

        if risk_limit_hit:
            reward -= 1.0  # large penalty for hitting risk limits
            terminated = True
            info["terminated_reason"] = "risk_limit"
        elif data_end or episode_done:
            truncated = True

        # Force close position at end of episode
        if (terminated or truncated) and self.position is not None:
            close_reward = self._close_position(current_price, "EPISODE_END")
            reward += close_reward

        self._done = terminated or truncated

        # Clip reward
        reward = float(np.clip(reward, -3.0, 3.0))

        # Get next observation
        if not self._done:
            obs = self._get_observation()
        else:
            obs = self._last_state if self._last_state is not None else np.zeros(160, dtype=np.float32)

        self._last_state = obs

        info.update({
            "equity": self.equity,
            "drawdown": dd,
            "total_trades": self.total_trades,
            "win_rate": self.winning_trades / max(self.total_trades, 1),
        })

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Build RL observation vector from current bars."""
        try:
            # Slice DataFrame up to current step (no look-ahead)
            df_slice = self.df_1h.iloc[max(0, self.current_step - 250): self.current_step + 1]

            if len(df_slice) < 30:
                return np.zeros(settings.rl_state_dim, dtype=np.float32)

            state: MarketState = self.feature_engine.compute_state(
                df_1h=df_slice,
                symbol=self.symbol,
            )
            self._last_market_state = state
            return state.to_rl_vector()

        except Exception as e:
            logger.warning("Observation error", error=str(e))
            return np.zeros(settings.rl_state_dim, dtype=np.float32)

    def _open_position(
        self, direction: str, price: float, bar: pd.Series
    ) -> float:
        """
        Open a new position. Returns immediate reward (cost).
        Uses ATR-based SL placement.
        """
        # ATR-based SL from recent bars
        recent = self.df_1h.iloc[max(0, self.current_step - 20): self.current_step + 1]
        atr = self._quick_atr(recent)

        sl_distance = atr * 1.5  # SL = 1.5 ATR away

        if direction == "LONG":
            stop_loss = price - sl_distance
            tp1 = price + sl_distance * 1.0   # 1R
            tp2 = price + sl_distance * 2.0   # 2R
        else:  # SHORT
            stop_loss = price + sl_distance
            tp1 = price - sl_distance * 1.0
            tp2 = price - sl_distance * 2.0

        # Position size based on risk
        risk_inr = self.equity * self.risk_per_trade
        size = risk_inr / sl_distance  # units

        # Apply entry fees + slippage
        slippage = price * self.slippage_bps
        actual_entry = price + slippage if direction == "LONG" else price - slippage
        fee_cost = risk_inr * self.taker_fee
        self.equity -= fee_cost

        self.position = Position(
            symbol=self.symbol,
            direction=direction,
            entry_price=actual_entry,
            stop_loss=stop_loss,
            tp1=tp1,
            tp2=tp2,
            size=size,
            risk_r=risk_inr,
            entry_step=self.current_step,
        )

        self.total_trades += 1

        # Small negative reward for transaction cost
        return -(fee_cost / max(risk_inr, 1)) * 0.1

    def _manage_position(
        self, high: float, low: float, close: float
    ) -> Tuple[float, bool]:
        """
        Check and execute SL, TP1, TP2, trailing stop.
        Returns (reward, position_closed).
        """
        pos = self.position
        reward = 0.0
        closed = False

        if pos.direction == "LONG":
            # Check Stop Loss
            if low <= pos.stop_loss:
                reward = self._close_position(pos.stop_loss, "SL")
                return reward, True

            # Check TP1
            if not pos.tp1_hit and high >= pos.tp1:
                # Close 50% at TP1
                tp1_reward = settings.reward_tp1 * settings.tp1_portion
                pos.remaining_size -= settings.tp1_portion
                pos.tp1_hit = True
                pos.sl_moved_to_be = True
                pos.stop_loss = pos.entry_price  # move SL to breakeven
                reward += tp1_reward
                pnl = (pos.tp1 - pos.entry_price) * pos.size * settings.tp1_portion
                self.equity += pnl

            # Check TP2
            if pos.tp1_hit and not pos.tp2_hit and high >= pos.tp2:
                tp2_reward = settings.reward_tp2 * settings.tp2_portion
                pos.remaining_size -= settings.tp2_portion
                pos.tp2_hit = True
                pos.tsl_active = True
                reward += tp2_reward
                pnl = (pos.tp2 - pos.entry_price) * pos.size * settings.tp2_portion
                self.equity += pnl

            # Trailing stop on remaining
            if pos.tp2_hit and pos.remaining_size > 0:
                tsl_trigger = close * (1 - pos.tsl_pct)
                if pos.tsl_price is None or tsl_trigger > pos.tsl_price:
                    pos.tsl_price = tsl_trigger

                if low <= pos.tsl_price:
                    tsl_reward = ((pos.tsl_price - pos.entry_price) /
                                  (pos.tp1 - pos.entry_price)) * pos.remaining_size
                    reward += max(settings.reward_tsl, tsl_reward)
                    pnl = (pos.tsl_price - pos.entry_price) * pos.size * pos.remaining_size
                    self.equity += pnl
                    self.position = None
                    closed = True

        else:  # SHORT
            # Check Stop Loss
            if high >= pos.stop_loss:
                reward = self._close_position(pos.stop_loss, "SL")
                return reward, True

            # Check TP1
            if not pos.tp1_hit and low <= pos.tp1:
                tp1_reward = settings.reward_tp1 * settings.tp1_portion
                pos.remaining_size -= settings.tp1_portion
                pos.tp1_hit = True
                pos.sl_moved_to_be = True
                pos.stop_loss = pos.entry_price
                reward += tp1_reward
                pnl = (pos.entry_price - pos.tp1) * pos.size * settings.tp1_portion
                self.equity += pnl

            # Check TP2
            if pos.tp1_hit and not pos.tp2_hit and low <= pos.tp2:
                tp2_reward = settings.reward_tp2 * settings.tp2_portion
                pos.remaining_size -= settings.tp2_portion
                pos.tp2_hit = True
                pos.tsl_active = True
                reward += tp2_reward
                pnl = (pos.entry_price - pos.tp2) * pos.size * settings.tp2_portion
                self.equity += pnl

            # Trailing stop
            if pos.tp2_hit and pos.remaining_size > 0:
                tsl_trigger = close * (1 + pos.tsl_pct)
                if pos.tsl_price is None or tsl_trigger < pos.tsl_price:
                    pos.tsl_price = tsl_trigger

                if high >= pos.tsl_price:
                    tsl_reward = ((pos.entry_price - pos.tsl_price) /
                                  (pos.entry_price - pos.tp1)) * pos.remaining_size
                    reward += max(settings.reward_tsl, tsl_reward)
                    pnl = (pos.entry_price - pos.tsl_price) * pos.size * pos.remaining_size
                    self.equity += pnl
                    self.position = None
                    closed = True

        # Update peak equity and drawdown
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        # Update unrealized P&L tracking
        if self.position and not closed:
            if pos.direction == "LONG":
                pos.unrealized_r = (close - pos.entry_price) / (pos.tp1 - pos.entry_price)
            else:
                pos.unrealized_r = (pos.entry_price - close) / (pos.entry_price - pos.tp1)
            pos.peak_unrealized_r = max(pos.peak_unrealized_r, pos.unrealized_r)

        return reward, closed

    def _close_position(self, price: float, reason: str) -> float:
        """Force close position. Returns reward."""
        pos = self.position
        if pos is None:
            return 0.0

        sl_distance = abs(pos.entry_price - pos.stop_loss)
        if sl_distance == 0:
            self.position = None
            return 0.0

        if pos.direction == "LONG":
            pnl = (price - pos.entry_price) * pos.size * pos.remaining_size
        else:
            pnl = (pos.entry_price - price) * pos.size * pos.remaining_size

        # Fees on exit
        fee = abs(pnl) * self.taker_fee
        net_pnl = pnl - fee
        self.equity += net_pnl
        self.daily_pnl += net_pnl

        # R-normalized reward
        r_value = pnl / max(pos.risk_r, 1)
        reward = float(np.clip(r_value, -1.5, 2.0))

        if pnl > 0:
            self.winning_trades += 1

        trade_record = {
            "direction": pos.direction,
            "entry": pos.entry_price,
            "exit": price,
            "reason": reason,
            "pnl_r": r_value,
            "tp1_hit": pos.tp1_hit,
            "tp2_hit": pos.tp2_hit,
        }
        self.trade_history.append(trade_record)

        self.position = None
        return reward

    def _quick_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Quick ATR calculation."""
        if len(df) < 2:
            return float(df["high"].iloc[-1] - df["low"].iloc[-1]) if len(df) else 100.0

        high = df["high"].values
        low = df["low"].values
        close = df["close"].values

        trs = []
        for i in range(1, len(high)):
            trs.append(max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            ))
        return float(np.mean(trs[-period:])) if trs else 100.0

    def render(self, mode="human"):
        pos_str = "NONE"
        if self.position:
            pos_str = f"{self.position.direction} @ {self.position.entry_price:.2f}"
        print(
            f"Step: {self.current_step} | "
            f"Equity: ₹{self.equity:,.0f} | "
            f"Position: {pos_str} | "
            f"Trades: {self.total_trades} | "
            f"Win rate: {self.winning_trades}/{self.total_trades}"
        )

    def get_episode_stats(self) -> Dict[str, Any]:
        """Get complete episode statistics."""
        total = len(self.trade_history)
        wins = sum(1 for t in self.trade_history if t["pnl_r"] > 0)
        pnl_rs = [t["pnl_r"] for t in self.trade_history]
        total_r = sum(pnl_rs)

        gross_profit = sum(r for r in pnl_rs if r > 0)
        gross_loss = abs(sum(r for r in pnl_rs if r < 0))

        return {
            "total_trades": total,
            "win_rate": wins / max(total, 1),
            "total_r": total_r,
            "avg_r": total_r / max(total, 1),
            "profit_factor": gross_profit / max(gross_loss, 0.001),
            "final_equity": self.equity,
            "return_pct": (self.equity - self.initial_capital) / self.initial_capital * 100,
            "max_drawdown": (self.peak_equity - self.equity) / self.peak_equity * 100,
        }
