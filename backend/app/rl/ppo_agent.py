"""
ZECRO-RL — PPO Policy Network + Agent
PyTorch implementation with separate policy and value heads.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import structlog

from app.config import settings

logger = structlog.get_logger(__name__)

# Actions
ACTIONS = {0: "HOLD", 1: "LONG", 2: "SHORT", 3: "EXIT"}


class PPOPolicyNetwork(nn.Module):
    """
    Shared backbone with separate Policy (Actor) and Value (Critic) heads.

    Architecture:
        Input (160) → Dense(256) → LN → ReLU → Dropout
                    → Dense(128) → LN → ReLU → Dropout
                    → Dense(64)  → LN → ReLU
                    → Policy Head: Dense(4) → Softmax
                    → Value Head:  Dense(1)
    """

    def __init__(
        self,
        state_dim: int = settings.rl_state_dim,
        action_dim: int = settings.rl_action_dim,
        hidden_dims: Tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.1,
    ):
        super().__init__()

        # ── Shared backbone ────────────────────────────────────────────────────
        layers = []
        in_dim = state_dim
        for i, h_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout if i < len(hidden_dims) - 1 else 0.0),
            ])
            in_dim = h_dim

        self.backbone = nn.Sequential(*layers)

        # ── Policy Head (Actor) ────────────────────────────────────────────────
        self.policy_head = nn.Linear(hidden_dims[-1], action_dim)

        # ── Value Head (Critic) ────────────────────────────────────────────────
        self.value_head = nn.Linear(hidden_dims[-1], 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)

        # Policy head: smaller initialization for stability
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        # Value head: small initialization
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            state: (batch_size, state_dim) tensor

        Returns:
            (action_logits, state_value)
        """
        features = self.backbone(state)
        action_logits = self.policy_head(features)
        state_value = self.value_head(features)
        return action_logits, state_value

    def get_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        """
        Get action from current policy.

        Args:
            state: numpy state vector
            deterministic: if True, take argmax (greedy)

        Returns:
            (action_int, log_prob, state_value)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            logits, value = self.forward(state_tensor)

            probs = F.softmax(logits, dim=-1)

            if deterministic:
                action = int(probs.argmax(dim=-1).item())
                log_prob = float(torch.log(probs[0, action]).item())
            else:
                dist = torch.distributions.Categorical(probs)
                action_tensor = dist.sample()
                action = int(action_tensor.item())
                log_prob = float(dist.log_prob(action_tensor).item())

            return action, log_prob, float(value.item())

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.

        Returns:
            (log_probs, state_values, entropy)
        """
        logits, values = self.forward(states)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, values.squeeze(-1), entropy


class PPOAgent:
    """
    PPO (Proximal Policy Optimization) agent.
    Compatible with the TradingEnvironment.
    """

    def __init__(
        self,
        state_dim: int = settings.rl_state_dim,
        action_dim: int = settings.rl_action_dim,
        learning_rate: float = settings.rl_learning_rate,
        gamma: float = settings.rl_gamma,
        clip_epsilon: float = settings.rl_clip_epsilon,
        entropy_coef: float = settings.rl_entropy_coef,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = settings.rl_n_epochs,
        batch_size: int = settings.rl_batch_size,
        device: str = "auto",
    ):
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # Device selection
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info("PPO Agent initialized", device=str(self.device))

        # Policy network
        self.policy = PPOPolicyNetwork(state_dim, action_dim).to(self.device)

        # Optimizer with gradient clipping
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=learning_rate,
            eps=1e-5,
        )

        # Training metrics
        self.training_step = 0
        self.episode_count = 0

    def predict(
        self, state: np.ndarray, deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """Get action from policy."""
        return self.policy.get_action(state, deterministic=deterministic)

    def select_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """Alias for predict / get_action."""
        return self.predict(state, deterministic=deterministic)

    def update(
        self,
        rollout_buffer: "RolloutBuffer",
        last_value: float = 0.0,
    ) -> Dict[str, float]:
        """
        PPO update from collected rollout.

        Args:
            rollout_buffer: collected experience buffer

        Returns:
            Training metrics dict
        """
        # Compute advantages
        advantages = rollout_buffer.compute_advantages(self.gamma)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss_list = []
        policy_loss_list = []
        value_loss_list = []
        entropy_list = []
        clip_fraction_list = []

        # Convert to tensors
        states = torch.FloatTensor(rollout_buffer.states).to(self.device)
        actions = torch.LongTensor(rollout_buffer.actions).to(self.device)
        old_log_probs = torch.FloatTensor(rollout_buffer.log_probs).to(self.device)
        returns = torch.FloatTensor(rollout_buffer.returns).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)

        # Multiple epochs of PPO updates
        for epoch in range(self.n_epochs):
            # Mini-batch updates
            indices = np.random.permutation(len(states))

            for start in range(0, len(states), self.batch_size):
                batch_idx = indices[start: start + self.batch_size]
                if len(batch_idx) < 4:
                    continue

                # Evaluate actions under current policy
                log_probs, values, entropy = self.policy.evaluate_actions(
                    states[batch_idx], actions[batch_idx]
                )

                # PPO clip ratio
                ratio = torch.exp(log_probs - old_log_probs[batch_idx])
                adv_batch = advantages_tensor[batch_idx]

                # Clipped surrogate objective
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv_batch
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value function loss
                value_loss = F.mse_loss(values, returns[batch_idx])

                # Entropy bonus (encourage exploration)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Track metrics
                clip_fraction = ((ratio - 1).abs() > self.clip_epsilon).float().mean()
                total_loss_list.append(loss.item())
                policy_loss_list.append(policy_loss.item())
                value_loss_list.append(value_loss.item())
                entropy_list.append(-entropy_loss.item())
                clip_fraction_list.append(clip_fraction.item())

        self.training_step += 1

        return {
            "total_loss": np.mean(total_loss_list),
            "policy_loss": np.mean(policy_loss_list),
            "value_loss": np.mean(value_loss_list),
            "entropy": np.mean(entropy_list),
            "clip_fraction": np.mean(clip_fraction_list),
        }

    def save(self, path: str, version: str = "v001", metadata: Optional[Dict] = None) -> str:
        """Save policy to disk with version."""
        save_dir = Path(settings.rl_models_dir) / version
        save_dir.mkdir(parents=True, exist_ok=True)

        model_path = save_dir / "policy.pt"
        save_data = {
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_step": self.training_step,
            "episode_count": self.episode_count,
            "version": version,
            "saved_at": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        torch.save(save_data, model_path)
        logger.info("Policy saved", path=str(model_path), version=version)
        return str(model_path)

    def load(self, path: str) -> Dict:
        """Load policy from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_step = checkpoint.get("training_step", 0)
        self.episode_count = checkpoint.get("episode_count", 0)
        logger.info("Policy loaded", path=path, version=checkpoint.get("version"))
        return checkpoint.get("metadata", {})

    def load_latest(self) -> Optional[str]:
        """Load the latest (most recent) policy."""
        models_dir = Path(settings.rl_models_dir)
        if not models_dir.exists():
            return None

        versions = sorted(
            [d for d in models_dir.iterdir() if d.is_dir() and (d / "policy.pt").exists()]
        )
        if not versions:
            return None

        latest = versions[-1] / "policy.pt"
        self.load(str(latest))
        return str(latest)


class RolloutBuffer:
    """
    Buffer for storing experience during rollout.
    Supports GAE (Generalized Advantage Estimation).
    """

    def __init__(self, n_steps: int = settings.rl_n_steps, state_dim: int = settings.rl_state_dim):
        self.n_steps = n_steps
        self.state_dim = state_dim
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []
        self.dones: List[bool] = []
        self.returns: List[float] = []

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def compute_advantages(
        self, gamma: float, gae_lambda: float = 0.95
    ) -> np.ndarray:
        """Compute GAE advantages and returns."""
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)

        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0.0
                next_non_terminal = 0.0
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - float(dones[t + 1])

            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        self.returns = (advantages + values).tolist()
        return advantages

    def is_ready(self) -> bool:
        return len(self.states) >= self.n_steps

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
        self.returns.clear()

    def __len__(self) -> int:
        return len(self.states)


from typing import List  # noqa: E402
