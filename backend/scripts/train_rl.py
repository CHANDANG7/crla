"""
ZECRO-RL — PPO Policy Training Script
Runs chronological walk-forward training, evaluates validation performance, and saves versioned models.

Usage:
    python scripts/train_rl.py --symbol BTCUSD --episodes 150 --auto-promote
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

os.environ["DISABLE_SQLALCHEMY_CEXT"] = "1"
from pathlib import Path

# Add app parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import structlog
from sqlalchemy import desc, select

from app.config import settings
from app.database import async_session_factory, init_db
from app.models import MarketCandle, PolicyVersion
from app.rl.policy_manager import policy_manager
from app.rl.ppo_agent import PPOAgent
from app.rl.trainer import PPOTrainer

logger = structlog.get_logger(__name__)


async def load_candles(symbol: str = "BTCUSD", timeframe: str = "1h", data_path: str = None) -> pd.DataFrame:
    """Load historical candles from CSV or TimescaleDB."""
    if data_path and os.path.exists(data_path):
        logger.info("Loading candles from CSV file", path=data_path)
        df = pd.read_csv(data_path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.sort_values("timestamp").reset_index(drop=True)

    csv_default = Path(f"./data/{symbol}_{timeframe}.csv")
    if csv_default.exists():
        logger.info("Loading candles from default CSV cache", path=str(csv_default))
        df = pd.read_csv(csv_default)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.sort_values("timestamp").reset_index(drop=True)

    # Load from DB
    logger.info("Loading candles from TimescaleDB", symbol=symbol, timeframe=timeframe)
    async with async_session_factory() as db:
        result = await db.execute(
            select(MarketCandle)
            .where(
                MarketCandle.symbol == symbol,
                MarketCandle.timeframe == timeframe,
            )
            .order_by(MarketCandle.timestamp.asc())
        )
        candles = result.scalars().all()

        if not candles:
            # If no data found, generate synthetic market data for initial demonstration & testing
            logger.warning(f"No historical candles found in DB/CSV for {symbol}. Generating synthetic walk data for testing.")
            return generate_synthetic_candles(1500)

        data = [
            {
                "timestamp": c.timestamp,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
            }
            for c in candles
        ]
        return pd.DataFrame(data)


def generate_synthetic_candles(n_bars: int = 1500) -> pd.DataFrame:
    """Generate realistic synthetic crypto OHLCV series for offline initial verification."""
    import numpy as np
    np.random.seed(42)

    timestamps = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=n_bars, freq="1h")
    price = 60000.0
    records = []

    for ts in timestamps:
        ret = np.random.normal(0.0002, 0.008)
        open_p = price
        close_p = open_p * (1.0 + ret)
        high_p = max(open_p, close_p) * (1.0 + abs(np.random.normal(0, 0.003)))
        low_p = min(open_p, close_p) * (1.0 - abs(np.random.normal(0, 0.003)))
        volume = abs(np.random.normal(500, 150))

        records.append({
            "timestamp": ts,
            "open": round(open_p, 2),
            "high": round(high_p, 2),
            "low": round(low_p, 2),
            "close": round(close_p, 2),
            "volume": round(volume, 2),
        })
        price = close_p

    return pd.DataFrame(records)


async def main():
    parser = argparse.ArgumentParser(description="ZECRO-RL PPO Policy Trainer")
    parser.add_argument("--symbol", type=str, default="BTCUSD", help="Symbol to train on")
    parser.add_argument("--episodes", type=int, default=80, help="Number of training episodes")
    parser.add_argument("--train-split", type=float, default=0.70, help="Train split fraction")
    parser.add_argument("--data-path", type=str, default=None, help="Optional CSV file path")
    parser.add_argument("--auto-promote", action="store_true", help="Promote to live paper trading if gates pass")

    args = parser.parse_args()

    await init_db()

    df = await load_candles(symbol=args.symbol, timeframe="1h", data_path=args.data_path)
    logger.info(f"Loaded {len(df)} candles for training.")

    if len(df) < 300:
        logger.error(f"Insufficient candles ({len(df)}). At least 300 required for training.")
        return

    agent = PPOAgent()
    trainer = PPOTrainer(
        df_1h=df,
        symbol=args.symbol,
        train_split=args.train_split,
        episode_length=200,
        agent=agent,
    )

    results = trainer.train(total_episodes=args.episodes)
    metrics = results.get("best_metrics", {})

    logger.info("Training Finished", **metrics)

    # Determine policy version
    async with async_session_factory() as db:
        latest_policy = await db.scalar(
            select(PolicyVersion).order_by(PolicyVersion.id.desc()).limit(1)
        )
    version_name = policy_manager.get_next_version_name(latest_policy)

    # Save policy
    saved_policy = await policy_manager.save_policy(
        agent=agent,
        version_name=version_name,
        metrics=metrics,
        hyperparams={
            "learning_rate": settings.rl_learning_rate,
            "gamma": settings.rl_gamma,
            "clip_epsilon": settings.rl_clip_epsilon,
            "episodes": args.episodes,
        },
        notes=f"Trained on {args.symbol} ({len(df)} candles)",
    )

    if args.auto_promote:
        promoted = await policy_manager.promote_policy(version_name)
        if promoted:
            logger.info(f"🚀 Policy {version_name} successfully promoted to active paper trading!")


if __name__ == "__main__":
    asyncio.run(main())
