"""
ZECRO-RL — Paper Trading Execution Engine
Orchestrates the live/simulated trading loop:
Market State → 1H Bias → 15M Entry → RL Policy Action → Risk Gate → Position Execution & Lifecycle.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid

import numpy as np
import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.ws_api import broadcast_analysis, broadcast_trade_event
from app.config import settings
from app.database import async_session_factory
from app.execution.order_manager import order_manager
from app.execution.position_manager import position_manager
from app.models import (
    ActionEnum, BiasEnum, MarketCandle, MarketRegime,
    PortfolioSnapshot, TradeExecution, TradeReward, TradeSetup,
)
from app.risk.risk_manager import risk_manager
from app.rl.ppo_agent import PPOAgent
from app.strategy.bias_engine import bias_engine
from app.strategy.entry_engine import entry_engine
from app.strategy.regime_detector import regime_detector
from app.ta.feature_engine import feature_engine

logger = structlog.get_logger(__name__)


class PaperEngine:
    """
    Paper Trading Execution Engine.
    Coordinates all trading systems without risking real capital.
    """

    def __init__(self):
        self.agent: Optional[PPOAgent] = None
        self.is_running: bool = False
        self.active_policy_version: str = "v001_untrained"
        self._candle_buffers: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    def set_agent(self, agent: PPOAgent, policy_version: str = "v001"):
        """Attach active trained RL agent."""
        self.agent = agent
        self.active_policy_version = policy_version
        logger.info("Paper engine attached RL policy", version=policy_version)

    async def start_autonomous_loop(self, symbols: Optional[List[str]] = None):
        """
        24/7 Autonomous Market Monitoring & Execution Loop.
        Periodically fetches live candles from Delta Exchange, runs TA feature extraction,
        RL policy inference, risk checks, and executes trades automatically.
        """
        symbols = symbols or settings.symbols
        self.is_running = True

        logger.info("⚡ Started 24/7 Autonomous Market Monitor & Trading Loop", symbols=symbols)

        from app.data.delta_client import DeltaExchangeClient
        from app.rl.policy_manager import policy_manager

        # Try loading active policy from DB or fallback
        try:
            loaded = await policy_manager.load_active_policy()
            if loaded:
                self.set_agent(loaded[0], loaded[1])
            else:
                self.set_agent(PPOAgent(), "v001_initial")
        except Exception as e:
            logger.warning("Could not auto-load active policy, using default PPO agent", error=str(e))
            self.set_agent(PPOAgent(), "v001_initial")

        while self.is_running:
            try:
                async with DeltaExchangeClient() as client:
                    # Fetch active perpetual products from Delta Exchange
                    valid_products = set()
                    try:
                        prods = await client.get_products()
                        valid_products = {p.get("symbol") for p in prods if p.get("symbol")}
                    except Exception as pe:
                        logger.warning("Could not fetch active Delta Exchange products list", error=str(pe))

                    for symbol in symbols:
                        try:
                            # Auto-resolve target symbol name on Delta Exchange
                            target_symbol = symbol
                            if valid_products:
                                if symbol in valid_products:
                                    target_symbol = symbol
                                elif symbol.endswith("USD") and f"{symbol}T" in valid_products:
                                    target_symbol = f"{symbol}T"
                                elif symbol.endswith("USDT") and symbol[:-1] in valid_products:
                                    target_symbol = symbol[:-1]
                                elif symbol not in valid_products:
                                    continue  # Skip if symbol does not exist on Delta Exchange

                            end_time = int(datetime.now(timezone.utc).timestamp())
                            start_1h = end_time - (100 * 3600)
                            start_15m = end_time - (100 * 900)

                            candles_1h = await client.get_ohlc(target_symbol, "1h", start_1h, end_time)
                            candles_15m = await client.get_ohlc(target_symbol, "15m", start_15m, end_time)

                            if candles_1h and candles_15m:
                                for c in candles_1h[-60:]:
                                    candle_dict = {
                                        "timestamp": datetime.fromtimestamp(c["start_time"], tz=timezone.utc) if isinstance(c.get("start_time"), (int, float)) else datetime.now(timezone.utc),
                                        "open": float(c["open"]),
                                        "high": float(c["high"]),
                                        "low": float(c["low"]),
                                        "close": float(c["close"]),
                                        "volume": float(c.get("volume", 0)),
                                    }
                                    await self.process_candle(target_symbol, "1h", candle_dict)

                                for c in candles_15m[-10:]:
                                    candle_dict = {
                                        "timestamp": datetime.fromtimestamp(c["start_time"], tz=timezone.utc) if isinstance(c.get("start_time"), (int, float)) else datetime.now(timezone.utc),
                                        "open": float(c["open"]),
                                        "high": float(c["high"]),
                                        "low": float(c["low"]),
                                        "close": float(c["close"]),
                                        "volume": float(c.get("volume", 0)),
                                    }
                                    await self.process_candle(target_symbol, "15m", candle_dict)
                        except Exception as sym_e:
                            logger.debug("Symbol processing skip", symbol=symbol, error=str(sym_e))


            except Exception as e:
                logger.error("Error in autonomous trading loop tick", error=str(e))


            # Poll market every 60 seconds
            await asyncio.sleep(60)


    async def process_candle(
        self,
        symbol: str,
        timeframe: str,
        candle: Dict[str, Any],  # {"timestamp", "open", "high", "low", "close", "volume"}
    ):
        """
        Main candle processing pipeline.
        Invoked on each closed or live candle tick.
        """
        # Maintain internal buffer of candles
        if symbol not in self._candle_buffers:
            self._candle_buffers[symbol] = {"1h": [], "15m": []}

        buf = self._candle_buffers[symbol].setdefault(timeframe, [])
        buf.append(candle)
        if len(buf) > 300:
            buf.pop(0)

        # ── 1. Update existing open positions first ────────────────────────────
        await self._update_open_positions(symbol, candle)

        # We only evaluate new trade setups on 15M candle completion
        if timeframe != settings.timeframe_entry:
            return

        buf_1h = self._candle_buffers[symbol].get("1h", [])
        buf_15m = self._candle_buffers[symbol].get("15m", [])

        if len(buf_1h) < 20 or len(buf_15m) < 20:
            return  # Need minimum historical context

        # ── 2. Compute TA Features ─────────────────────────────────────────────
        state_1h = feature_engine.compute_state(symbol, "1h", buf_1h)
        state_15m = feature_engine.compute_state(symbol, "15m", buf_15m)

        # ── 3. Strategy: 1H Bias & 15M Entry Setup ─────────────────────────────
        bias_res = await bias_engine.compute_bias(state_1h, use_llm=False)
        entry_sig = entry_engine.evaluate_entry(state_15m, bias_res)

        # ── 4. RL Agent Action Selection ───────────────────────────────────────
        state_vector = feature_engine.to_rl_vector(state_15m)
        rl_action = ActionEnum.HOLD
        action_idx = 0

        if self.agent is not None:
            action_idx, _, _ = self.agent.select_action(state_vector, deterministic=True)
            action_map = {0: ActionEnum.HOLD, 1: ActionEnum.LONG, 2: ActionEnum.SHORT, 3: ActionEnum.EXIT}
            rl_action = action_map.get(action_idx, ActionEnum.HOLD)
        else:
            # Fallback: follow strategy rule if entry signal is valid
            if entry_sig.is_valid:
                rl_action = entry_sig.action

        # ── 5. Risk Gate & Order Execution ─────────────────────────────────────
        if entry_sig.is_valid and rl_action in (ActionEnum.LONG, ActionEnum.SHORT) and rl_action == entry_sig.action:
            direction_str = "LONG" if rl_action == ActionEnum.LONG else "SHORT"

            risk_check = risk_manager.check_trade(
                symbol=symbol,
                direction=direction_str,
                entry_price=entry_sig.entry_price,
                stop_loss=entry_sig.stop_loss,
                tp1=entry_sig.tp1,
            )

            if risk_check.approved:
                # Execute simulated paper order
                fill = order_manager.execute_market_order(
                    symbol=symbol,
                    direction=direction_str,
                    price=entry_sig.entry_price,
                    quantity=risk_check.position_size,
                    position_size_inr=risk_check.position_size * entry_sig.entry_price,
                )

                trade_id = str(uuid.uuid4())

                # Track position
                position_manager.open_position(
                    trade_id=trade_id,
                    symbol=symbol,
                    direction=direction_str,
                    fill=fill,
                    stop_loss=entry_sig.stop_loss,
                    tp1=entry_sig.tp1,
                    tp2=entry_sig.tp2,
                    risk_amount_inr=risk_check.risk_amount_inr,
                    trailing_stop_pct=settings.trailing_stop_pct,
                    policy_version=self.active_policy_version,
                )

                # Update risk manager
                risk_manager.record_trade_open(
                    trade_id=trade_id,
                    symbol=symbol,
                    direction=direction_str,
                    entry_price=fill.fill_price,
                    stop_loss=entry_sig.stop_loss,
                    risk_amount_inr=risk_check.risk_amount_inr,
                    position_size=fill.quantity,
                )

                # Persist to database
                await self._persist_trade_open(
                    trade_id=trade_id,
                    symbol=symbol,
                    direction=direction_str,
                    fill=fill,
                    stop_loss=entry_sig.stop_loss,
                    tp1=entry_sig.tp1,
                    tp2=entry_sig.tp2,
                    risk_amount_inr=risk_check.risk_amount_inr,
                    regime=bias_res.regime,
                )

                # Broadcast live trade open to dashboard
                await broadcast_trade_event({
                    "event": "TRADE_OPEN",
                    "trade_id": trade_id,
                    "symbol": symbol,
                    "direction": direction_str,
                    "entry_price": fill.fill_price,
                    "stop_loss": entry_sig.stop_loss,
                    "tp1": entry_sig.tp1,
                    "tp2": entry_sig.tp2,
                    "quantity": fill.quantity,
                })

    async def _update_open_positions(self, symbol: str, candle: Dict[str, Any]):
        """Check all active positions for symbol against latest candle."""
        open_pos_ids = [
            tid for tid, p in position_manager.positions.items() if p.symbol == symbol
        ]

        for trade_id in open_pos_ids:
            res = position_manager.update_position(
                trade_id=trade_id,
                high=candle["high"],
                low=candle["low"],
                close=candle["close"],
                current_time=candle.get("timestamp"),
            )

            if res.partial_tp_event:
                await broadcast_trade_event({
                    "event": f"PARTIAL_TP_{res.partial_tp_event}",
                    "trade_id": trade_id,
                    "symbol": symbol,
                    "realized_pnl_inr": res.realized_pnl_inr,
                })

            if res.position_closed:
                # Update risk manager
                risk_manager.record_trade_close(symbol, res.net_pnl_inr)

                # Persist trade close to database
                await self._persist_trade_close(
                    trade_id=trade_id,
                    update_res=res,
                    exit_time=candle.get("timestamp", datetime.now(timezone.utc)),
                )

                # Broadcast closed trade event
                await broadcast_trade_event({
                    "event": "TRADE_CLOSED",
                    "trade_id": trade_id,
                    "symbol": symbol,
                    "exit_reason": res.exit_reason,
                    "exit_price": res.exit_price,
                    "net_pnl_inr": res.net_pnl_inr,
                    "realized_r": res.realized_r,
                })

    async def _persist_trade_open(
        self,
        trade_id: str,
        symbol: str,
        direction: str,
        fill: Any,
        stop_loss: float,
        tp1: float,
        tp2: Optional[float],
        risk_amount_inr: float,
        regime: str,
    ):
        """Save opened trade to database."""
        try:
            async with async_session_factory() as db:
                trade = TradeExecution(
                    id=trade_id,
                    symbol=symbol,
                    action=direction,
                    entry_price=fill.fill_price,
                    stop_loss=stop_loss,
                    tp1=tp1,
                    tp2=tp2,
                    trailing_stop_pct=settings.trailing_stop_pct,
                    position_size=fill.quantity,
                    position_size_inr=fill.position_size_inr,
                    risk_amount_inr=risk_amount_inr,
                    risk_r=1.0,
                    leverage=1,
                    entry_time=fill.timestamp,
                    fees_inr=fill.fee_inr,
                    slippage_inr=fill.slippage_inr,
                    policy_version=self.active_policy_version,
                    market_regime=regime,
                    is_paper=True,
                    is_open=True,
                )
                db.add(trade)
                await db.commit()
        except Exception as e:
            logger.error("Failed to persist trade open to DB", error=str(e))

    async def _persist_trade_close(
        self,
        trade_id: str,
        update_res: Any,
        exit_time: datetime,
    ):
        """Save closed trade outcome and compute RL reward."""
        try:
            async with async_session_factory() as db:
                result = await db.execute(
                    select(TradeExecution).where(TradeExecution.id == trade_id)
                )
                trade = result.scalar_one_or_none()
                if trade:
                    trade.is_open = False
                    trade.exit_time = exit_time
                    trade.exit_price = update_res.exit_price
                    trade.exit_reason = update_res.exit_reason
                    trade.realized_pnl_inr = update_res.realized_pnl_inr
                    trade.realized_r = update_res.realized_r
                    trade.fees_inr = update_res.fees_inr
                    trade.net_pnl_inr = update_res.net_pnl_inr

                    # Compute reward record
                    reward_record = TradeReward(
                        trade_id=trade_id,
                        realized_r=update_res.realized_r,
                        transaction_cost_r=update_res.fees_inr / max(trade.risk_amount_inr, 1.0),
                        drawdown_penalty=0.0,
                        overtrading_penalty=0.0,
                        invalid_entry_penalty=0.0,
                        final_reward=float(np.clip(update_res.realized_r, -3.0, 3.0)),
                    )
                    db.add(reward_record)

                    # Also snapshot portfolio state
                    stats = risk_manager.get_dashboard_stats()
                    snap = PortfolioSnapshot(
                        timestamp=datetime.now(timezone.utc),
                        equity_inr=stats["current_equity_inr"],
                        available_inr=stats["available_inr"],
                        realized_pnl_inr=stats["total_pnl_inr"],
                        total_pnl_inr=stats["total_pnl_inr"],
                        return_pct=stats["return_pct"],
                        drawdown_pct=stats["current_drawdown_pct"],
                        peak_equity=stats["peak_equity_inr"],
                        open_positions=stats["open_positions"],
                        daily_pnl_inr=stats["daily_pnl_inr"],
                        policy_version=self.active_policy_version,
                    )
                    db.add(snap)

                    await db.commit()
        except Exception as e:
            logger.error("Failed to persist trade close to DB", error=str(e))


paper_engine = PaperEngine()
