"""
ZECRO-RL — Position Manager
Manages active paper positions, partial TP scaling (50% TP1, 25% TP2, 25% Runner),
breakeven stop-loss moves, and trailing stops.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import structlog

from app.execution.order_manager import OrderFill, order_manager
from app.risk.risk_manager import risk_manager

logger = structlog.get_logger(__name__)


@dataclass
class Position:
    trade_id: str
    symbol: str
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    initial_quantity: float
    remaining_quantity: float
    stop_loss: float
    tp1: float
    tp2: Optional[float]
    trailing_stop_pct: float
    risk_amount_inr: float
    position_size_inr: float
    entry_time: datetime
    policy_version: Optional[str] = None
    setup_id: Optional[str] = None

    # Partial exit states
    tp1_filled: bool = False
    tp1_price: Optional[float] = None
    tp1_time: Optional[datetime] = None
    tp1_qty: float = 0.0

    tp2_filled: bool = False
    tp2_price: Optional[float] = None
    tp2_time: Optional[datetime] = None
    tp2_qty: float = 0.0

    sl_moved_to_be: bool = False
    tsl_active: bool = False
    tsl_price: Optional[float] = None
    peak_price: float = field(init=False)

    total_realized_pnl_inr: float = 0.0
    total_fees_inr: float = 0.0
    total_slippage_inr: float = 0.0

    def __post_init__(self):
        self.peak_price = self.entry_price


@dataclass
class PositionUpdateResult:
    position_closed: bool
    exit_reason: Optional[str] = None  # "SL", "TP1", "TP2", "TSL", "MANUAL", "TIMEOUT"
    exit_price: Optional[float] = None
    realized_pnl_inr: float = 0.0
    realized_r: float = 0.0
    fees_inr: float = 0.0
    net_pnl_inr: float = 0.0
    partial_tp_event: Optional[str] = None  # "TP1", "TP2"


class PositionManager:
    """
    Tracks and updates all active paper positions.
    """

    def __init__(self):
        self.positions: Dict[str, Position] = {}  # trade_id -> Position

    def open_position(
        self,
        trade_id: str,
        symbol: str,
        direction: str,
        fill: OrderFill,
        stop_loss: float,
        tp1: float,
        tp2: Optional[float],
        risk_amount_inr: float,
        trailing_stop_pct: float = 0.5,
        policy_version: Optional[str] = None,
        setup_id: Optional[str] = None,
    ) -> Position:
        """Create and track a new open position."""
        pos = Position(
            trade_id=trade_id,
            symbol=symbol,
            direction=direction,
            entry_price=fill.fill_price,
            initial_quantity=fill.quantity,
            remaining_quantity=fill.quantity,
            stop_loss=stop_loss,
            tp1=tp1,
            tp2=tp2,
            trailing_stop_pct=trailing_stop_pct,
            risk_amount_inr=risk_amount_inr,
            position_size_inr=fill.position_size_inr,
            entry_time=fill.timestamp,
            policy_version=policy_version,
            setup_id=setup_id,
            total_fees_inr=fill.fee_inr,
            total_slippage_inr=fill.slippage_inr,
        )
        self.positions[trade_id] = pos
        return pos

    def update_position(
        self,
        trade_id: str,
        high: float,
        low: float,
        close: float,
        current_time: Optional[datetime] = None,
    ) -> PositionUpdateResult:
        """
        Process a new candle/tick against an open position.
        Checks Stop Loss, TP1, TP2, and Trailing Stop.
        """
        if trade_id not in self.positions:
            return PositionUpdateResult(position_closed=False)

        pos = self.positions[trade_id]
        now = current_time or datetime.now(timezone.utc)
        sl_dist = abs(pos.entry_price - pos.stop_loss)

        # ── LONG POSITION LOGIC ────────────────────────────────────────────────
        if pos.direction == "LONG":
            pos.peak_price = max(pos.peak_price, high)

            # 1. Check Stop Loss
            if low <= pos.stop_loss:
                exit_fill = order_manager.execute_market_order(
                    pos.symbol, "SHORT", pos.stop_loss, pos.remaining_quantity, pos.remaining_quantity * pos.stop_loss
                )
                pnl = (exit_fill.fill_price - pos.entry_price) * pos.remaining_quantity
                pos.total_realized_pnl_inr += pnl
                pos.total_fees_inr += exit_fill.fee_inr
                pos.total_slippage_inr += exit_fill.slippage_inr
                net_pnl = pos.total_realized_pnl_inr - pos.total_fees_inr - pos.total_slippage_inr
                realized_r = net_pnl / max(pos.risk_amount_inr, 1.0)

                exit_reason = "BE" if pos.sl_moved_to_be else "SL"
                del self.positions[trade_id]

                return PositionUpdateResult(
                    position_closed=True,
                    exit_reason=exit_reason,
                    exit_price=exit_fill.fill_price,
                    realized_pnl_inr=pos.total_realized_pnl_inr,
                    realized_r=realized_r,
                    fees_inr=pos.total_fees_inr,
                    net_pnl_inr=net_pnl,
                )

            # 2. Check TP1 (50% scale out + Move SL to Breakeven)
            if not pos.tp1_filled and high >= pos.tp1:
                tp1_qty = pos.initial_quantity * 0.50
                exit_fill = order_manager.execute_market_order(
                    pos.symbol, "SHORT", pos.tp1, tp1_qty, tp1_qty * pos.tp1
                )
                pnl = (exit_fill.fill_price - pos.entry_price) * tp1_qty
                pos.total_realized_pnl_inr += pnl
                pos.total_fees_inr += exit_fill.fee_inr
                pos.total_slippage_inr += exit_fill.slippage_inr
                pos.remaining_quantity -= tp1_qty
                pos.tp1_filled = True
                pos.tp1_price = exit_fill.fill_price
                pos.tp1_time = now
                pos.tp1_qty = tp1_qty

                # Move SL to breakeven + buffer
                pos.stop_loss = pos.entry_price
                pos.sl_moved_to_be = True
                logger.info("TP1 Filled for LONG — SL moved to Breakeven", trade_id=trade_id, price=pos.tp1)

                return PositionUpdateResult(
                    position_closed=False,
                    partial_tp_event="TP1",
                    realized_pnl_inr=pos.total_realized_pnl_inr,
                    fees_inr=pos.total_fees_inr,
                )

            # 3. Check TP2 (25% scale out + Activate Trailing Stop)
            if pos.tp1_filled and not pos.tp2_filled and pos.tp2 and high >= pos.tp2:
                tp2_qty = pos.initial_quantity * 0.25
                exit_fill = order_manager.execute_market_order(
                    pos.symbol, "SHORT", pos.tp2, tp2_qty, tp2_qty * pos.tp2
                )
                pnl = (exit_fill.fill_price - pos.entry_price) * tp2_qty
                pos.total_realized_pnl_inr += pnl
                pos.total_fees_inr += exit_fill.fee_inr
                pos.total_slippage_inr += exit_fill.slippage_inr
                pos.remaining_quantity -= tp2_qty
                pos.tp2_filled = True
                pos.tp2_price = exit_fill.fill_price
                pos.tp2_time = now
                pos.tp2_qty = tp2_qty

                # Activate Trailing Stop for remaining 25% runner
                pos.tsl_active = True
                pos.tsl_price = pos.peak_price * (1.0 - (pos.trailing_stop_pct / 100.0))
                logger.info("TP2 Filled for LONG — Trailing Stop Active", trade_id=trade_id, price=pos.tp2)

                return PositionUpdateResult(
                    position_closed=False,
                    partial_tp_event="TP2",
                    realized_pnl_inr=pos.total_realized_pnl_inr,
                    fees_inr=pos.total_fees_inr,
                )

            # 4. Update and check Trailing Stop Loss for runner
            if pos.tsl_active and pos.tsl_price:
                new_tsl = pos.peak_price * (1.0 - (pos.trailing_stop_pct / 100.0))
                pos.tsl_price = max(pos.tsl_price, new_tsl)

                if low <= pos.tsl_price:
                    exit_fill = order_manager.execute_market_order(
                        pos.symbol, "SHORT", pos.tsl_price, pos.remaining_quantity, pos.remaining_quantity * pos.tsl_price
                    )
                    pnl = (exit_fill.fill_price - pos.entry_price) * pos.remaining_quantity
                    pos.total_realized_pnl_inr += pnl
                    pos.total_fees_inr += exit_fill.fee_inr
                    pos.total_slippage_inr += exit_fill.slippage_inr
                    net_pnl = pos.total_realized_pnl_inr - pos.total_fees_inr - pos.total_slippage_inr
                    realized_r = net_pnl / max(pos.risk_amount_inr, 1.0)

                    del self.positions[trade_id]
                    return PositionUpdateResult(
                        position_closed=True,
                        exit_reason="TSL",
                        exit_price=exit_fill.fill_price,
                        realized_pnl_inr=pos.total_realized_pnl_inr,
                        realized_r=realized_r,
                        fees_inr=pos.total_fees_inr,
                        net_pnl_inr=net_pnl,
                    )

        # ── SHORT POSITION LOGIC ───────────────────────────────────────────────
        elif pos.direction == "SHORT":
            pos.peak_price = min(pos.peak_price, low)

            # 1. Check Stop Loss
            if high >= pos.stop_loss:
                exit_fill = order_manager.execute_market_order(
                    pos.symbol, "LONG", pos.stop_loss, pos.remaining_quantity, pos.remaining_quantity * pos.stop_loss
                )
                pnl = (pos.entry_price - exit_fill.fill_price) * pos.remaining_quantity
                pos.total_realized_pnl_inr += pnl
                pos.total_fees_inr += exit_fill.fee_inr
                pos.total_slippage_inr += exit_fill.slippage_inr
                net_pnl = pos.total_realized_pnl_inr - pos.total_fees_inr - pos.total_slippage_inr
                realized_r = net_pnl / max(pos.risk_amount_inr, 1.0)

                exit_reason = "BE" if pos.sl_moved_to_be else "SL"
                del self.positions[trade_id]

                return PositionUpdateResult(
                    position_closed=True,
                    exit_reason=exit_reason,
                    exit_price=exit_fill.fill_price,
                    realized_pnl_inr=pos.total_realized_pnl_inr,
                    realized_r=realized_r,
                    fees_inr=pos.total_fees_inr,
                    net_pnl_inr=net_pnl,
                )

            # 2. Check TP1 (50% scale out + Move SL to Breakeven)
            if not pos.tp1_filled and low <= pos.tp1:
                tp1_qty = pos.initial_quantity * 0.50
                exit_fill = order_manager.execute_market_order(
                    pos.symbol, "LONG", pos.tp1, tp1_qty, tp1_qty * pos.tp1
                )
                pnl = (pos.entry_price - exit_fill.fill_price) * tp1_qty
                pos.total_realized_pnl_inr += pnl
                pos.total_fees_inr += exit_fill.fee_inr
                pos.total_slippage_inr += exit_fill.slippage_inr
                pos.remaining_quantity -= tp1_qty
                pos.tp1_filled = True
                pos.tp1_price = exit_fill.fill_price
                pos.tp1_time = now
                pos.tp1_qty = tp1_qty

                # Move SL to breakeven
                pos.stop_loss = pos.entry_price
                pos.sl_moved_to_be = True
                logger.info("TP1 Filled for SHORT — SL moved to Breakeven", trade_id=trade_id, price=pos.tp1)

                return PositionUpdateResult(
                    position_closed=False,
                    partial_tp_event="TP1",
                    realized_pnl_inr=pos.total_realized_pnl_inr,
                    fees_inr=pos.total_fees_inr,
                )

            # 3. Check TP2 (25% scale out + Activate Trailing Stop)
            if pos.tp1_filled and not pos.tp2_filled and pos.tp2 and low <= pos.tp2:
                tp2_qty = pos.initial_quantity * 0.25
                exit_fill = order_manager.execute_market_order(
                    pos.symbol, "LONG", pos.tp2, tp2_qty, tp2_qty * pos.tp2
                )
                pnl = (pos.entry_price - exit_fill.fill_price) * tp2_qty
                pos.total_realized_pnl_inr += pnl
                pos.total_fees_inr += exit_fill.fee_inr
                pos.total_slippage_inr += exit_fill.slippage_inr
                pos.remaining_quantity -= tp2_qty
                pos.tp2_filled = True
                pos.tp2_price = exit_fill.fill_price
                pos.tp2_time = now
                pos.tp2_qty = tp2_qty

                # Activate Trailing Stop for runner
                pos.tsl_active = True
                pos.tsl_price = pos.peak_price * (1.0 + (pos.trailing_stop_pct / 100.0))
                logger.info("TP2 Filled for SHORT — Trailing Stop Active", trade_id=trade_id, price=pos.tp2)

                return PositionUpdateResult(
                    position_closed=False,
                    partial_tp_event="TP2",
                    realized_pnl_inr=pos.total_realized_pnl_inr,
                    fees_inr=pos.total_fees_inr,
                )

            # 4. Check Trailing Stop Loss for runner
            if pos.tsl_active and pos.tsl_price:
                new_tsl = pos.peak_price * (1.0 + (pos.trailing_stop_pct / 100.0))
                pos.tsl_price = min(pos.tsl_price, new_tsl)

                if high >= pos.tsl_price:
                    exit_fill = order_manager.execute_market_order(
                        pos.symbol, "LONG", pos.tsl_price, pos.remaining_quantity, pos.remaining_quantity * pos.tsl_price
                    )
                    pnl = (pos.entry_price - exit_fill.fill_price) * pos.remaining_quantity
                    pos.total_realized_pnl_inr += pnl
                    pos.total_fees_inr += exit_fill.fee_inr
                    pos.total_slippage_inr += exit_fill.slippage_inr
                    net_pnl = pos.total_realized_pnl_inr - pos.total_fees_inr - pos.total_slippage_inr
                    realized_r = net_pnl / max(pos.risk_amount_inr, 1.0)

                    del self.positions[trade_id]
                    return PositionUpdateResult(
                        position_closed=True,
                        exit_reason="TSL",
                        exit_price=exit_fill.fill_price,
                        realized_pnl_inr=pos.total_realized_pnl_inr,
                        realized_r=realized_r,
                        fees_inr=pos.total_fees_inr,
                        net_pnl_inr=net_pnl,
                    )

        return PositionUpdateResult(position_closed=False)


position_manager = PositionManager()
