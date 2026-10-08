"""
ZECRO-RL — Order Manager
Handles simulated paper order fills with realistic fees and slippage.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import uuid
from typing import Optional

import structlog

from app.config import settings
from app.risk.risk_manager import risk_manager

logger = structlog.get_logger(__name__)


@dataclass
class OrderFill:
    order_id: str
    symbol: str
    direction: str  # "LONG" or "SHORT"
    requested_price: float
    fill_price: float
    quantity: float
    position_size_inr: float
    fee_inr: float
    slippage_inr: float
    timestamp: datetime


class OrderManager:
    """
    Simulates paper order execution with realistic slippage and trading fees.
    """

    def __init__(
        self,
        fee_rate: float = settings.taker_fee,
        slippage_rate: float = settings.slippage_default,
    ):
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate

    def execute_market_order(
        self,
        symbol: str,
        direction: str,
        price: float,
        quantity: float,
        position_size_inr: float,
    ) -> OrderFill:
        """
        Simulate market order execution with slippage and taker fees.
        """
        # Slippage: longs fill higher, shorts fill lower
        slippage_factor = 1.0 + self.slippage_rate if direction == "LONG" else 1.0 - self.slippage_rate
        fill_price = round(price * slippage_factor, 2)

        order_val_inr = fill_price * quantity
        fee_inr = round(order_val_inr * self.fee_rate, 2)
        slippage_inr = round(abs(fill_price - price) * quantity, 2)

        order_id = str(uuid.uuid4())

        logger.info(
            "Paper order executed",
            order_id=order_id,
            symbol=symbol,
            direction=direction,
            req_price=price,
            fill_price=fill_price,
            qty=quantity,
            fee=fee_inr,
        )

        return OrderFill(
            order_id=order_id,
            symbol=symbol,
            direction=direction,
            requested_price=price,
            fill_price=fill_price,
            quantity=quantity,
            position_size_inr=order_val_inr,
            fee_inr=fee_inr,
            slippage_inr=slippage_inr,
            timestamp=datetime.now(timezone.utc),
        )


order_manager = OrderManager()
