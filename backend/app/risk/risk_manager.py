"""
ZECRO-RL — Risk Manager (Hard Deterministic Rules)
The LLM and RL policy CANNOT override these rules.
This is the final safety gate before any trade execution.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Dict, List, Optional, Tuple

import structlog

from app.config import settings

logger = structlog.get_logger(__name__)


@dataclass
class RiskCheckResult:
    approved: bool
    rejection_reasons: List[str]
    risk_amount_inr: float = 0.0
    position_size: float = 0.0
    max_loss_inr: float = 0.0


@dataclass
class PortfolioRiskState:
    """Current portfolio risk metrics."""
    equity: float
    peak_equity: float
    available: float
    daily_pnl: float
    daily_start_equity: float
    open_positions: int
    open_positions_value: float
    current_drawdown_pct: float
    daily_loss_pct: float


class RiskManager:
    """
    Hard-coded risk management rules.
    These rules are IMMUTABLE — no LLM or RL agent can override them.

    Rules:
    1. Risk per trade ≤ 0.25% of portfolio equity
    2. Daily loss ≤ ₹75,000
    3. Portfolio drawdown ≤ 10%
    4. Max open positions = 3
    5. Stop loss MUST be defined
    6. Stop loss MUST make mathematical sense (LONG: SL < entry, SHORT: SL > entry)
    7. Risk:Reward must be at least 1:1
    """

    def __init__(
        self,
        initial_capital: float = settings.initial_capital_inr,
        risk_per_trade: float = settings.risk_per_trade,
        max_daily_loss_inr: float = settings.max_daily_loss_inr,
        max_portfolio_dd: float = settings.max_portfolio_dd,
        max_open_positions: int = settings.max_open_positions,
    ):
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_daily_loss_inr = max_daily_loss_inr
        self.max_portfolio_dd = max_portfolio_dd
        self.max_open_positions = max_open_positions

        # Portfolio state
        self.equity = initial_capital
        self.peak_equity = initial_capital
        self.available = initial_capital
        self.daily_pnl = 0.0
        self.daily_start_equity = initial_capital
        self.daily_reset_date: date = datetime.now(timezone.utc).date()
        self.open_positions: Dict[str, Dict] = {}  # symbol -> position info

        logger.info(
            "Risk Manager initialized",
            initial_capital=f"₹{initial_capital:,.0f}",
            risk_per_trade=f"{risk_per_trade * 100:.2f}%",
            max_daily_loss=f"₹{max_daily_loss_inr:,.0f}",
            max_dd=f"{max_portfolio_dd * 100:.1f}%",
        )

    def _reset_daily_if_needed(self):
        """Reset daily P&L at start of new trading day."""
        today = datetime.now(timezone.utc).date()
        if today != self.daily_reset_date:
            self.daily_pnl = 0.0
            self.daily_start_equity = self.equity
            self.daily_reset_date = today
            logger.info("Daily P&L reset", date=str(today))

    def check_trade(
        self,
        symbol: str,
        direction: str,           # "LONG" or "SHORT"
        entry_price: float,
        stop_loss: float,
        tp1: float,
        current_equity: Optional[float] = None,
    ) -> RiskCheckResult:
        """
        Validate a proposed trade against all risk rules.

        Args:
            symbol: Trading symbol
            direction: "LONG" or "SHORT"
            entry_price: Proposed entry price
            stop_loss: Proposed stop loss price
            tp1: First take profit level

        Returns:
            RiskCheckResult with approval status and sizing
        """
        self._reset_daily_if_needed()

        equity = current_equity or self.equity
        rejections = []

        # ── Rule 1: Stop loss must exist ──────────────────────────────────────
        if stop_loss is None or stop_loss <= 0:
            rejections.append("REJECTED: Stop loss is not defined")
            return RiskCheckResult(False, rejections)

        # ── Rule 2: Stop loss must be logical ─────────────────────────────────
        if direction == "LONG" and stop_loss >= entry_price:
            rejections.append(
                f"REJECTED: LONG stop loss ({stop_loss}) must be BELOW entry ({entry_price})"
            )
        if direction == "SHORT" and stop_loss <= entry_price:
            rejections.append(
                f"REJECTED: SHORT stop loss ({stop_loss}) must be ABOVE entry ({entry_price})"
            )

        if rejections:
            return RiskCheckResult(False, rejections)

        # ── Rule 3: Risk:Reward check (minimum 1:1) ────────────────────────────
        sl_distance = abs(entry_price - stop_loss)
        tp1_distance = abs(tp1 - entry_price) if tp1 else 0

        if tp1_distance < sl_distance:
            rejections.append(
                f"REJECTED: RR ratio ({tp1_distance/sl_distance:.2f}) < 1:1 minimum"
            )

        # ── Rule 4: Daily loss limit ───────────────────────────────────────────
        if self.daily_pnl <= -self.max_daily_loss_inr:
            rejections.append(
                f"REJECTED: Daily loss limit reached "
                f"(₹{abs(self.daily_pnl):,.0f} / ₹{self.max_daily_loss_inr:,.0f})"
            )

        # ── Rule 5: Portfolio drawdown limit ───────────────────────────────────
        current_dd = (self.peak_equity - equity) / self.peak_equity
        if current_dd >= self.max_portfolio_dd:
            rejections.append(
                f"REJECTED: Portfolio drawdown limit reached "
                f"({current_dd * 100:.2f}% / {self.max_portfolio_dd * 100:.1f}%)"
            )

        # ── Rule 6: Max open positions ─────────────────────────────────────────
        if len(self.open_positions) >= self.max_open_positions:
            rejections.append(
                f"REJECTED: Max open positions reached "
                f"({len(self.open_positions)}/{self.max_open_positions})"
            )

        # ── Rule 7: No duplicate positions on same symbol ─────────────────────
        if symbol in self.open_positions:
            rejections.append(f"REJECTED: Already have open position in {symbol}")

        if rejections:
            logger.warning("Trade rejected", symbol=symbol, reasons=rejections)
            return RiskCheckResult(False, rejections)

        # ── Calculate position sizing ──────────────────────────────────────────
        risk_amount_inr = equity * self.risk_per_trade
        position_size = risk_amount_inr / sl_distance  # units

        # Cap risk amount (never exceed max)
        max_risk = equity * self.risk_per_trade
        if risk_amount_inr > max_risk:
            risk_amount_inr = max_risk
            position_size = risk_amount_inr / sl_distance

        logger.info(
            "Trade approved",
            symbol=symbol,
            direction=direction,
            entry=entry_price,
            sl=stop_loss,
            risk_inr=f"₹{risk_amount_inr:,.0f}",
            size=f"{position_size:.6f}",
        )

        return RiskCheckResult(
            approved=True,
            rejection_reasons=[],
            risk_amount_inr=risk_amount_inr,
            position_size=position_size,
            max_loss_inr=risk_amount_inr,  # 1R = max loss
        )

    def get_portfolio_state(self) -> PortfolioRiskState:
        """Get current portfolio risk metrics."""
        self._reset_daily_if_needed()
        current_dd = (self.peak_equity - self.equity) / self.peak_equity
        daily_loss_pct = self.daily_pnl / max(self.daily_start_equity, 1) * 100

        return PortfolioRiskState(
            equity=self.equity,
            peak_equity=self.peak_equity,
            available=self.available,
            daily_pnl=self.daily_pnl,
            daily_start_equity=self.daily_start_equity,
            open_positions=len(self.open_positions),
            open_positions_value=sum(
                p.get("position_value", 0) for p in self.open_positions.values()
            ),
            current_drawdown_pct=current_dd * 100,
            daily_loss_pct=daily_loss_pct,
        )

    def record_trade_open(
        self,
        trade_id: str,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        risk_amount_inr: float,
        position_size: float,
    ):
        """Record that a trade has been opened."""
        self.open_positions[symbol] = {
            "trade_id": trade_id,
            "direction": direction,
            "entry": entry_price,
            "stop_loss": stop_loss,
            "risk_inr": risk_amount_inr,
            "position_size": position_size,
            "position_value": position_size * entry_price,
        }
        self.available -= risk_amount_inr
        logger.info("Trade opened", trade_id=trade_id, symbol=symbol)

    def record_trade_close(
        self,
        symbol: str,
        pnl_inr: float,
    ):
        """Record that a trade has been closed."""
        if symbol in self.open_positions:
            risk_inr = self.open_positions[symbol].get("risk_inr", 0)
            self.available += risk_inr
            del self.open_positions[symbol]

        self.equity += pnl_inr
        self.daily_pnl += pnl_inr

        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        logger.info(
            "Trade closed",
            symbol=symbol,
            pnl=f"₹{pnl_inr:,.0f}",
            equity=f"₹{self.equity:,.0f}",
        )

    def get_dashboard_stats(self) -> Dict:
        """Stats for dashboard display."""
        self._reset_daily_if_needed()
        total_pnl = self.equity - self.initial_capital
        return_pct = total_pnl / self.initial_capital * 100
        dd_pct = (self.peak_equity - self.equity) / self.peak_equity * 100

        return {
            "initial_capital_inr": self.initial_capital,
            "current_equity_inr": self.equity,
            "available_inr": self.available,
            "total_pnl_inr": total_pnl,
            "return_pct": return_pct,
            "peak_equity_inr": self.peak_equity,
            "current_drawdown_pct": dd_pct,
            "daily_pnl_inr": self.daily_pnl,
            "max_daily_loss_inr": self.max_daily_loss_inr,
            "daily_loss_remaining_inr": self.max_daily_loss_inr + self.daily_pnl,
            "open_positions": len(self.open_positions),
            "max_positions": self.max_open_positions,
            "risk_per_trade_pct": self.risk_per_trade * 100,
            "risk_per_trade_inr": self.equity * self.risk_per_trade,
        }


# Singleton
risk_manager = RiskManager()
