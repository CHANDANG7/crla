"""
ZECRO-RL — Performance Metrics Calculator
Computes institutional-grade trading performance metrics:
Sharpe, Sortino, Calmar, Profit Factor, Expectancy (R), Max Drawdown.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class PerformanceMetrics:
    total_trades: int
    winning_trades: int
    losing_trades: int
    breakeven_trades: int
    win_rate_pct: float
    total_pnl_inr: float
    return_pct: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown_pct: float
    avg_trade_r: float
    avg_win_r: float
    avg_loss_r: float
    expectancy_r: float


class PerformanceCalculator:
    """Calculates comprehensive performance metrics for trading episodes and policies."""

    @staticmethod
    def calculate_from_trades(
        trades: List[Dict[str, Any]],
        initial_capital: float = 10000000.0,
        risk_free_rate: float = 0.06,  # 6% annual risk-free rate for INR
    ) -> PerformanceMetrics:
        """
        Calculate metrics from list of completed trade dicts.
        Each trade dict has: {"net_pnl_inr", "realized_r", "exit_reason", ...}
        """
        if not trades:
            return PerformanceMetrics(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                breakeven_trades=0,
                win_rate_pct=0.0,
                total_pnl_inr=0.0,
                return_pct=0.0,
                profit_factor=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                calmar_ratio=0.0,
                max_drawdown_pct=0.0,
                avg_trade_r=0.0,
                avg_win_r=0.0,
                avg_loss_r=0.0,
                expectancy_r=0.0,
            )

        pnls = [t.get("net_pnl_inr", 0.0) for t in trades]
        r_multiples = [t.get("realized_r", 0.0) for t in trades]

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        be = [p for p in pnls if p == 0]

        win_rs = [r for r in r_multiples if r > 0]
        loss_rs = [r for r in r_multiples if r < 0]

        total_trades = len(trades)
        win_count = len(wins)
        loss_count = len(losses)
        be_count = len(be)

        win_rate = (win_count / total_trades) * 100.0 if total_trades > 0 else 0.0
        total_pnl = sum(pnls)
        return_pct = (total_pnl / initial_capital) * 100.0

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 1.0)
        )

        avg_win_r = float(np.mean(win_rs)) if win_rs else 0.0
        avg_loss_r = float(abs(np.mean(loss_rs))) if loss_rs else 0.0
        avg_trade_r = float(np.mean(r_multiples)) if r_multiples else 0.0

        # Expectancy: (WinRate * AvgWinR) - (LossRate * AvgLossR)
        w_prob = win_count / total_trades if total_trades > 0 else 0.0
        l_prob = loss_count / total_trades if total_trades > 0 else 0.0
        expectancy_r = (w_prob * avg_win_r) - (l_prob * avg_loss_r)

        # Equity Curve & Max Drawdown
        equity = initial_capital
        equity_curve = [initial_capital]
        for p in pnls:
            equity += p
            equity_curve.append(equity)

        peak = initial_capital
        drawdowns = []
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100.0
            drawdowns.append(dd)

        max_dd = max(drawdowns) if drawdowns else 0.0

        # Sharpe & Sortino (per-trade basis annualized ~252 trading sessions)
        trade_returns = [p / initial_capital for p in pnls]
        std_returns = float(np.std(trade_returns)) if len(trade_returns) > 1 else 0.0
        mean_return = float(np.mean(trade_returns)) if trade_returns else 0.0

        # Downside deviation
        negative_returns = [r for r in trade_returns if r < 0]
        downside_std = float(np.std(negative_returns)) if len(negative_returns) > 1 else std_returns

        # Annualization factor sqrt(252 * 4) approx for active trading
        annual_factor = np.sqrt(252)
        sharpe = (mean_return / std_returns * annual_factor) if std_returns > 1e-8 else 0.0
        sortino = (mean_return / downside_std * annual_factor) if downside_std > 1e-8 else 0.0
        calmar = (return_pct / max_dd) if max_dd > 0 else return_pct

        return PerformanceMetrics(
            total_trades=total_trades,
            winning_trades=win_count,
            losing_trades=loss_count,
            breakeven_trades=be_count,
            win_rate_pct=round(win_rate, 2),
            total_pnl_inr=round(total_pnl, 2),
            return_pct=round(return_pct, 4),
            profit_factor=round(profit_factor, 2),
            sharpe_ratio=round(sharpe, 2),
            sortino_ratio=round(sortino, 2),
            calmar_ratio=round(calmar, 2),
            max_drawdown_pct=round(max_dd, 2),
            avg_trade_r=round(avg_trade_r, 3),
            avg_win_r=round(avg_win_r, 3),
            avg_loss_r=round(avg_loss_r, 3),
            expectancy_r=round(expectancy_r, 3),
        )


performance_calculator = PerformanceCalculator()
