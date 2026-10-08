"""
ZECRO-RL — Trade Journal
Maintains structured trade logs, audit entries, and post-trade analysis.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


class TradeJournal:
    """Institutional trade journaling and logging."""

    @staticmethod
    def format_trade_entry(
        trade: Dict[str, Any], setup: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Format complete trade log entry for analytics."""
        return {
            "trade_id": trade.get("id"),
            "symbol": trade.get("symbol"),
            "action": trade.get("action"),
            "entry_price": trade.get("entry_price"),
            "stop_loss": trade.get("stop_loss"),
            "tp1": trade.get("tp1"),
            "tp2": trade.get("tp2"),
            "exit_price": trade.get("exit_price"),
            "exit_reason": trade.get("exit_reason"),
            "realized_r": trade.get("realized_r"),
            "net_pnl_inr": trade.get("net_pnl_inr"),
            "fees_inr": trade.get("fees_inr"),
            "regime": trade.get("market_regime"),
            "policy_version": trade.get("policy_version"),
            "entry_time": trade.get("entry_time"),
            "exit_time": trade.get("exit_time"),
            "setup_evidence": setup.get("llm_analysis", {}).get("evidence", []) if setup else [],
            "journal_timestamp": datetime.now(timezone.utc).isoformat(),
        }


trade_journal = TradeJournal()
