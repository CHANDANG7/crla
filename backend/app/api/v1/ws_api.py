"""
ZECRO-RL — WebSocket API for real-time dashboard updates.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import structlog

from app.risk.risk_manager import risk_manager

router = APIRouter()
logger = structlog.get_logger(__name__)


class ConnectionManager:
    """Manage active WebSocket connections."""

    def __init__(self):
        self.active: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.add(ws)
        logger.info("WS client connected", total=len(self.active))

    def disconnect(self, ws: WebSocket):
        self.active.discard(ws)
        logger.info("WS client disconnected", total=len(self.active))

    async def broadcast(self, data: dict):
        """Broadcast to all connected clients."""
        message = json.dumps(data)
        dead = set()
        for ws in self.active:
            try:
                await ws.send_text(message)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self.active.discard(ws)


ws_manager = ConnectionManager()


@router.websocket("/dashboard")
async def dashboard_ws(websocket: WebSocket):
    """
    Real-time dashboard WebSocket.
    Sends portfolio state updates every 5 seconds.
    """
    await ws_manager.connect(websocket)
    try:
        while True:
            # Send portfolio state
            stats = risk_manager.get_dashboard_stats()
            await websocket.send_json({
                "type": "portfolio_update",
                "data": {
                    **stats,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            })
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


async def broadcast_trade_event(trade_data: dict):
    """Broadcast a trade event to all dashboard clients."""
    await ws_manager.broadcast({
        "type": "trade_event",
        "data": trade_data,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


async def broadcast_analysis(analysis_data: dict):
    """Broadcast new analysis to dashboard."""
    await ws_manager.broadcast({
        "type": "analysis_update",
        "data": analysis_data,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
