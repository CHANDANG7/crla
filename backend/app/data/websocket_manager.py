"""
ZECRO-RL — Delta Exchange WebSocket Manager
Handles real-time candle, order book, and ticker feeds.
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set

import websockets
import structlog

from app.config import settings

logger = structlog.get_logger(__name__)


class DeltaWebSocketManager:
    """
    Delta Exchange WebSocket client for real-time market data.
    Subscribes to: candlestick, l2_orderbook, v2/ticker feeds.
    """

    WS_URL = settings.delta_ws_url

    def __init__(self):
        self._ws = None
        self._subscriptions: Set[str] = set()
        self._handlers: Dict[str, List[Callable]] = {}
        self._running = False
        self._reconnect_delay = 5
        self._heartbeat_interval = 30
        self._latest_candles: Dict[str, Dict] = {}  # symbol_tf -> latest candle

    def on(self, channel: str, handler: Callable) -> None:
        """Register a handler for a channel type."""
        if channel not in self._handlers:
            self._handlers[channel] = []
        self._handlers[channel].append(handler)

    def subscribe_candles(self, symbol: str, resolution: str) -> str:
        """Build candle subscription string."""
        # Delta uses resolution in seconds for WS
        resolution_map = {
            "1m": "1", "5m": "5", "15m": "15", "30m": "30",
            "1h": "60", "4h": "240", "1d": "D",
        }
        res = resolution_map.get(resolution, "60")
        return f"candlestick_{res}_{symbol}"

    def subscribe_ticker(self, symbol: str) -> str:
        return f"v2/ticker.{symbol}"

    def subscribe_orderbook(self, symbol: str) -> str:
        return f"l2_orderbook.{symbol}"

    async def connect(
        self,
        symbols: List[str],
        timeframes: List[str] = ["15m", "1h"],
    ) -> None:
        """Start WebSocket connection with auto-reconnect."""
        self._running = True

        # Build subscription list
        channels = []
        for symbol in symbols:
            for tf in timeframes:
                channels.append(self.subscribe_candles(symbol, tf))
            channels.append(self.subscribe_ticker(symbol))

        while self._running:
            try:
                await self._connect_loop(channels)
            except Exception as exc:
                if self._running:
                    logger.warning(
                        "WebSocket disconnected, reconnecting...",
                        error=str(exc),
                        delay=self._reconnect_delay,
                    )
                    await asyncio.sleep(self._reconnect_delay)

    async def _connect_loop(self, channels: List[str]) -> None:
        """Main WebSocket loop."""
        logger.info("Connecting to Delta WebSocket", url=self.WS_URL)

        async with websockets.connect(
            self.WS_URL,
            ping_interval=self._heartbeat_interval,
            ping_timeout=10,
        ) as ws:
            self._ws = ws
            logger.info("WebSocket connected")

            # Subscribe to channels
            await self._subscribe(ws, channels)

            # Start heartbeat
            heartbeat_task = asyncio.create_task(self._heartbeat(ws))

            try:
                async for raw_msg in ws:
                    await self._handle_message(raw_msg)
            finally:
                heartbeat_task.cancel()

    async def _subscribe(self, ws, channels: List[str]) -> None:
        """Send subscription message."""
        msg = {
            "type": "subscribe",
            "payload": {
                "channels": [
                    {"name": ch} for ch in channels
                ]
            }
        }
        await ws.send(json.dumps(msg))
        logger.info("Subscribed to channels", count=len(channels))

    async def _heartbeat(self, ws) -> None:
        """Send heartbeat pings."""
        while True:
            await asyncio.sleep(self._heartbeat_interval)
            try:
                await ws.send(json.dumps({"type": "ping"}))
            except Exception:
                break

    async def _handle_message(self, raw: str) -> None:
        """Parse and dispatch incoming messages."""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        msg_type = msg.get("type", "")

        if msg_type == "candlestick_v2":
            await self._handle_candle(msg)
        elif msg_type == "ticker_v2":
            await self._handle_ticker(msg)
        elif msg_type == "l2_updates":
            await self._handle_orderbook(msg)
        elif msg_type in ("pong", "subscriptions"):
            pass  # ignore heartbeat responses
        else:
            logger.debug("Unhandled message type", type=msg_type)

    async def _handle_candle(self, msg: Dict) -> None:
        """Process incoming candlestick data."""
        try:
            symbol = msg.get("symbol", "")
            data = msg.get("candle", {})
            resolution = msg.get("resolution", "")

            candle = {
                "symbol": symbol,
                "timeframe": self._resolution_to_tf(resolution),
                "timestamp": datetime.fromtimestamp(data["ts"], tz=timezone.utc),
                "open": float(data["open"]),
                "high": float(data["high"]),
                "low": float(data["low"]),
                "close": float(data["close"]),
                "volume": float(data.get("volume", 0)),
            }

            key = f"{symbol}_{resolution}"
            self._latest_candles[key] = candle

            # Dispatch to handlers
            for handler in self._handlers.get("candle", []):
                await handler(candle)

        except (KeyError, ValueError) as e:
            logger.warning("Error parsing candle", error=str(e))

    async def _handle_ticker(self, msg: Dict) -> None:
        """Process ticker updates."""
        try:
            ticker = {
                "symbol": msg.get("symbol", ""),
                "price": float(msg.get("close", 0)),
                "volume_24h": float(msg.get("volume", 0)),
                "funding_rate": float(msg.get("funding_rate", 0)),
                "open_interest": float(msg.get("oi", 0)),
                "timestamp": datetime.now(tz=timezone.utc),
            }

            for handler in self._handlers.get("ticker", []):
                await handler(ticker)

        except (KeyError, ValueError) as e:
            logger.warning("Error parsing ticker", error=str(e))

    async def _handle_orderbook(self, msg: Dict) -> None:
        """Process order book updates."""
        for handler in self._handlers.get("orderbook", []):
            await handler(msg)

    def _resolution_to_tf(self, resolution: str) -> str:
        """Convert Delta resolution string to timeframe string."""
        mapping = {
            "1": "1m", "5": "5m", "15": "15m",
            "30": "30m", "60": "1h", "240": "4h", "D": "1d",
        }
        return mapping.get(str(resolution), resolution)

    def get_latest_candle(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Get the most recent received candle for symbol/tf."""
        res_map = {"1m": "1", "5m": "5", "15m": "15", "1h": "60", "1d": "D"}
        key = f"{symbol}_{res_map.get(timeframe, timeframe)}"
        return self._latest_candles.get(key)

    async def disconnect(self) -> None:
        """Gracefully disconnect."""
        self._running = False
        if self._ws:
            await self._ws.close()


# Global singleton
ws_manager = DeltaWebSocketManager()
