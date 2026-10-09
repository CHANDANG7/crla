"""
ZECRO-RL — Delta Exchange REST Client
Handles historical OHLC data, market info, and REST interactions.
"""
from __future__ import annotations

import hashlib
import hmac
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import httpx
import structlog

from app.config import settings

logger = structlog.get_logger(__name__)

# Delta Exchange resolution codes
DELTA_RESOLUTIONS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "1d": 86400,
}


class DeltaExchangeClient:
    """
    Delta Exchange REST API client.
    Documentation: https://docs.delta.exchange
    """

    def __init__(
        self,
        api_key: str = settings.delta_api_key,
        api_secret: str = settings.delta_api_secret,
        base_url: str = settings.delta_base_url,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "DeltaExchangeClient":
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(30.0),
            headers={"Accept": "application/json"},
        )
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    def _sign_request(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        """Generate HMAC-SHA256 signature for authenticated requests."""
        timestamp = str(int(time.time()))
        message = method.upper() + timestamp + path + body
        signature = hmac.new(
            self.api_secret.encode(),
            message.encode(),
            hashlib.sha256,
        ).hexdigest()
        return {
            "api-key": self.api_key,
            "timestamp": timestamp,
            "signature": signature,
            "Content-Type": "application/json",
        }

    async def _get(
        self,
        path: str,
        params: Optional[Dict] = None,
        authenticated: bool = False,
    ) -> Any:
        """Perform a GET request."""
        assert self._client is not None, "Use as async context manager"
        query_string = f"?{urlencode(params)}" if params else ""
        full_path = path + query_string
        headers = {}
        if authenticated:
            headers = self._sign_request("GET", full_path)

        resp = await self._client.get(path, params=params, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, dict) and data.get("success") is False:
            raise ValueError(f"Delta API error: {data.get('error', data)}")

        return data

    # ── Public endpoints ───────────────────────────────────────────────────────

    async def get_products(self) -> List[Dict]:
        """Get all perpetual products listed on Delta Exchange."""
        data = await self._get("/v2/products")
        products = data.get("result", [])
        # Filter to perpetuals only
        return [p for p in products if p.get("contract_type") == "perpetual_futures"]

    async def get_tickers(self) -> List[Dict]:
        """Get all current tickers."""
        data = await self._get("/v2/tickers")
        return data.get("result", [])

    async def get_ticker(self, symbol: str) -> Dict:
        """Get ticker for a specific symbol."""
        data = await self._get(f"/v2/tickers/{symbol}")
        return data.get("result", {})

    async def get_ohlc(
        self,
        symbol: str,
        resolution: str,
        start: int,
        end: int,
    ) -> List[Dict]:
        """
        Fetch historical OHLCV candles.

        Args:
            symbol: e.g. "BTCUSD"
            resolution: "1m", "5m", "15m", "1h", "1d"
            start: unix timestamp (seconds)
            end: unix timestamp (seconds)

        Returns:
            List of candle dicts with t, o, h, l, c, v keys
        """
        resolution_code = DELTA_RESOLUTIONS.get(resolution)
        if resolution_code is None:
            raise ValueError(f"Invalid resolution: {resolution}. Valid: {list(DELTA_RESOLUTIONS.keys())}")

        # Delta Exchange API v2 /v2/history/candles requires 'start' and 'end' query parameters
        params = {
            "symbol": symbol,
            "resolution": str(resolution),
            "start": start,
            "end": end,
        }

        data = await self._get("/v2/history/candles", params=params)
        candles = data.get("result", [])

        if not candles:
            return []

        # Case A: Result is a list of candle objects [{"time":..., "open":...}]
        if isinstance(candles, list):
            result = []
            for c in candles:
                result.append({
                    "start_time": c.get("time") or c.get("t") or c.get("start_time"),
                    "open": float(c.get("open") or c.get("o", 0)),
                    "high": float(c.get("high") or c.get("h", 0)),
                    "low": float(c.get("low") or c.get("l", 0)),
                    "close": float(c.get("close") or c.get("c", 0)),
                    "volume": float(c.get("volume") or c.get("v", 0)),
                })
            return result

        # Case B: Result is a dictionary of arrays {"t": [...], "o": [...], ...}
        if isinstance(candles, dict) and "t" in candles:
            times = candles["t"]
            opens = candles.get("o", [])
            highs = candles.get("h", [])
            lows = candles.get("l", [])
            closes = candles.get("c", [])
            volumes = candles.get("v", [0] * len(times))

            result = []
            for i in range(len(times)):
                result.append({
                    "start_time": times[i],
                    "open": float(opens[i]) if i < len(opens) else 0.0,
                    "high": float(highs[i]) if i < len(highs) else 0.0,
                    "low": float(lows[i]) if i < len(lows) else 0.0,
                    "close": float(closes[i]) if i < len(closes) else 0.0,
                    "volume": float(volumes[i]) if i < len(volumes) else 0.0,
                })
            return result

        return []


    async def get_ohlc_paginated(
        self,
        symbol: str,
        resolution: str,
        start: datetime,
        end: datetime,
        max_candles_per_req: int = 2000,
    ) -> List[Dict]:
        """
        Fetch OHLCV candles with automatic pagination for large date ranges.
        Delta API limits ~2000 candles per request.
        """
        resolution_seconds = DELTA_RESOLUTIONS[resolution]
        start_ts = int(start.timestamp())
        end_ts = int(end.timestamp())

        window = max_candles_per_req * resolution_seconds
        all_candles: List[Dict] = []
        current_start = start_ts

        while current_start < end_ts:
            current_end = min(current_start + window, end_ts)
            logger.info(
                "Fetching candles",
                symbol=symbol,
                resolution=resolution,
                from_ts=current_start,
                to_ts=current_end,
            )
            candles = await self.get_ohlc(symbol, resolution, current_start, current_end)
            all_candles.extend(candles)
            current_start = current_end

            # Rate limit: ~30 req/sec on public endpoints
            await asyncio.sleep(0.1)

        # Deduplicate and sort
        seen = set()
        unique = []
        for c in all_candles:
            key = (c["timestamp"],)
            if key not in seen:
                seen.add(key)
                unique.append(c)

        unique.sort(key=lambda x: x["timestamp"])
        return unique

    async def get_funding_history(
        self,
        symbol: str,
        start: int,
        end: int,
    ) -> List[Dict]:
        """Get historical funding rates."""
        params = {"symbol": symbol, "from": start, "to": end}
        data = await self._get("/v2/funding_history", params=params)
        return data.get("result", [])

    async def get_open_interest(self, symbol: str) -> Dict:
        """Get current open interest for a symbol."""
        data = await self._get(f"/v2/open_interest/{symbol}")
        return data.get("result", {})

    async def get_orderbook(self, symbol: str, depth: int = 10) -> Dict:
        """Get current order book."""
        data = await self._get(f"/v2/l2orderbook/{symbol}", params={"depth": depth})
        return data.get("result", {})

    async def get_recent_trades(self, symbol: str) -> List[Dict]:
        """Get recent trades."""
        data = await self._get(f"/v2/trades/{symbol}")
        return data.get("result", [])


import asyncio  # noqa: E402 (needed at bottom for asyncio.sleep in paginated)
