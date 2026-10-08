"""
ZECRO-RL — Historical Candle Downloader
Fetches historical OHLCV data from Delta Exchange and stores in TimescaleDB.

Usage:
    python scripts/download_history.py --symbol BTCUSD --timeframe 1h --days 60
    python scripts/download_history.py --all --days 90
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
import time
from datetime import datetime, timezone
from pathlib import Path

# Add app parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import structlog
from sqlalchemy.dialects.postgresql import insert

from app.config import settings
from app.data.delta_client import DELTA_RESOLUTIONS, DeltaExchangeClient
from app.database import async_session_factory, init_db
from app.models import MarketCandle

logger = structlog.get_logger(__name__)


async def download_candles_for_symbol(
    client: DeltaExchangeClient,
    symbol: str,
    timeframe: str,
    days: int = 60,
    save_csv: bool = True,
):
    """Download historical candles and upsert into database."""
    resolution = DELTA_RESOLUTIONS.get(timeframe)
    if not resolution:
        logger.error(f"Unsupported timeframe: {timeframe}")
        return

    now_ts = int(time.time())
    start_ts = now_ts - (days * 86400)

    logger.info(
        "Downloading candles",
        symbol=symbol,
        timeframe=timeframe,
        days=days,
        start=datetime.fromtimestamp(start_ts, tz=timezone.utc).isoformat(),
        end=datetime.fromtimestamp(now_ts, tz=timezone.utc).isoformat(),
    )

    all_candles = []
    curr_start = start_ts
    step = resolution * 1000  # Max 1000 bars per request

    while curr_start < now_ts:
        curr_end = min(curr_start + step, now_ts)
        try:
            raw = await client.get_ohlc(
                symbol=symbol,
                resolution=timeframe,
                start=curr_start,
                end=curr_end,
            )
            if raw:
                all_candles.extend(raw)
            await asyncio.sleep(0.2)  # Rate limit courtesy
        except Exception as e:
            logger.warning("Error fetching chunk", start=curr_start, error=str(e))

        curr_start = curr_end

    if not all_candles:
        logger.warning(f"No candles retrieved for {symbol} ({timeframe})")
        return

    # Deduplicate and sort by timestamp
    seen = set()
    unique_candles = []
    for c in all_candles:
        t = c.get("time") or c.get("t")
        if t and t not in seen:
            seen.add(t)
            unique_candles.append({
                "timestamp": datetime.fromtimestamp(t, tz=timezone.utc),
                "open": float(c.get("open") or c.get("o")),
                "high": float(c.get("high") or c.get("h")),
                "low": float(c.get("low") or c.get("l")),
                "close": float(c.get("close") or c.get("c")),
                "volume": float(c.get("volume") or c.get("v") or 0.0),
            })

    unique_candles.sort(key=lambda x: x["timestamp"])
    logger.info(f"Retrieved {len(unique_candles)} unique {timeframe} candles for {symbol}")

    # Upsert to database
    async with async_session_factory() as db:
        for chunk in [unique_candles[i:i+500] for i in range(0, len(unique_candles), 500)]:
            records = [
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "timestamp": c["timestamp"],
                    "open": c["open"],
                    "high": c["high"],
                    "low": c["low"],
                    "close": c["close"],
                    "volume": c["volume"],
                }
                for c in chunk
            ]
            stmt = insert(MarketCandle).values(records)
            stmt = stmt.on_conflict_do_update(
                constraint="uq_candle",
                set_={
                    "open": stmt.excluded.open,
                    "high": stmt.excluded.high,
                    "low": stmt.excluded.low,
                    "close": stmt.excluded.close,
                    "volume": stmt.excluded.volume,
                },
            )
            await db.execute(stmt)
        await db.commit()

    logger.info(f"Database upsert complete for {symbol} ({timeframe})")

    # Optional CSV cache
    if save_csv:
        data_dir = Path("./data")
        data_dir.mkdir(parents=True, exist_ok=True)
        csv_file = data_dir / f"{symbol}_{timeframe}.csv"
        df = pd.DataFrame(unique_candles)
        df.to_csv(csv_file, index=False)
        logger.info(f"Saved CSV cache to {csv_file}")


async def main():
    parser = argparse.ArgumentParser(description="Download Delta Exchange Historical Candles")
    parser.add_argument("--symbol", type=str, default="BTCUSD", help="Symbol to download")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe (15m, 1h, 1d)")
    parser.add_argument("--days", type=int, default=60, help="Number of historical days")
    parser.add_argument("--all", action="store_true", help="Download all configured symbols & timeframes")

    args = parser.parse_args()

    await init_db()

    async with DeltaExchangeClient() as client:
        if args.all:
            for sym in settings.symbols:
                for tf in ["15m", "1h"]:
                    await download_candles_for_symbol(client, sym, tf, args.days)
        else:
            await download_candles_for_symbol(client, args.symbol, args.timeframe, args.days)


if __name__ == "__main__":
    asyncio.run(main())
