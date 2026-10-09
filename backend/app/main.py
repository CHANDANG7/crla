"""
ZECRO-RL — FastAPI Application Entry Point
"""
from __future__ import annotations

import os
import sys

# Ensure UTF-8 output encoding for Windows console (e.g. ₹ symbol in logs)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Disable SQLAlchemy Cython C-extensions to avoid Windows Application Control DLL blocks
os.environ["DISABLE_SQLALCHEMY_CEXT"] = "1"
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.config import settings
from app.database import close_db, init_db

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup → yield → shutdown."""
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    await init_db()
    logger.info("Database initialized")

    # Launch 24/7 Autonomous Market Monitoring & Execution Loop
    import asyncio
    from app.execution.paper_engine import paper_engine
    asyncio.create_task(paper_engine.start_autonomous_loop())
    logger.info("⚡ Autonomous 24/7 market monitoring task launched")

    yield


    # Shutdown
    logger.info("Shutting down...")
    await close_db()


# ── Application ────────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "ZECRO-RL: Reinforcement Technical Analysis Trading Agent. "
        "Delta Exchange + Groq LLM + PPO RL Policy"
    ),
    lifespan=lifespan,
)

# ── Middleware ─────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# ── Routes ─────────────────────────────────────────────────────────────────────
from app.api.v1 import dashboard, trades, market, analysis, rl_api, knowledge, ws_api  # noqa

app.include_router(dashboard.router, prefix="/api/v1/dashboard", tags=["Dashboard"])
app.include_router(trades.router, prefix="/api/v1/trades", tags=["Trades"])
app.include_router(market.router, prefix="/api/v1/market", tags=["Market"])
app.include_router(analysis.router, prefix="/api/v1/analysis", tags=["Analysis"])
app.include_router(rl_api.router, prefix="/api/v1/rl", tags=["RL"])
app.include_router(knowledge.router, prefix="/api/v1/knowledge", tags=["Knowledge"])
app.include_router(ws_api.router, prefix="/ws", tags=["WebSocket"])


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
    }


@app.get("/")
async def root():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "redoc": "/redoc",
    }
