"""
ZECRO-RL — Database Layer
Async SQLAlchemy engine + session factory for TimescaleDB.
"""
from __future__ import annotations

import os

# Disable SQLAlchemy Cython C-extensions to avoid Windows Application Control DLL blocks
os.environ["DISABLE_SQLALCHEMY_CEXT"] = "1"

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.config import settings


# ── Engine ─────────────────────────────────────────────────────────────────────
engine: AsyncEngine = create_async_engine(
    settings.database_url,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    pool_pre_ping=True,
    echo=settings.debug,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Alias for scripts & services importing async_session_factory
async_session_factory = AsyncSessionLocal


# ── Base model ─────────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


# ── Dependency ─────────────────────────────────────────────────────────────────
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """For use outside FastAPI dependency injection."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Initialize database, create extensions and tables."""
    async with engine.begin() as conn:
        # Enable pgvector extension
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        # Enable TimescaleDB extension
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE"))
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    await engine.dispose()
