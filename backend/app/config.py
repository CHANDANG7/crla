"""
ZECRO-RL — Configuration
All settings loaded from environment variables / .env file.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ────────────────────────────────────────────────────────────
    app_name: str = "ZECRO-RL"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    log_file: str = "./logs/zecro_rl.log"

    # ── API ────────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_secret_key: str = "change_me_in_production"
    cors_origins: List[str] = ["http://localhost:5173", "http://localhost:3000"]

    # ── Database ───────────────────────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://zecro:zecro_pass@localhost:5432/zecro_rl"
    database_sync_url: str = "postgresql://zecro:zecro_pass@localhost:5432/zecro_rl"
    db_pool_size: int = 20
    db_max_overflow: int = 40

    # ── Delta Exchange ─────────────────────────────────────────────────────────
    delta_api_key: str = ""
    delta_api_secret: str = ""
    delta_base_url: str = "https://api.delta.exchange"
    delta_ws_url: str = "wss://socket.delta.exchange"

    # ── Groq ───────────────────────────────────────────────────────────────────
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    groq_temperature: float = 0.1
    groq_max_tokens: int = 2048

    # ── Embeddings ─────────────────────────────────────────────────────────────
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    openai_api_key: str = ""

    # ── Paper Trading ──────────────────────────────────────────────────────────
    initial_capital_inr: float = 10_000_000.0   # ₹1 crore
    risk_per_trade: float = 0.0025               # 0.25%
    max_daily_loss_inr: float = 75_000.0
    max_portfolio_dd: float = 0.10               # 10%
    max_open_positions: int = 3
    paper_trading_mode: bool = True
    default_leverage: int = 1

    # ── Fee Model (Delta Exchange) ─────────────────────────────────────────────
    maker_fee: float = 0.0002   # 0.02%
    taker_fee: float = 0.0005   # 0.05%
    slippage_bps: float = 5.0   # 0.05% base slippage

    # ── RL ─────────────────────────────────────────────────────────────────────
    rl_algorithm: str = "PPO"
    rl_learning_rate: float = 3e-4
    rl_gamma: float = 0.99
    rl_clip_epsilon: float = 0.2
    rl_entropy_coef: float = 0.01
    rl_n_steps: int = 2048
    rl_batch_size: int = 64
    rl_n_epochs: int = 10
    rl_models_dir: str = "./models/policies"
    rl_state_dim: int = 160
    rl_action_dim: int = 4  # HOLD, LONG, SHORT, EXIT

    # ── Assets ─────────────────────────────────────────────────────────────────
    symbols: List[str] = [
        "BTCUSD", "ETHUSD", "SOLUSD", "BNBUSD", "XRPUSD", "DOGEUSD", "ADAUSD", "AVAXUSD",
        "LINKUSD", "DOTUSD", "SUIUSD", "APTUSD", "NEARUSD", "POLUSD", "MATICUSD", "ARBUSD",
        "OPUSD", "TIAUSD", "SEIUSD", "ATOMUSD", "FTMUSD", "KASUSD", "TRXUSD", "LTCUSD",
        "BCHUSD", "ICPUSD", "UNIUSD", "AAVEUSD", "INJUSD", "PENDLEUSD", "RUNEUSD", "LDOUSD",
        "MKRUSD", "RENDERUSD", "FETUSD", "TAOUSD", "GRTUSD", "WLDUSD", "FILUSD", "PEPEUSD",
        "WIFUSD", "SHIBUSD", "BONKUSD", "FLOKIUSD", "POPCATUSD", "PAXGUSD", "XAUTUSD", "SLVONUSD",
        "SNDKBUSD", "TSLAXUSD", "SOXLBUSD", "GOOGLXUSD", "SPCXXUSD", "SPYXUSD", "METAXUSD", "MSTRBUSD",
        "NBISBUSD", "NVDAXUSD", "QQQXUSD", "MRVLBUSD", "AAPLXUSD", "MUBUSD", "DRAMBUSD", "AMZNXUSD",
        "CBRSBUSD", "LITEBUSD", "WDCBUSD", "SKHYBUSD", "PLTRBUSD", "RKLBBUSD", "INTCBUSD", "CRCLXUSD",
        "COINXUSD", "ARMBUSD", "EWYBUSD", "BABABUSD", "HOODBUSD", "AMDBUSD", "TSMBUSD", "AVAXUSDT", "INJUSDT"
    ]
    default_symbol: str = "BTCUSD"

    timeframe_analysis: str = "1h"
    timeframe_entry: str = "15m"
    timeframes_all: List[str] = ["1m", "5m", "15m", "1h", "4h", "1d"]
    default_timeframes: List[str] = ["1h", "15m", "5m", "1m"]
    primary_timeframe: str = "1h"

    # ── Fee Model & Slippage ───────────────────────────────────────────────────
    maker_fee: float = 0.0002   # 0.02%
    taker_fee: float = 0.0005   # 0.05%
    slippage_bps: float = 5.0   # 0.05% base slippage
    slippage_default: float = 0.0005
    trailing_stop_pct: float = 0.5  # 0.5% trailing stop

    # ── Risk reward ────────────────────────────────────────────────────────────
    reward_tp1: float = 1.0    # +1R
    reward_tp2: float = 2.0    # +2R
    reward_sl: float = -1.0    # -1R
    reward_tsl: float = 0.5    # +0.5R minimum
    # Partial TP splits
    tp1_portion: float = 0.50  # 50% at TP1
    tp2_portion: float = 0.25  # 25% at TP2
    tsl_portion: float = 0.25  # 25% trailing stop

    # ── Telegram ───────────────────────────────────────────────────────────────
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # ── Redis ──────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── Paths ──────────────────────────────────────────────────────────────────
    books_dir: str = "/books"
    models_dir: str = "./models/policies"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors(cls, v):
        if isinstance(v, str):
            import json
            return json.loads(v)
        return v

    @field_validator("default_timeframes", mode="before")
    @classmethod
    def parse_timeframes(cls, v):
        if isinstance(v, str):
            import json
            return json.loads(v)
        return v

    @property
    def models_path(self) -> Path:
        return Path(self.models_dir)

    @property
    def books_path(self) -> Path:
        return Path(self.books_dir)


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
