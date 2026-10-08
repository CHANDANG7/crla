"""
ZECRO-RL — Groq Analysis Agent
Uses Groq LLM with structured JSON output + tool calling for TA reasoning.
Groq is the REASONING layer — NOT the trading decision maker.
The RL policy makes the final action decision.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import structlog
from groq import AsyncGroq
from pydantic import BaseModel, Field

from app.config import settings
from app.ta.feature_engine import MarketState

logger = structlog.get_logger(__name__)


# ── Structured Output Schemas ──────────────────────────────────────────────────

class TAAnalysisOutput(BaseModel):
    """Structured output from Groq 1H analysis agent."""
    bias: str = Field(..., description="LONG, SHORT, or NEUTRAL")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence 0-1")
    setup_type: str = Field(..., description="e.g. LIQUIDITY_SWEEP_BOS, BREAKOUT, PULLBACK")
    market_regime: str = Field(..., description="TRENDING_UP, TRENDING_DOWN, RANGING, etc.")
    evidence: List[str] = Field(..., description="List of supporting evidence points")
    invalidation: str = Field(..., description="What would invalidate this bias")
    entry_type: str = Field("market", description="market or limit")
    risk_model: str = Field("1R", description="1R, 0.5R, etc.")
    key_levels: Dict[str, Optional[float]] = Field(
        default_factory=dict,
        description="Important price levels"
    )
    notes: Optional[str] = Field(None, description="Additional context")


class EntryAnalysisOutput(BaseModel):
    """15M entry signal from Groq entry agent."""
    entry_valid: bool = Field(..., description="Is entry condition met?")
    entry_setup: str = Field(..., description="Entry pattern name")
    entry_confidence: float = Field(..., ge=0.0, le=1.0)
    suggested_entry: Optional[float] = Field(None, description="Suggested entry price")
    suggested_sl: Optional[float] = Field(None, description="Suggested stop loss")
    suggested_tp1: Optional[float] = Field(None, description="Suggested TP1")
    suggested_tp2: Optional[float] = Field(None, description="Suggested TP2")
    entry_reasons: List[str] = Field(default_factory=list)
    wait_for: Optional[str] = Field(None, description="What to wait for if not ready")


# ── System Prompts ─────────────────────────────────────────────────────────────

ANALYSIS_SYSTEM_PROMPT = """You are ZECRO, an expert technical analysis engine for cryptocurrency markets.

Your role is to analyze market structure and produce structured TA reasoning.
You are NOT making trading decisions — the RL policy does that.
You are providing structured market context.

You follow these principles:
1. Market structure > indicators (structure first, then momentum)
2. Liquidity sweeps are high-probability setups
3. Higher timeframe bias always overrides lower timeframe
4. No trade is better than a bad trade
5. Always define invalidation before entry

Analyze with extreme precision. No vague language.
Output ONLY valid JSON matching the required schema."""

ENTRY_SYSTEM_PROMPT = """You are ZECRO Entry Engine analyzing 15-minute charts for specific entry signals.

Given:
- 1H bias from analysis agent
- Current 15M market state

Determine if a valid entry setup exists on the 15M.

A valid LONG entry requires:
- Sell-side liquidity sweep (price goes below key low then reverses)
- Bullish CHoCH or BOS on 15M
- Volume expansion on the reversal candle
- Price closing above swept low

A valid SHORT entry requires:
- Buy-side liquidity sweep (price goes above key high then reverses)
- Bearish CHoCH or BOS on 15M  
- Volume expansion on the reversal candle
- Price closing below swept high

Output ONLY valid JSON matching the required schema."""


class GroqAnalysisAgent:
    """
    Groq-powered 1H TA analysis agent.
    Uses structured JSON output for safe, parseable responses.
    """

    def __init__(self):
        self.client = AsyncGroq(api_key=settings.groq_api_key)
        self.model = settings.groq_model

    async def analyze_1h(
        self,
        state: MarketState,
        knowledge_context: Optional[str] = None,
    ) -> TAAnalysisOutput:
        """
        Analyze 1H market state and produce structured TA output.

        Args:
            state: 1H MarketState from FeatureEngine
            knowledge_context: Relevant TA book knowledge from RAG

        Returns:
            TAAnalysisOutput with structured bias and reasoning
        """
        state_dict = state.to_dict()

        # Build user message
        user_message = self._build_analysis_prompt(state_dict, knowledge_context)

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=settings.groq_temperature,
                max_tokens=settings.groq_max_tokens,
                response_format={"type": "json_object"},
            )

            raw = response.choices[0].message.content
            data = json.loads(raw)
            result = TAAnalysisOutput(**data)

            logger.info(
                "1H Analysis complete",
                symbol=state.symbol,
                bias=result.bias,
                confidence=result.confidence,
                setup=result.setup_type,
            )

            return result

        except Exception as e:
            logger.error("Groq analysis failed", error=str(e))
            # Return neutral bias on failure — never crash the trading loop
            return TAAnalysisOutput(
                bias="NEUTRAL",
                confidence=0.0,
                setup_type="ANALYSIS_ERROR",
                market_regime="UNKNOWN",
                evidence=["Analysis failed"],
                invalidation="N/A",
                risk_model="0R",
            )

    def _build_analysis_prompt(
        self, state: Dict[str, Any], knowledge: Optional[str]
    ) -> str:
        """Build structured analysis prompt."""

        prompt = f"""Analyze the following 1-hour market state for {state.get('symbol', 'UNKNOWN')}:

## Market Structure
- Trend: {state.get('trend_direction')}
- BOS Bullish: {state.get('bos_bullish')} | BOS Bearish: {state.get('bos_bearish')}
- CHoCH Bullish: {state.get('choch_bullish')} | CHoCH Bearish: {state.get('choch_bearish')}
- Higher High: {state.get('higher_high')} | Higher Low: {state.get('higher_low')}
- Lower High: {state.get('lower_high')} | Lower Low: {state.get('lower_low')}
- In Range: {state.get('in_range')} | Breakout: {state.get('breakout')} | Breakdown: {state.get('breakdown')}
- Swing High: {state.get('swing_high')} | Swing Low: {state.get('swing_low')}

## Price Action
- Candle Type: {state.get('candle_type')}
- Engulfing Bull: {state.get('engulfing_bull')} | Engulfing Bear: {state.get('engulfing_bear')}
- Pin Bar Bull: {state.get('pin_bar_bull')} | Pin Bar Bear: {state.get('pin_bar_bear')}
- Rejection Bull: {state.get('rejection_bull')} | Rejection Bear: {state.get('rejection_bear')}
- Momentum Bull: {state.get('momentum_bull')} | Momentum Bear: {state.get('momentum_bear')}
- Body %: {state.get('candle_body_pct', 0):.2f}

## Liquidity
- Prev High Swept: {state.get('prev_high_swept')} | Prev Low Swept: {state.get('prev_low_swept')}
- Equal Highs: {state.get('equal_highs')} | Equal Lows: {state.get('equal_lows')}
- Liquidity Grab Bull: {state.get('liquidity_grab_bull')}
- Liquidity Grab Bear: {state.get('liquidity_grab_bear')}
- BSL Level: {state.get('bsl_level')} ({state.get('dist_to_bsl_pct', 0):.2f}% away)
- SSL Level: {state.get('ssl_level')} ({state.get('dist_to_ssl_pct', 0):.2f}% away)

## Momentum
- RSI: {state.get('rsi', 50):.1f} | RSI Divergence: {state.get('rsi_divergence')}
- ADX: {state.get('adx', 0):.1f} | Trend Strong: {state.get('trend_strong')}
- MACD Hist: {state.get('macd_hist', 0):.4f} | MACD Cross: {state.get('macd_cross')}
- Price vs EMA20: {state.get('price_vs_ema20', 0):.2f}%
- Price vs EMA50: {state.get('price_vs_ema50', 0):.2f}%
- Price vs EMA200: {state.get('price_vs_ema200', 0):.2f}%

## Volatility
- ATR (normalized): {state.get('atr_normalized', 0):.3f}%
- ATR Percentile: {state.get('atr_percentile', 50):.0f}
- BB Width: {state.get('bb_width', 0):.2f}%
- Range Expansion: {state.get('range_expansion')} | Compression: {state.get('range_compression')}

## Volume
- Relative Volume: {state.get('relative_volume', 1):.2f}x
- Volume Spike: {state.get('volume_spike')}
- Volume Divergence Bull: {state.get('volume_divergence_bull')}
- Volume Divergence Bear: {state.get('volume_divergence_bear')}
"""

        if knowledge:
            prompt += f"""
## Relevant TA Knowledge
{knowledge}
"""

        prompt += """
Provide your analysis as JSON with these exact fields:
{
    "bias": "LONG" | "SHORT" | "NEUTRAL",
    "confidence": 0.0-1.0,
    "setup_type": "descriptive setup name",
    "market_regime": "TRENDING_UP" | "TRENDING_DOWN" | "RANGING" | "HIGH_VOLATILITY" | "LOW_VOLATILITY" | "BREAKOUT" | "BREAKDOWN",
    "evidence": ["reason 1", "reason 2", ...],
    "invalidation": "what would invalidate this bias",
    "entry_type": "market" | "limit",
    "risk_model": "1R",
    "key_levels": {"support": null, "resistance": null},
    "notes": "additional context"
}"""

        return prompt

    async def analyze_15m_entry(
        self,
        state_15m: MarketState,
        bias_1h: TAAnalysisOutput,
    ) -> EntryAnalysisOutput:
        """
        Analyze 15M chart for entry signal given 1H bias.

        Args:
            state_15m: 15M MarketState
            bias_1h: Analysis output from 1H agent

        Returns:
            EntryAnalysisOutput with entry signal
        """
        if bias_1h.bias == "NEUTRAL":
            return EntryAnalysisOutput(
                entry_valid=False,
                entry_setup="NO_BIAS",
                entry_confidence=0.0,
                entry_reasons=["No 1H directional bias"],
                wait_for="1H directional bias",
            )

        state_dict = state_15m.to_dict()
        current_price = state_15m.current_price

        user_message = f"""Given:

1H BIAS: {bias_1h.bias} (confidence: {bias_1h.confidence:.0%})
1H SETUP: {bias_1h.setup_type}
1H EVIDENCE: {', '.join(bias_1h.evidence[:3])}

15M MARKET STATE:
- Trend: {state_dict.get('trend_direction')}
- BOS: Bull={state_dict.get('bos_bullish')} Bear={state_dict.get('bos_bearish')}
- CHoCH: Bull={state_dict.get('choch_bullish')} Bear={state_dict.get('choch_bearish')}
- Prev Low Swept: {state_dict.get('prev_low_swept')}
- Prev High Swept: {state_dict.get('prev_high_swept')}
- Volume Spike: {state_dict.get('volume_spike')} (RVOL: {state_dict.get('relative_volume', 1):.2f}x)
- Candle: {state_dict.get('candle_type')}
- RSI: {state_dict.get('rsi', 50):.1f}
- Current Price: {current_price}

Determine if a valid 15M {bias_1h.bias} entry exists.

Output as JSON:
{{
    "entry_valid": true|false,
    "entry_setup": "setup name",
    "entry_confidence": 0.0-1.0,
    "suggested_entry": price or null,
    "suggested_sl": price or null,
    "suggested_tp1": price or null,
    "suggested_tp2": price or null,
    "entry_reasons": ["reason 1", "reason 2"],
    "wait_for": "what to wait for if not ready, else null"
}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": ENTRY_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=settings.groq_temperature,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )

            raw = response.choices[0].message.content
            data = json.loads(raw)
            result = EntryAnalysisOutput(**data)

            logger.info(
                "15M entry analysis",
                symbol=state_15m.symbol,
                valid=result.entry_valid,
                setup=result.entry_setup,
                confidence=result.entry_confidence,
            )

            return result

        except Exception as e:
            logger.error("Groq entry analysis failed", error=str(e))
            return EntryAnalysisOutput(
                entry_valid=False,
                entry_setup="ANALYSIS_ERROR",
                entry_confidence=0.0,
                entry_reasons=["Analysis failed"],
                wait_for="Retry",
            )


# Singleton
groq_agent = GroqAnalysisAgent()
