"""
ZECRO-RL — Higher Timeframe (1H) Bias Engine
Combines deterministic technical structure + Groq LLM reasoning into actionable directional bias.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import structlog

from app.llm.analysis_agent import TAAnalysisOutput, groq_agent
from app.models import BiasEnum
from app.strategy.regime_detector import regime_detector
from app.ta.feature_engine import MarketState

logger = structlog.get_logger(__name__)


@dataclass
class BiasResult:
    symbol: str
    timeframe: str
    bias: BiasEnum
    confidence: float
    setup_type: str
    regime: str
    evidence: List[str]
    invalidation: str
    llm_output: Optional[TAAnalysisOutput] = None


class BiasEngine:
    """
    1H Bias Engine.
    Evaluates market regime, structure, liquidity, and LLM reasoning.
    """

    async def compute_bias(
        self,
        state_1h: MarketState,
        use_llm: bool = True,
        knowledge_context: Optional[str] = None,
    ) -> BiasResult:
        """
        Compute higher-timeframe directional bias.
        """
        regime_res = regime_detector.detect(state_1h)
        regime = regime_res.regime.value

        evidence = []
        deterministic_bias = BiasEnum.NEUTRAL
        score = 0  # positive = bullish, negative = bearish

        # 1. Structure evaluation
        if state_1h.trend_direction == "BULLISH" or state_1h.higher_high or state_1h.higher_low:
            score += 2
            evidence.append(f"Bullish structure ({state_1h.trend_direction})")
        elif state_1h.trend_direction == "BEARISH" or state_1h.lower_high or state_1h.lower_low:
            score -= 2
            evidence.append(f"Bearish structure ({state_1h.trend_direction})")

        if state_1h.bos_bullish:
            score += 2
            evidence.append("Bullish Break of Structure (BOS)")
        elif state_1h.bos_bearish:
            score -= 2
            evidence.append("Bearish Break of Structure (BOS)")

        if state_1h.choch_bullish:
            score += 3
            evidence.append("Bullish Change of Character (CHoCH)")
        elif state_1h.choch_bearish:
            score -= 3
            evidence.append("Bearish Change of Character (CHoCH)")

        # 2. Liquidity evaluation
        if state_1h.prev_low_swept or state_1h.liquidity_grab_bull:
            score += 2
            evidence.append("Sell-side liquidity swept")
        if state_1h.prev_high_swept or state_1h.liquidity_grab_bear:
            score -= 2
            evidence.append("Buy-side liquidity swept")

        # 3. Momentum & EMAs
        if state_1h.price_vs_ema50 > 0 and state_1h.ema50_slope > 0:
            score += 1
            evidence.append("Price above ascending 50 EMA")
        elif state_1h.price_vs_ema50 < 0 and state_1h.ema50_slope < 0:
            score -= 1
            evidence.append("Price below descending 50 EMA")

        if state_1h.rsi_divergence == 1:
            score += 2
            evidence.append("Bullish RSI divergence")
        elif state_1h.rsi_divergence == -1:
            score -= 2
            evidence.append("Bearish RSI divergence")

        # Determine deterministic bias
        if score >= 3:
            deterministic_bias = BiasEnum.LONG
            setup_type = "BULLISH_CONTINUATION_OR_REVERSAL"
        elif score <= -3:
            deterministic_bias = BiasEnum.SHORT
            setup_type = "BEARISH_CONTINUATION_OR_REVERSAL"
        else:
            deterministic_bias = BiasEnum.NEUTRAL
            setup_type = "RANGE_OR_CONSOLIDATION"

        confidence = min(0.95, max(0.40, abs(score) / 10.0))
        invalidation = (
            f"Below swing low {state_1h.swing_low}"
            if deterministic_bias == BiasEnum.LONG
            else f"Above swing high {state_1h.swing_high}"
            if deterministic_bias == BiasEnum.SHORT
            else "Break of range boundaries"
        )

        llm_output = None
        if use_llm:
            try:
                llm_output = await groq_agent.analyze_1h(
                    state=state_1h, knowledge_context=knowledge_context
                )
                if llm_output and llm_output.bias != "NEUTRAL":
                    # If LLM produces structured output, combine confidence
                    final_bias = (
                        BiasEnum.LONG
                        if llm_output.bias == "LONG"
                        else BiasEnum.SHORT
                        if llm_output.bias == "SHORT"
                        else BiasEnum.NEUTRAL
                    )
                    return BiasResult(
                        symbol=state_1h.symbol,
                        timeframe=state_1h.timeframe,
                        bias=final_bias,
                        confidence=round((confidence + llm_output.confidence) / 2.0, 3),
                        setup_type=llm_output.setup_type,
                        regime=llm_output.market_regime or regime,
                        evidence=llm_output.evidence or evidence,
                        invalidation=llm_output.invalidation or invalidation,
                        llm_output=llm_output,
                    )
            except Exception as e:
                logger.warning("LLM bias analysis fallback to deterministic", error=str(e))

        return BiasResult(
            symbol=state_1h.symbol,
            timeframe=state_1h.timeframe,
            bias=deterministic_bias,
            confidence=round(confidence, 3),
            setup_type=setup_type,
            regime=regime,
            evidence=evidence,
            invalidation=invalidation,
            llm_output=llm_output,
        )


bias_engine = BiasEngine()
