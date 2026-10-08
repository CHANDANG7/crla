"""
ZECRO-RL — TA Knowledge Retriever
Retrieves and formats technical analysis knowledge context for Groq LLM reasoning.
"""
from __future__ import annotations

from typing import List, Optional

import structlog

from app.knowledge.vector_store import vector_store
from app.ta.feature_engine import MarketState

logger = structlog.get_logger(__name__)


class TARetriever:
    """RAG retriever for technical analysis literature."""

    async def get_context_for_market_state(
        self, state: MarketState, limit: int = 3
    ) -> Optional[str]:
        """
        Synthesize relevant TA book concepts based on current market state patterns.
        """
        # Formulate query based on key detected patterns
        query_terms = []
        if state.prev_low_swept or state.liquidity_grab_bull:
            query_terms.append("Sell side liquidity sweep reversal entry stop loss")
        elif state.prev_high_swept or state.liquidity_grab_bear:
            query_terms.append("Buy side liquidity sweep reversal entry stop loss")
        elif state.bos_bullish or state.choch_bullish:
            query_terms.append("Bullish break of structure change of character continuation")
        elif state.bos_bearish or state.choch_bearish:
            query_terms.append("Bearish break of structure change of character continuation")
        elif state.rsi_divergence != 0:
            query_terms.append("RSI divergence momentum exhaustion reversal setup")
        elif state.in_range:
            query_terms.append("Range trading support resistance breakout failure")
        else:
            query_terms.append("Trend following moving average pullback high probability entry")

        query_str = " ".join(query_terms)

        try:
            chunks = await vector_store.search_similar(query_str, limit=limit)
            if not chunks:
                return None

            formatted = []
            for i, c in enumerate(chunks, 1):
                source = f"{c.get('book', 'TA Reference')} (Page {c.get('page', '?')})"
                formatted.append(
                    f"### Reference {i}: {c.get('concept', 'Concept')} — {source}\n"
                    f"{c['content'][:400]}...\n"
                )

            return "\n".join(formatted)

        except Exception as e:
            logger.warning("Knowledge retrieval failed", error=str(e))
            return None


ta_retriever = TARetriever()
