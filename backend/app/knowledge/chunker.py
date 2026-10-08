"""
ZECRO-RL — TA Knowledge Chunker
Splits extracted book pages into structured concept chunks.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

import structlog

logger = structlog.get_logger(__name__)

# Key technical analysis concepts to tag
KNOWN_CONCEPTS = [
    "Market Structure",
    "Break of Structure",
    "Change of Character",
    "Liquidity Sweep",
    "Fair Value Gap",
    "Order Block",
    "Support and Resistance",
    "Pin Bar Reversal",
    "Engulfing Pattern",
    "Inside Bar",
    "RSI Divergence",
    "Moving Average Cross",
    "Volume Spread Analysis",
    "Risk Management",
    "Position Sizing",
    "Trend Following",
    "Range Trading",
]


class TAChunker:
    """Chunks text pages into semantic units with concept tagging."""

    def __init__(self, chunk_size: int = 600, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_pages(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split pages into overlapping text chunks and detect concept tags.
        """
        chunks = []

        for p in pages:
            text = p["text"]
            book = p["book"]
            page_num = p["page"]

            words = text.split()
            if not words:
                continue

            # Slide window over words
            start = 0
            while start < len(words):
                end = min(start + self.chunk_size, len(words))
                chunk_text = " ".join(words[start:end])

                if len(chunk_text) >= 100:
                    concept = self._detect_concept(chunk_text)
                    conditions, confirmations, invalidations = self._extract_rules(chunk_text)

                    chunks.append({
                        "concept": concept,
                        "chunk_type": "text",
                        "content": chunk_text,
                        "source_book": book,
                        "source_page": page_num,
                        "conditions": conditions,
                        "confirmation": confirmations,
                        "invalidation": invalidations,
                        "related_concepts": [c for c in KNOWN_CONCEPTS if c.lower() in chunk_text.lower()],
                    })

                if end == len(words):
                    break
                start += self.chunk_size - self.overlap

        logger.info("Chunking complete", total_chunks=len(chunks))
        return chunks

    def _detect_concept(self, text: str) -> str:
        """Find most prominent TA concept in text."""
        text_lower = text.lower()
        for concept in KNOWN_CONCEPTS:
            if concept.lower() in text_lower:
                return concept
        return "General Technical Analysis"

    def _extract_rules(self, text: str) -> tuple[List[str], List[str], List[str]]:
        """Extract condition, confirmation, and invalidation sentences if present."""
        sentences = re.split(r"[.!?]\s+", text)
        conditions = []
        confirmations = []
        invalidations = []

        for s in sentences:
            s_low = s.lower()
            if any(k in s_low for k in ["when", "if ", "condition", "setup requires", "look for"]):
                conditions.append(s.strip())
            if any(k in s_low for k in ["confirm", "signal", "trigger", "entry occurs"]):
                confirmations.append(s.strip())
            if any(k in s_low for k in ["invalid", "fails", "stop loss", "disproved"]):
                invalidations.append(s.strip())

        return conditions[:3], confirmations[:3], invalidations[:3]


chunker = TAChunker()
