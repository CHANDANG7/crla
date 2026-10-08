"""
ZECRO-RL — Vector Store Interface
Handles storing and similarity-searching TA knowledge chunks in PostgreSQL + pgvector.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session_factory
from app.knowledge.embedder import embedder
from app.models import TAKnowledge

logger = structlog.get_logger(__name__)


class VectorStore:
    """Interface to pgvector storage for TA concepts."""

    async def add_chunks(
        self, chunks: List[Dict[str, Any]], db: Optional[AsyncSession] = None
    ) -> int:
        """
        Embed and persist chunks to ta_knowledge table.
        """
        texts = [c["content"] for c in chunks]
        embeddings = embedder.embed_batch(texts)

        async def _save(session: AsyncSession):
            count = 0
            for chunk_data, emb in zip(chunks, embeddings):
                obj = TAKnowledge(
                    concept=chunk_data.get("concept", "General"),
                    chunk_type=chunk_data.get("chunk_type", "text"),
                    content=chunk_data["content"],
                    conditions=chunk_data.get("conditions", []),
                    confirmation=chunk_data.get("confirmation", []),
                    invalidation=chunk_data.get("invalidation", []),
                    related_concepts=chunk_data.get("related_concepts", []),
                    source_book=chunk_data.get("source_book"),
                    source_page=chunk_data.get("source_page"),
                    embedding=emb,
                )
                session.add(obj)
                count += 1
            await session.commit()
            return count

        if db is not None:
            return await _save(db)
        else:
            async with async_session_factory() as session:
                return await _save(session)

    async def search_similar(
        self, query: str, limit: int = 4, db: Optional[AsyncSession] = None
    ) -> List[Dict[str, Any]]:
        """
        Find top-k chunks closest to query vector using cosine distance.
        """
        query_emb = embedder.embed_text(query)

        async def _query(session: AsyncSession):
            try:
                # pgvector cosine distance: TAKnowledge.embedding.cosine_distance(query_emb)
                stmt = (
                    select(TAKnowledge)
                    .order_by(TAKnowledge.embedding.cosine_distance(query_emb))
                    .limit(limit)
                )
                result = await session.execute(stmt)
                chunks = result.scalars().all()
                return [
                    {
                        "concept": c.concept,
                        "content": c.content,
                        "book": c.source_book,
                        "page": c.source_page,
                        "conditions": c.conditions,
                        "confirmation": c.confirmation,
                        "invalidation": c.invalidation,
                    }
                    for c in chunks
                ]
            except Exception as e:
                logger.warning("Vector search query failed, using text fallback", error=str(e))
                # Fallback to ILIKE search
                stmt = (
                    select(TAKnowledge)
                    .where(TAKnowledge.content.ilike(f"%{query[:30]}%"))
                    .limit(limit)
                )
                result = await session.execute(stmt)
                chunks = result.scalars().all()
                return [
                    {
                        "concept": c.concept,
                        "content": c.content,
                        "book": c.source_book,
                        "page": c.source_page,
                        "conditions": c.conditions,
                        "confirmation": c.confirmation,
                        "invalidation": c.invalidation,
                    }
                    for c in chunks
                ]

        if db is not None:
            return await _query(db)
        else:
            async with async_session_factory() as session:
                return await _query(session)


vector_store = VectorStore()
