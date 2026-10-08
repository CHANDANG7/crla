"""
ZECRO-RL — TA Knowledge Base API
Endpoints for querying ingested TA books and concepts.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import distinct, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import TAKnowledge

router = APIRouter()


class SearchQuery(BaseModel):
    query: str
    concept: Optional[str] = None
    limit: int = 5


@router.get("/stats")
async def get_knowledge_stats(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Knowledge base statistics (total chunks, unique books, concepts)."""
    total_chunks = await db.scalar(select(func.count(TAKnowledge.id))) or 0
    unique_books = (
        await db.scalars(
            select(distinct(TAKnowledge.source_book)).where(
                TAKnowledge.source_book.isnot(None)
            )
        )
    ).all()
    unique_concepts = (
        await db.scalars(select(distinct(TAKnowledge.concept)))
    ).all()

    return {
        "total_chunks": total_chunks,
        "books_indexed": list(unique_books),
        "total_books": len(unique_books),
        "concepts_indexed": list(unique_concepts),
        "total_concepts": len(unique_concepts),
    }


@router.get("/concepts")
async def get_concepts(db: AsyncSession = Depends(get_db)) -> List[str]:
    """List all unique TA concepts indexed in knowledge base."""
    concepts = (
        await db.scalars(select(distinct(TAKnowledge.concept)).order_by(TAKnowledge.concept))
    ).all()
    return list(concepts)


@router.get("/search")
async def search_knowledge(
    q: str = Query(..., min_length=2),
    concept: Optional[str] = None,
    limit: int = Query(5, ge=1, le=20),
    db: AsyncSession = Depends(get_db),
) -> List[Dict[str, Any]]:
    """Text-based search in knowledge base chunks."""
    query = select(TAKnowledge).where(
        TAKnowledge.content.ilike(f"%{q}%")
    ).limit(limit)

    if concept:
        query = query.where(TAKnowledge.concept == concept)

    result = await db.execute(query)
    chunks = result.scalars().all()

    return [
        {
            "id": c.id,
            "concept": c.concept,
            "chunk_type": c.chunk_type,
            "content": c.content,
            "source_book": c.source_book,
            "source_page": c.source_page,
            "conditions": c.conditions,
            "confirmation": c.confirmation,
            "invalidation": c.invalidation,
        }
        for c in chunks
    ]


@router.post("/query")
async def query_knowledge(
    req: SearchQuery,
    db: AsyncSession = Depends(get_db),
) -> List[Dict[str, Any]]:
    """Query knowledge base for relevant concepts/conditions."""
    return await search_knowledge(
        q=req.query, concept=req.concept, limit=req.limit, db=db
    )
