"""
ZECRO-RL — TA Books Ingestion Pipeline
Extracts text from PDF books in books/ folder, chunks concepts, embeds vectors, and indexes in pgvector.

Usage:
    python scripts/ingest_books.py --path ../books
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add app parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import structlog

from app.database import init_db
from app.knowledge.chunker import chunker
from app.knowledge.pdf_extractor import pdf_extractor
from app.knowledge.vector_store import vector_store

logger = structlog.get_logger(__name__)


async def ingest_directory(books_path: Path):
    """Scan directory for PDF files and index into knowledge base."""
    if not books_path.exists():
        logger.error("Books directory does not exist", path=str(books_path))
        return

    pdf_files = list(books_path.glob("*.pdf")) + list(books_path.glob("**/*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {books_path}. Place your TA PDFs there and rerun.")
        return

    logger.info(f"Found {len(pdf_files)} PDF books to ingest.")

    total_chunks = 0
    for pdf_path in pdf_files:
        logger.info(f"Processing {pdf_path.name}...")
        pages = pdf_extractor.extract(pdf_path)
        if not pages:
            logger.warning(f"Could not extract pages from {pdf_path.name}")
            continue

        chunks = chunker.chunk_pages(pages)
        if not chunks:
            continue

        saved_count = await vector_store.add_chunks(chunks)
        total_chunks += saved_count
        logger.info(f"Ingested {saved_count} chunks from {pdf_path.name}")

    logger.info(f"Knowledge ingestion complete! Total {total_chunks} chunks stored in pgvector.")


async def main():
    parser = argparse.ArgumentParser(description="Ingest TA Books into Vector Store")
    parser.add_argument(
        "--path",
        type=str,
        default="../books",
        help="Path to folder containing PDF books",
    )
    args = parser.parse_args()

    await init_db()
    books_dir = Path(args.path).resolve()
    await ingest_directory(books_dir)


if __name__ == "__main__":
    asyncio.run(main())
