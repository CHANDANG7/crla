"""
ZECRO-RL — Text Embedder
Generates 384-dimensional dense embeddings for pgvector storage and semantic search.
"""
from __future__ import annotations

import hashlib
from typing import List, Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class Embedder:
    """Embedding generator using sentence-transformers (384-dim)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                logger.info("Loaded embedding model", model=self.model_name)
            except Exception as e:
                logger.warning(
                    "sentence-transformers not available or offline. Using deterministic embedding fallback.",
                    error=str(e),
                )
        return self._model

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string to a 384-dim vector."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of text strings to 384-dim vectors."""
        model = self._get_model()
        if model is not None:
            try:
                embeddings = model.encode(texts, normalize_embeddings=True)
                return [emb.tolist() for emb in embeddings]
            except Exception as e:
                logger.error("Embedding generation failed, using fallback", error=str(e))

        # Deterministic fallback: Generate 384-dim normalized pseudo-vector via hashing
        fallback_embs = []
        for t in texts:
            # Seed pseudo-random generator with text hash
            seed = int(hashlib.md5(t.encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.RandomState(seed)
            vec = rng.randn(384).astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            fallback_embs.append(vec.tolist())

        return fallback_embs


embedder = Embedder()
