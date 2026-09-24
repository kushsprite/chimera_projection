"""In-memory vector index and retrieval.

At our scale (tens of chunks per match) a numpy matrix plus cosine similarity
is all we need; no vector database. The index tries Gemini embeddings first
and transparently falls back to TF-IDF if Gemini is unavailable, so retrieval
always works. Query and documents always use the same backend.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .chunking import Chunk
from .embeddings import EmbeddingUnavailable, GeminiEmbedder, TfidfEmbedder

log = logging.getLogger(__name__)


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


@dataclass
class Hit:
    chunk: Chunk
    score: float
    query: str


class VectorIndex:
    def __init__(self, chunks: list[Chunk], gemini: Optional[GeminiEmbedder]):
        self.chunks = chunks
        self.gemini = gemini
        self.backend = "none"
        self.fallback_reason: Optional[str] = None
        self._embedder = None
        self._matrix = np.zeros((0, 1), dtype=np.float32)
        if chunks:
            self._build()

    def _build(self) -> None:
        texts = [c.text for c in self.chunks]
        if self.gemini is not None and self.gemini.available:
            try:
                self._matrix = _normalize(self.gemini.embed(texts, "RETRIEVAL_DOCUMENT"))
                self._embedder, self.backend = self.gemini, "gemini"
                return
            except EmbeddingUnavailable as e:
                self.fallback_reason = str(e)
                log.warning("Falling back to TF-IDF retrieval: %s", e)
        elif self.gemini is not None:
            self.fallback_reason = "GEMINI_API_KEY not set"
        self._use_tfidf(texts)

    def _use_tfidf(self, texts: list[str]) -> None:
        tfidf = TfidfEmbedder().fit(texts)
        self._matrix = _normalize(tfidf.embed(texts))
        self._embedder, self.backend = tfidf, "tfidf"

    def _embed_query(self, query: str) -> np.ndarray:
        if self.backend == "gemini":
            try:
                return _normalize(self._embedder.embed([query], "RETRIEVAL_QUERY"))[0]
            except EmbeddingUnavailable as e:
                # quota ran out between indexing and querying: rebuild the whole index on TF-IDF
                self.fallback_reason = str(e)
                log.warning("Query embedding failed, rebuilding index on TF-IDF: %s", e)
                self._use_tfidf([c.text for c in self.chunks])
        return _normalize(self._embedder.embed([query]))[0]

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        if not self.chunks:
            return []
        q = self._embed_query(query)
        scores = self._matrix @ q
        order = np.argsort(-scores)[:top_k]
        return [Hit(self.chunks[i], float(scores[i]), query) for i in order]

    def multi_search(self, queries: list[str], per_query: int = 3, total: int = 6,
                     max_per_doc: int = 2) -> list[Hit]:
        """Run several focused queries and merge, keeping each chunk once at its best score.

        max_per_doc stops one long article from crowding out every other source.
        """
        best: dict[str, Hit] = {}
        for q in queries:
            for h in self.search(q, per_query):
                cid = h.chunk.chunk_id
                if cid not in best or h.score > best[cid].score:
                    best[cid] = h
        ranked = sorted(best.values(), key=lambda h: h.score, reverse=True)
        out, per_doc = [], {}
        for h in ranked:
            did = h.chunk.doc.doc_id
            if per_doc.get(did, 0) >= max_per_doc:
                continue
            per_doc[did] = per_doc.get(did, 0) + 1
            out.append(h)
            if len(out) >= total:
                break
        return out
