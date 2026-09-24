"""Text embeddings with a persistent cache and a no-network fallback.

GeminiEmbedder   gemini-embedding-2 via google-genai. Texts are embedded one
                 per call, because batched calls returned a single vector in
                 our testing. Every vector is cached in SQLite, keyed by a hash
                 of (model, task_type, text), so identical text never costs quota
                 twice.
TfidfEmbedder    pure-numpy TF-IDF over unigrams+bigrams. Used automatically
                 when Gemini is unavailable (no key, quota exhausted, offline).
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
import sqlite3
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


class EmbeddingUnavailable(RuntimeError):
    """Raised when the embedding provider cannot be used right now."""


class EmbeddingCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB NOT NULL)")
        self._conn.commit()
        self._lock = threading.Lock()

    @staticmethod
    def key(model: str, task: str, text: str) -> str:
        return hashlib.sha256(f"{model}\x00{task}\x00{text}".encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[np.ndarray]:
        with self._lock:
            row = self._conn.execute("SELECT vec FROM emb WHERE key = ?", (key,)).fetchone()
        return np.frombuffer(row[0], dtype=np.float32) if row else None

    def put(self, key: str, vec: np.ndarray) -> None:
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO emb (key, vec) VALUES (?, ?)",
                               (key, np.asarray(vec, dtype=np.float32).tobytes()))
            self._conn.commit()


def _is_quota_error(e: Exception) -> bool:
    s = str(e)
    return "RESOURCE_EXHAUSTED" in s or "429" in s or "quota" in s.lower()


def _is_not_found(e: Exception) -> bool:
    s = str(e)
    return "NOT_FOUND" in s or "404" in s


class GeminiEmbedder:
    name = "gemini"

    def __init__(self, api_key: Optional[str], model: str, cache: EmbeddingCache, client=None):
        self.api_key = api_key
        self.model = model
        self.cache = cache
        self._client = client
        self._resolved_model: Optional[str] = None

    @property
    def available(self) -> bool:
        return bool(self.api_key) or self._client is not None

    def _get_client(self):
        if self._client is None:
            if not self.api_key:
                raise EmbeddingUnavailable("GEMINI_API_KEY is not set")
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _discover_model(self) -> str:
        client = self._get_client()
        names = []
        for m in client.models.list():
            actions = getattr(m, "supported_actions", None) or []
            if "embedContent" in actions:
                names.append(m.name.replace("models/", ""))
        for pref in ("gemini-embedding-2", "gemini-embedding-001"):
            if pref in names:
                return pref
        stable = [n for n in names if "preview" not in n]
        if stable or names:
            return (stable or names)[0]
        raise EmbeddingUnavailable("No Gemini model supports embedContent for this key")

    def _embed_one(self, text: str, task_type: str) -> np.ndarray:
        from google.genai import types
        client = self._get_client()
        model = self._resolved_model or self.model
        last = None
        for attempt in range(4):
            try:
                resp = client.models.embed_content(
                    model=model, contents=text,
                    config=types.EmbedContentConfig(task_type=task_type),
                )
                return np.asarray(resp.embeddings[0].values, dtype=np.float32)
            except Exception as e:
                last = e
                if _is_quota_error(e):
                    raise EmbeddingUnavailable(f"Gemini embedding quota exhausted: {e}") from e
                if _is_not_found(e) and self._resolved_model is None:
                    self._resolved_model = model = self._discover_model()
                    log.warning("Embedding model %s not found, using %s", self.model, model)
                    continue
                time.sleep(2 ** attempt)
        raise EmbeddingUnavailable(f"Gemini embedding failed: {last}")

    def embed(self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
        vecs = []
        for t in texts:
            model = self._resolved_model or self.model
            v = self.cache.get(self.cache.key(model, task_type, t))
            if v is None:
                v = self._embed_one(t, task_type)
                self.cache.put(self.cache.key(self._resolved_model or model, task_type, t), v)
            vecs.append(v)
        return np.vstack(vecs) if vecs else np.zeros((0, 1), dtype=np.float32)


_STOP = set("""a an the and or of to in on for at by with from is are was were be been this that
these those it its as into than then there their they he she his her we our you your not no
but if so do does did has have had will would can could should may might also after before
over under about more most very just""".split())


def _tokens(text: str) -> list[str]:
    words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 1]
    return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]


class TfidfEmbedder:
    """Fitted on the chunks of one request; queries are projected into that space."""

    name = "tfidf"

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.idf: Optional[np.ndarray] = None

    def fit(self, texts: list[str]) -> "TfidfEmbedder":
        docs = [set(_tokens(t)) for t in texts]
        df = Counter(tok for d in docs for tok in d)
        self.vocab = {tok: i for i, tok in enumerate(sorted(df))}
        n = max(len(texts), 1)
        self.idf = np.array([math.log((1 + n) / (1 + df[t])) + 1.0 for t in sorted(df)], dtype=np.float32)
        return self

    def embed(self, texts: list[str], task_type: str = "") -> np.ndarray:
        if self.idf is None:
            raise RuntimeError("TfidfEmbedder must be fitted first")
        m = np.zeros((len(texts), max(len(self.vocab), 1)), dtype=np.float32)
        for r, t in enumerate(texts):
            counts = Counter(tok for tok in _tokens(t) if tok in self.vocab)
            for tok, c in counts.items():
                m[r, self.vocab[tok]] = (1 + math.log(c)) * self.idf[self.vocab[tok]]
        return m
