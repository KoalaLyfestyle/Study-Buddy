"""Embedding backends.

`SentenceTransformerEmbedder` is the real one (BAAI/bge-small-en, run
locally). `HashingEmbedder` needs no model download. It is deterministic
and lexical only, so tests and offline checks use it.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Protocol, Sequence

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    id: str  # part of the cache key: change it and everything is re-embedded

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class HashingEmbedder:
    """Hashed unigrams and bigrams. Offline, fast, not semantic."""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.id = f"hashing-v1-{dim}"

    def _vector(self, text: str) -> np.ndarray:
        tokens = _TOKEN.findall(text.lower())
        features = tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]
        vec = np.zeros(self.dim, dtype=np.float32)
        for feature in features:
            h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "little")
            vec[h % self.dim] += 1.0 if (h >> 63) == 0 else -1.0
        return vec

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalise(np.stack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class SentenceTransformerEmbedder:
    """Local sentence-transformers model, loaded on first use."""

    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str = "BAAI/bge-small-en"):
        self.model_name = model_name
        self.id = f"st:{model_name}"
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vectors = self._load().encode(list(texts), normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        prefix = self.QUERY_PREFIX if "bge" in self.model_name.lower() else ""
        return self.embed_documents([prefix + text])[0]


def default_embedder() -> Embedder:
    """`STUDYBUDDY_EMBEDDER=hashing` for offline use; otherwise a model name."""
    choice = os.environ.get("STUDYBUDDY_EMBEDDER", "BAAI/bge-small-en")
    if choice == "hashing":
        return HashingEmbedder()
    return SentenceTransformerEmbedder(choice)
