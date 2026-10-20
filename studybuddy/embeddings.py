"""Embedding backends.

`SentenceTransformerEmbedder` is the real one (BAAI/bge-small-en, run
locally). `HashingEmbedder` needs no model download. It is deterministic
and lexical only, so tests and offline checks use it.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
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


DEFAULT_MODEL = "BAAI/bge-small-en"
# The Hugging Face commit of DEFAULT_MODEL to use, checked on Winston's Mac on
# 19 October 2026 with `python -m studybuddy.embeddings`. Set it to None to use
# whatever commit is in the local cache (or the Hub's main) instead.
DEFAULT_REVISION: str | None = "2275a7bdee235e9b4f01fa73aa60d3311983cfea"

_COMMIT = re.compile(r"[0-9a-f]{40}")


def _hub_cache() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return home / "hub"


def cached_commit(model_name: str, revision: str = "main") -> str | None:
    """The commit SHA that `revision` points to in the local Hugging Face cache, if any.

    Reads the cache's refs file only, so it needs no network and no extra package.
    """
    if _COMMIT.fullmatch(revision):
        return revision
    ref = _hub_cache() / f"models--{model_name.replace('/', '--')}" / "refs" / revision
    try:
        commit = ref.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return commit if _COMMIT.fullmatch(commit) else None


def remote_commit(model_name: str, revision: str = "main") -> str | None:
    """Ask the Hugging Face Hub which commit `revision` is. None if offline or unavailable.

    Used only when the model is neither pinned nor cached, so that the first
    import already records the commit it will download.
    """
    try:
        from huggingface_hub import HfApi

        sha = HfApi().model_info(model_name, revision=revision).sha
    except Exception:  # no package, no network, unknown model: fall back to "unpinned"
        return None
    return sha if sha and _COMMIT.fullmatch(sha) else None


class SentenceTransformerEmbedder:
    """Local sentence-transformers model, loaded on first use.

    The model's commit is part of `id`, and so of every file's cache key: a
    different revision of the weights re-embeds the library instead of mixing
    old and new vectors. Pass a commit SHA as `revision` to pin it. Without
    one, the commit that `main` points to (in the local cache, or else on the
    Hub) is used and loaded, so the ID matches the weights. "unpinned" is
    recorded only when neither can be found, and then the model cannot be
    downloaded either.
    """

    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str = DEFAULT_MODEL, revision: str | None = None):
        if revision is None and model_name == DEFAULT_MODEL:
            revision = DEFAULT_REVISION
        self.model_name = model_name
        wanted = revision or "main"
        self.revision = cached_commit(model_name, wanted) or remote_commit(model_name, wanted) or revision
        self.id = f"st:{model_name}@{self.revision or 'unpinned'}"
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, revision=self.revision)
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
    """`STUDYBUDDY_EMBEDDER=hashing` for offline use; otherwise a model name,
    optionally with a revision: `BAAI/bge-small-en@<commit sha>`."""
    choice = os.environ.get("STUDYBUDDY_EMBEDDER", DEFAULT_MODEL)
    if choice == "hashing":
        return HashingEmbedder()
    name, _, revision = choice.partition("@")
    return SentenceTransformerEmbedder(name, revision or None)


if __name__ == "__main__":
    # Print the commit of the default model in the local cache, to pin DEFAULT_REVISION.
    commit = cached_commit(DEFAULT_MODEL)
    if commit is None:
        raise SystemExit(f"{DEFAULT_MODEL} is not in the local Hugging Face cache ({_hub_cache()}). Import once first.")
    print(commit)
