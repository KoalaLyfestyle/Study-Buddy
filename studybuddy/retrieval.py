"""Lexical and hybrid retrieval over the same chunks as dense search.

Three methods share one interface, `retrieve(library, query, k, course, week)`,
and return chunk IDs with scores, best first:

- `dense`: the library's embedding search (`Library.search`).
- `bm25`: SQLite FTS5 with its built-in BM25 ranking.
- `tfidf`: cosine similarity over sublinear TF-IDF vectors, as an extra baseline.
- `hybrid`: reciprocal rank fusion (RRF) of dense and BM25.

Course, week, and file-kind filters are applied before ranking in every method, so a
filtered search never returns a chunk from another course or week. Hybrid
takes dense candidates whether or not they share a word with the query.
"""

from __future__ import annotations

import math
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .library import Library

_WORD = re.compile(r"[A-Za-z0-9]+")
_STOP = frozenset(
    "a an and are as at be by does do for from how i in is it my of on or so that the this to "
    "what when where which why with you your".split()
)


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    score: float


def _terms(text: str) -> list[str]:
    return [t for t in (w.lower() for w in _WORD.findall(text)) if t not in _STOP]


class LexicalIndex:
    """FTS5 and TF-IDF indexes built from a library's chunks table.

    Build it after an import. It reads the chunk text once and keeps
    its own in-memory SQLite database, so it never changes the library.
    """

    def __init__(self, library: Library):
        self.library = library
        with library._lock:
            rows = library.db.execute(
                "SELECT c.chunk_id, COALESCE(c.index_text, c.text), f.course, f.week, f.kind FROM chunks c "
                "JOIN files f ON f.relpath = c.relpath WHERE c.embedder = ?",
                (library.embedder.id,),
            ).fetchall()
        self.chunk_ids = [r[0] for r in rows]
        self._meta = {r[0]: (r[2], r[3], r[4]) for r in rows}

        self.fts = sqlite3.connect(":memory:", check_same_thread=False)
        self.fts.execute(
            "CREATE VIRTUAL TABLE chunks_fts USING fts5("
            "chunk_id UNINDEXED, course UNINDEXED, week UNINDEXED, kind UNINDEXED, text, "
            "tokenize='porter unicode61')"
        )
        self.fts.executemany(
            "INSERT INTO chunks_fts VALUES (?,?,?,?,?)", [(r[0], r[2], r[3], r[4], r[1]) for r in rows]
        )

        # Sublinear TF-IDF with smoothed IDF, rows L2-normalised.
        docs = [Counter(_terms(r[1])) for r in rows]
        df = Counter(t for d in docs for t in d)
        self.vocab = {t: i for i, t in enumerate(sorted(df))}
        n = len(docs)
        self.idf = np.array([math.log((1 + n) / (1 + df[t])) + 1 for t in sorted(df)], dtype=np.float32)
        self.matrix = np.zeros((n, len(self.vocab)), dtype=np.float32)
        for i, d in enumerate(docs):
            for t, c in d.items():
                self.matrix[i, self.vocab[t]] = (1 + math.log(c)) * self.idf[self.vocab[t]]
        norms = np.linalg.norm(self.matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.matrix /= norms

    def size_bytes(self) -> dict[str, int]:
        """Rough storage cost of each index."""
        fts_pages = self.fts.execute("PRAGMA page_count").fetchone()[0]
        page_size = self.fts.execute("PRAGMA page_size").fetchone()[0]
        nonzero = int(np.count_nonzero(self.matrix))
        return {
            "bm25": fts_pages * page_size,
            # Sparse storage: one float32 value and one int32 column per non-zero entry.
            "tfidf": nonzero * 8,
        }

    def _allowed(self, chunk_id: str, course: str | None, week: int | None, kind: str | None) -> bool:
        c, w, t = self._meta[chunk_id]
        return (course is None or c == course) and (week is None or w == week) and (kind is None or t == kind)

    def bm25(
        self, query: str, k: int, course: str | None = None, week: int | None = None, kind: str | None = None
    ) -> list[Hit]:
        terms = sorted(set(_terms(query)))
        if not terms:
            return []
        # Quote each term so FTS5 syntax in a query (AND, *, :) is never interpreted.
        match = " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)
        sql = "SELECT chunk_id, bm25(chunks_fts) FROM chunks_fts WHERE chunks_fts MATCH ?"
        params: list = [match]
        if course is not None:
            sql += " AND course = ?"
            params.append(course)
        if week is not None:
            sql += " AND week = ?"
            params.append(week)
        if kind is not None:
            sql += " AND kind = ?"
            params.append(kind)
        sql += " ORDER BY bm25(chunks_fts), chunk_id LIMIT ?"
        params.append(k)
        # FTS5's bm25() is lower-is-better; negate it so higher is better everywhere.
        return [Hit(cid, -score) for cid, score in self.fts.execute(sql, params)]

    def tfidf(
        self, query: str, k: int, course: str | None = None, week: int | None = None, kind: str | None = None
    ) -> list[Hit]:
        q = np.zeros(len(self.vocab), dtype=np.float32)
        for t, c in Counter(_terms(query)).items():
            if t in self.vocab:
                q[self.vocab[t]] = (1 + math.log(c)) * self.idf[self.vocab[t]]
        if not q.any():
            return []
        q /= np.linalg.norm(q)
        scores = self.matrix @ q
        hits = []
        for i in np.argsort(-scores, kind="stable"):
            if scores[i] <= 0:
                break
            cid = self.chunk_ids[i]
            if self._allowed(cid, course, week, kind):
                hits.append(Hit(cid, float(scores[i])))
                if len(hits) == k:
                    break
        return hits


def dense(
    library: Library, query: str, k: int, course: str | None = None, week: int | None = None, kind: str | None = None
) -> list[Hit]:
    return [Hit(r.chunk_id, r.score) for r in library.search(query, k=k, course=course, week=week, kind=kind)]


def rrf(rankings: list[list[Hit]], k: int, c: int = 60) -> list[Hit]:
    """Reciprocal rank fusion: score = sum of 1 / (c + rank) over the rankings."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (c + rank)
    best = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
    return [Hit(cid, s) for cid, s in best]


def hybrid(
    library: Library,
    lexical: LexicalIndex,
    query: str,
    k: int,
    course: str | None = None,
    week: int | None = None,
    kind: str | None = None,
    depth: int = 20,
    c: int = 60,
) -> list[Hit]:
    """RRF of the top `depth` dense and BM25 hits."""
    return rrf(
        [dense(library, query, depth, course, week, kind), lexical.bm25(query, depth, course, week, kind)],
        k=k,
        c=c,
    )
