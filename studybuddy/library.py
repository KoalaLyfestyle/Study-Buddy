"""A course library: incremental import into SQLite, and search with citations.

The importer only reads the source folder. It never writes there, so
human-edited files such as `notes.md` are never replaced.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import sources
from .chunking import CHUNKER_VERSION, chunk_section
from .embeddings import Embedder

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS files (
    relpath TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    course TEXT NOT NULL,
    week INTEGER,
    sha256 TEXT NOT NULL,
    cache_key TEXT NOT NULL,
    status TEXT NOT NULL,          -- ok or error
    error TEXT,
    n_chunks INTEGER NOT NULL,
    indexed_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    relpath TEXT NOT NULL REFERENCES files(relpath),
    ordinal INTEGER NOT NULL,
    text TEXT NOT NULL,
    location TEXT NOT NULL,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    embedder TEXT NOT NULL,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_relpath ON chunks(relpath);
"""

SUPPORTED_LIST = ", ".join(sorted(set(sources.SUPPORTED_SUFFIXES)))


class EmptyLibraryError(Exception):
    """No course material to import or search. The message says what to do."""


def empty_folder_message(folder: Path) -> str:
    return (
        f"No course material found in {folder}. "
        f"Add PDF, Markdown, or notebook files ({SUPPORTED_LIST}), one folder per course, "
        f"for example {folder}/<course>/week-01/notes.md. "
        f"Then import them with: python -m studybuddy import {folder}"
    )


@dataclass
class ImportReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    files_parsed: int = 0
    chunks_embedded: int = 0

    def summary(self) -> str:
        text = (
            f"{len(self.added)} added, {len(self.updated)} updated, "
            f"{len(self.unchanged)} unchanged, {len(self.removed)} removed, "
            f"{len(self.failed)} failed; {self.chunks_embedded} chunks embedded"
        )
        for relpath, error in sorted(self.failed.items()):
            text += f"\n  failed: {relpath}: {error}"
        return text


@dataclass(frozen=True)
class Citation:
    relpath: str
    doc_id: str
    kind: str
    course: str
    week: int | None
    location: dict
    char_start: int
    char_end: int
    source_sha256: str

    def label(self) -> str:
        loc = self.location
        if "page" in loc:
            where = f"page {loc['page']}"
        elif "cell_id" in loc:
            where = f"cell {loc['cell_index'] + 1} ({loc['cell_id']})"
        elif loc.get("headings"):
            where = " > ".join(loc["headings"])
        else:
            where = f"line {loc.get('line', 1)}"
        return f"{self.relpath}, {where}"


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    score: float
    text: str
    citation: Citation


class Library:
    def __init__(self, source_dir: str | Path | None, index_dir: str | Path, embedder: Embedder):
        """source_dir None means: the folder this index was last imported from."""
        self.index_dir = Path(index_dir)
        self.embedder = embedder
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.index_dir / "library.sqlite"), isolation_level=None)
        self.db.executescript(SCHEMA)
        if source_dir is None:
            row = self.db.execute("SELECT value FROM meta WHERE key='root'").fetchone()
            source_dir = row[0] if row else "data"
        self.source_dir = Path(source_dir).resolve()

    def close(self) -> None:
        self.db.close()

    def _cache_key(self, sha: str) -> str:
        return (
            f"{sha}|parser={sources.PARSER_VERSION}|chunker={CHUNKER_VERSION}"
            f"|embedder={self.embedder.id}"
        )

    # ---- import -------------------------------------------------------

    def import_folder(self) -> ImportReport:
        """Bring the index up to date with the source folder.

        Unchanged files are skipped without parsing or embedding. A file
        that fails to parse keeps its last good records and is reported.
        If the run fails as a whole (for example the embedder fails), the
        transaction rolls back and the previous index stays intact.
        """
        root = self.source_dir
        if not root.is_dir():
            raise EmptyLibraryError(
                f"Course folder {root} does not exist. Create it and add course files. "
                + empty_folder_message(root)
            )
        report = ImportReport()
        paths = sources.scan_folder(root)
        db = self.db
        db.execute("BEGIN IMMEDIATE")
        try:
            stored_root = db.execute("SELECT value FROM meta WHERE key='root'").fetchone()
            if stored_root and stored_root[0] != str(root):
                # A different folder: start from an empty index.
                db.execute("DELETE FROM chunks")
                db.execute("DELETE FROM files")
            db.execute("INSERT OR REPLACE INTO meta VALUES ('root', ?)", (str(root),))

            existing = {
                row[0]: (row[1], row[2])
                for row in db.execute("SELECT relpath, cache_key, status FROM files")
            }
            seen = set()
            for path in paths:
                relpath = path.relative_to(root).as_posix()
                seen.add(relpath)
                self._import_file(path, relpath, existing.get(relpath), report)

            for relpath in sorted(set(existing) - seen):
                db.execute("DELETE FROM chunks WHERE relpath=?", (relpath,))
                db.execute("DELETE FROM files WHERE relpath=?", (relpath,))
                report.removed.append(relpath)
            db.execute("COMMIT")
        except BaseException:
            db.execute("ROLLBACK")
            raise
        if not paths:
            raise EmptyLibraryError(empty_folder_message(root))
        return report

    def _import_file(self, path: Path, relpath: str, previous, report: ImportReport) -> None:
        sha = sources.sha256_file(path)
        cache_key = self._cache_key(sha)
        if previous and previous[0] == cache_key and previous[1] == "ok":
            report.unchanged.append(relpath)
            return

        doc_id = sources.doc_id_for(relpath)
        kind = sources.file_kind(path)
        course, week = sources.course_and_week(relpath, self.source_dir.name)
        try:
            report.files_parsed += 1
            sections = sources.parse_file(path)
            chunks = [c for s in sections for c in chunk_section(doc_id, s.text, s.location)]
        except Exception as exc:  # a bad file must not stop the import
            error = f"{type(exc).__name__}: {exc}"
            report.failed[relpath] = error
            if previous:
                # Keep the last good chunks; record the failure.
                self.db.execute(
                    "UPDATE files SET status='error', error=? WHERE relpath=?", (error, relpath)
                )
            else:
                self.db.execute(
                    "INSERT INTO files VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (relpath, doc_id, kind, course, week, sha, "", "error", error, 0, time.time()),
                )
            return

        vectors = self.embedder.embed_documents([c.text for c in chunks])
        report.chunks_embedded += len(chunks)
        self.db.execute("DELETE FROM chunks WHERE relpath=?", (relpath,))
        self.db.execute(
            "INSERT OR REPLACE INTO files VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (relpath, doc_id, kind, course, week, sha, cache_key, "ok", None, len(chunks), time.time()),
        )
        self.db.executemany(
            "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?)",
            [
                (
                    c.chunk_id, relpath, c.ordinal, c.text, json.dumps(c.location),
                    c.char_start, c.char_end, self.embedder.id,
                    np.asarray(v, dtype=np.float32).tobytes(),
                )
                for c, v in zip(chunks, vectors)
            ],
        )
        (report.updated if previous else report.added).append(relpath)

    # ---- read ---------------------------------------------------------

    def failures(self) -> dict[str, str]:
        return dict(self.db.execute("SELECT relpath, error FROM files WHERE status='error'"))

    def courses(self) -> list[dict]:
        rows = self.db.execute(
            "SELECT course, COUNT(*), GROUP_CONCAT(DISTINCT week) FROM files GROUP BY course ORDER BY course"
        )
        return [
            {
                "course": course,
                "files": n,
                "weeks": sorted(int(w) for w in (weeks or "").split(",") if w),
            }
            for course, n, weeks in rows
        ]

    def chunk_count(self) -> int:
        return self.db.execute(
            "SELECT COUNT(*) FROM chunks WHERE embedder=?", (self.embedder.id,)
        ).fetchone()[0]

    def search(
        self, query: str, k: int = 4, course: str | None = None, week: int | None = None
    ) -> list[SearchResult]:
        """Dense search. Filters apply before ranking."""
        if self.chunk_count() == 0:
            raise EmptyLibraryError(empty_folder_message(self.source_dir))
        sql = (
            "SELECT c.chunk_id, c.text, c.location, c.char_start, c.char_end, c.embedding, "
            "f.relpath, f.doc_id, f.kind, f.course, f.week, f.sha256 "
            "FROM chunks c JOIN files f ON f.relpath = c.relpath WHERE c.embedder = ?"
        )
        params: list = [self.embedder.id]
        if course is not None:
            sql += " AND f.course = ?"
            params.append(course)
        if week is not None:
            sql += " AND f.week = ?"
            params.append(week)
        rows = self.db.execute(sql, params).fetchall()
        if not rows:
            return []
        matrix = np.stack([np.frombuffer(r[5], dtype=np.float32) for r in rows])
        scores = matrix @ np.asarray(self.embedder.embed_query(query), dtype=np.float32)
        order = np.argsort(-scores, kind="stable")[:k]
        return [self._result(rows[i], float(scores[i])) for i in order]

    def read_chunk(self, chunk_id: str) -> SearchResult | None:
        row = self.db.execute(
            "SELECT c.chunk_id, c.text, c.location, c.char_start, c.char_end, c.embedding, "
            "f.relpath, f.doc_id, f.kind, f.course, f.week, f.sha256 "
            "FROM chunks c JOIN files f ON f.relpath = c.relpath WHERE c.chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        return self._result(row, 0.0) if row else None

    @staticmethod
    def _result(row, score: float) -> SearchResult:
        chunk_id, text, location, start, end, _, relpath, doc_id, kind, course, week, sha = row
        return SearchResult(
            chunk_id=chunk_id,
            score=score,
            text=text,
            citation=Citation(
                relpath=relpath, doc_id=doc_id, kind=kind, course=course, week=week,
                location=json.loads(location), char_start=start, char_end=end,
                source_sha256=sha,
            ),
        )

    def check_citation(self, result: SearchResult) -> bool:
        """True if the citation still points at this text in the source file."""
        path = self.source_dir / result.citation.relpath
        if not path.is_file() or sources.sha256_file(path) != result.citation.source_sha256:
            return False
        for section in sources.parse_file(path):
            if section.location == result.citation.location:
                span = section.text[result.citation.char_start : result.citation.char_end]
                return span.strip() == result.text
        return False
