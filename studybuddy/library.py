"""A course library: incremental import into SQLite, and search with citations.

The importer only reads the source folder. It never writes there, so
human-edited files such as `notes.md` are never replaced.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field, replace
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
    embedding BLOB NOT NULL,
    index_text TEXT                -- text plus index-only context; NULL in old indexes means text
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
    skipped_links: list[str] = field(default_factory=list)  # links that lead outside the course folder
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
        for relpath in self.skipped_links:
            text += f"\n  skipped: {relpath}: a link to a file outside the course folder"
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
    char_end: int  # char_start and char_end index the parsed section text, not the raw file
    source_sha256: str
    stale: bool = False  # the file changed and the new version failed to import

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
        label = f"{self.relpath}, {where}"
        return label + " (out of date: the last import could not read this file, so this is its last good version)" if self.stale else label


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
        # One connection shared across threads (Streamlit reruns use new
        # threads); the lock serialises every use of it.
        self._lock = threading.RLock()
        self.db = sqlite3.connect(
            str(self.index_dir / "library.sqlite"), isolation_level=None, check_same_thread=False
        )
        self.db.executescript(SCHEMA)
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(chunks)")}
        if "index_text" not in columns:  # an index made before parser version 2
            self.db.execute("ALTER TABLE chunks ADD COLUMN index_text TEXT")
        if source_dir is None:
            row = self.db.execute("SELECT value FROM meta WHERE key='root'").fetchone()
            source_dir = row[0] if row else "data"
        self.source_dir = Path(source_dir).resolve()

    def close(self) -> None:
        with self._lock:
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
        found = sources.scan(root)
        if "." in found.unreadable:
            raise EmptyLibraryError(
                f"Cannot read course folder {root} ({found.unreadable['.']}). "
                "Check its permissions, then import again. The index was left as it was."
            )
        if not found.paths:
            # Leave the index alone: an empty folder is more often an
            # unmounted drive or a wrong path than a deliberate wipe.
            raise EmptyLibraryError(empty_folder_message(root))
        with self._lock:
            report = self._import_paths(root, found.paths, found.unreadable)
        report.skipped_links = found.outside
        return report

    def _import_paths(self, root: Path, paths: list[Path], unreadable: dict[str, str] | None = None) -> ImportReport:
        report = ImportReport()
        db = self.db
        db.execute("BEGIN IMMEDIATE")
        try:
            stored_root = db.execute("SELECT value FROM meta WHERE key='root'").fetchone()
            if stored_root and stored_root[0] != str(root):
                # A different folder: start from an empty index.
                db.execute("DELETE FROM chunks")
                db.execute("DELETE FROM files")
            if not stored_root or stored_root[0] != str(root):
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

            for folder, error in sorted((unreadable or {}).items()):
                report.failed[f"{folder}/"] = f"Folder could not be read: {error}"
            for relpath in sorted(set(existing) - seen):
                folder = _unreadable_parent(relpath, unreadable or {})
                if folder is not None:
                    # Not deleted: we could not look. Keep its last good records, marked out of date.
                    error = f"Folder {folder} could not be read: {unreadable[folder]}"
                    db.execute("UPDATE files SET status='error', error=? WHERE relpath=?", (error, relpath))
                    report.failed[relpath] = error
                    continue
                db.execute("DELETE FROM chunks WHERE relpath=?", (relpath,))
                db.execute("DELETE FROM files WHERE relpath=?", (relpath,))
                report.removed.append(relpath)
            db.execute("COMMIT")
        except BaseException:
            db.execute("ROLLBACK")
            raise
        return report

    def _import_file(self, path: Path, relpath: str, previous, report: ImportReport) -> None:
        doc_id = sources.doc_id_for(relpath)
        kind = sources.file_kind(path)
        course, week = sources.course_and_week(relpath, self.source_dir.name)
        sha = ""
        try:
            sha = sources.sha256_file(path)  # inside the try: the file may vanish or be unreadable
            cache_key = self._cache_key(sha)
            if previous and previous[0] == cache_key and previous[1] == "ok":
                report.unchanged.append(relpath)
                return
            report.files_parsed += 1
            sections = sources.parse_file(path)
            chunks = _unique_ids(
                [c for s in sections for c in chunk_section(doc_id, s.text, s.location, s.context)]
            )
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

        was_indexed = previous is not None and (previous[1] == "ok" or self._has_chunks(relpath))
        vectors = self.embedder.embed_documents([c.index_text for c in chunks])
        report.chunks_embedded += len(chunks)
        self.db.execute("DELETE FROM chunks WHERE relpath=?", (relpath,))
        self.db.execute(
            "INSERT OR REPLACE INTO files VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (relpath, doc_id, kind, course, week, sha, cache_key, "ok", None, len(chunks), time.time()),
        )
        self.db.executemany(
            "INSERT INTO chunks (chunk_id, relpath, ordinal, text, location, char_start, char_end, "
            "embedder, embedding, index_text) VALUES (?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    c.chunk_id, relpath, c.ordinal, c.text, json.dumps(c.location),
                    c.char_start, c.char_end, self.embedder.id,
                    np.asarray(v, dtype=np.float32).tobytes(),
                    c.index_text,
                )
                for c, v in zip(chunks, vectors)
            ],
        )
        (report.updated if was_indexed else report.added).append(relpath)

    def _has_chunks(self, relpath: str) -> bool:
        return self.db.execute("SELECT 1 FROM chunks WHERE relpath=? LIMIT 1", (relpath,)).fetchone() is not None

    # ---- read ---------------------------------------------------------

    def failures(self) -> dict[str, str]:
        with self._lock:
            return self._failures()

    def _failures(self) -> dict[str, str]:
        return dict(self.db.execute("SELECT relpath, error FROM files WHERE status='error'"))

    def courses(self) -> list[dict]:
        with self._lock:
            rows = self.db.execute(
                "SELECT course, COUNT(*), GROUP_CONCAT(DISTINCT week) FROM files "
                "GROUP BY course ORDER BY course"
            ).fetchall()
        return [
            {
                "course": course,
                "files": n,
                "weeks": sorted(int(w) for w in (weeks or "").split(",") if w),
            }
            for course, n, weeks in rows
        ]

    def chunk_count(self) -> int:
        with self._lock:
            return self.db.execute(
                "SELECT COUNT(*) FROM chunks WHERE embedder=?", (self.embedder.id,)
            ).fetchone()[0]

    def _other_embedders(self) -> list[str]:
        rows = self.db.execute(
            "SELECT DISTINCT embedder FROM chunks WHERE embedder != ?", (self.embedder.id,)
        )
        return [r[0] for r in rows]

    def search(
        self,
        query: str,
        k: int = 4,
        course: str | None = None,
        week: int | None = None,
        kind: str | None = None,
    ) -> list[SearchResult]:
        """Dense search. Filters (course, week, and file kind: pdf, markdown, notebook) apply before ranking."""
        with self._lock:
            return self._search(query, k, course, week, kind)

    def _search(self, query, k, course, week, kind=None) -> list[SearchResult]:
        if self.chunk_count() == 0:
            others = self._other_embedders()
            if others:
                raise EmptyLibraryError(
                    f"This index was built with embedder {', '.join(others)}, not "
                    f"{self.embedder.id}. Import {self.source_dir} again with the current "
                    f"embedder, or set STUDYBUDDY_EMBEDDER to match the index."
                )
            raise EmptyLibraryError(empty_folder_message(self.source_dir))
        sql = (
            "SELECT c.chunk_id, c.text, c.location, c.char_start, c.char_end, c.embedding, "
            "f.relpath, f.doc_id, f.kind, f.course, f.week, f.sha256, f.status "
            "FROM chunks c JOIN files f ON f.relpath = c.relpath WHERE c.embedder = ?"
        )
        params: list = [self.embedder.id]
        if course is not None:
            sql += " AND f.course = ?"
            params.append(course)
        if week is not None:
            sql += " AND f.week = ?"
            params.append(week)
        if kind is not None:
            sql += " AND f.kind = ?"
            params.append(kind)
        rows = self.db.execute(sql, params).fetchall()
        if not rows:
            return []
        matrix = np.stack([np.frombuffer(r[5], dtype=np.float32) for r in rows])
        scores = matrix @ np.asarray(self.embedder.embed_query(query), dtype=np.float32)
        order = np.argsort(-scores, kind="stable")[:k]
        return [self._result(rows[i], float(scores[i])) for i in order]

    def outline(self, course: str, week: int | None) -> list[SearchResult]:
        """Every chunk of one course week, in file and reading order. Score is 0."""
        sql = (
            "SELECT c.chunk_id, c.text, c.location, c.char_start, c.char_end, c.embedding, "
            "f.relpath, f.doc_id, f.kind, f.course, f.week, f.sha256, f.status "
            "FROM chunks c JOIN files f ON f.relpath = c.relpath WHERE f.course = ? AND c.embedder = ?"
        )
        params: list = [course, self.embedder.id]
        if week is None:
            sql += " AND f.week IS NULL"
        else:
            sql += " AND f.week = ?"
            params.append(week)
        sql += " ORDER BY f.relpath, c.rowid"
        with self._lock:
            rows = self.db.execute(sql, params).fetchall()
        return [self._result(r, 0.0) for r in rows]

    def read_chunk(self, chunk_id: str) -> SearchResult | None:
        with self._lock:
            row = self.db.execute(
                "SELECT c.chunk_id, c.text, c.location, c.char_start, c.char_end, c.embedding, "
                "f.relpath, f.doc_id, f.kind, f.course, f.week, f.sha256, f.status "
                "FROM chunks c JOIN files f ON f.relpath = c.relpath WHERE c.chunk_id = ?",
                (chunk_id,),
            ).fetchone()
        return self._result(row, 0.0) if row else None

    @staticmethod
    def _result(row, score: float) -> SearchResult:
        chunk_id, text, location, start, end, _, relpath, doc_id, kind, course, week, sha, status = row
        return SearchResult(
            chunk_id=chunk_id,
            score=score,
            text=text,
            citation=Citation(
                relpath=relpath, doc_id=doc_id, kind=kind, course=course, week=week,
                location=json.loads(location), char_start=start, char_end=end,
                source_sha256=sha,
                stale=status != "ok",
            ),
        )

    def check_citation(self, result: SearchResult) -> bool:
        """True if the citation still points at this text in the source file."""
        path = self.source_dir / result.citation.relpath
        if (
            not sources.inside(self.source_dir, path)
            or not path.is_file()
            or sources.sha256_file(path) != result.citation.source_sha256
        ):
            return False
        for section in sources.parse_file(path):
            if section.location == result.citation.location:
                span = section.text[result.citation.char_start : result.citation.char_end]
                return span.strip() == result.text
        return False


def _unreadable_parent(relpath: str, unreadable: dict[str, str]) -> str | None:
    """The unreadable folder that holds relpath, if any."""
    for folder in unreadable:
        if folder != "." and relpath.startswith(folder.rstrip("/") + "/"):
            return folder
    return None


def _unique_ids(chunks):
    """Two identical sections in one file would share an ID; number repeats."""
    seen: dict[str, int] = {}
    out = []
    for chunk in chunks:
        count = seen.get(chunk.chunk_id, 0)
        seen[chunk.chunk_id] = count + 1
        if count:
            new_id = hashlib.sha256(f"{chunk.chunk_id}#{count}".encode()).hexdigest()[:20]
            chunk = replace(chunk, chunk_id=new_id)
        out.append(chunk)
    return out
