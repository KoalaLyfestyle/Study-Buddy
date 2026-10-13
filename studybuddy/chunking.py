"""Split sections into chunks with stable IDs and character offsets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

CHUNKER_VERSION = "1"
MAX_CHARS = 1200
OVERLAP = 150


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    ordinal: int  # position within its section
    text: str
    location: dict
    char_start: int  # offsets into the section text
    char_end: int


def split_text(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP) -> list[tuple[int, int]]:
    """Return (start, end) spans that cover text, preferring paragraph breaks."""
    if len(text) <= max_chars:
        return [(0, len(text))]
    spans = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            cut = text.rfind("\n\n", start + max_chars // 2, end)
            if cut == -1:
                cut = text.rfind(" ", start + max_chars // 2, end)
            if cut != -1:
                end = cut
        spans.append((start, end))
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
        # Do not start a chunk in the middle of a word.
        space = text.find(" ", start, end)
        if space != -1:
            start = space + 1
    return spans


def make_chunk_id(doc_id: str, location: dict, ordinal: int, text: str) -> str:
    """Same file path, location, and text always give the same ID."""
    key = json.dumps([doc_id, location, ordinal, text], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


def chunk_section(doc_id: str, text: str, location: dict) -> list[Chunk]:
    chunks = []
    for ordinal, (start, end) in enumerate(split_text(text)):
        piece = text[start:end].strip()
        if not piece:
            continue
        chunks.append(
            Chunk(
                chunk_id=make_chunk_id(doc_id, location, ordinal, piece),
                ordinal=ordinal,
                text=piece,
                location=location,
                char_start=start,
                char_end=end,
            )
        )
    return chunks
