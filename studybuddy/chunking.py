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
    index_text: str = ""  # what keyword search sees: context plus text (embeddings use text)

    def __post_init__(self):
        if not self.index_text:
            object.__setattr__(self, "index_text", self.text)


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
    """Same file path, location, and text always give the same ID.

    The Markdown line number is left out, so editing one section does not
    change the IDs of the sections below it.
    """
    stable = {k: v for k, v in location.items() if k != "line"}
    key = json.dumps([doc_id, stable, ordinal, text], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


def chunk_section(doc_id: str, text: str, location: dict, context: str = "") -> list[Chunk]:
    """Chunks of one section. context is added to each chunk's index_text only,
    so chunk IDs, offsets, and cited text do not depend on it."""
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
                index_text=f"{context}\n\n{piece}" if context else piece,
            )
        )
    return chunks
