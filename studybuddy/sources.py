"""Read course files into sections with source locations.

Parsing never executes anything. Notebooks are read as JSON and their
outputs are ignored.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

PARSER_VERSION = "2"  # 2: notebook code cells carry index-only context

SUPPORTED_SUFFIXES = {
    ".pdf": "pdf",
    ".md": "markdown",
    ".markdown": "markdown",
    ".ipynb": "notebook",
}

_HEADING = re.compile(r"^ {0,3}(#{1,6})\s+(.+?)(?:\s+#+)?\s*$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_WEEK = re.compile(r"^(?:week|wk|w)[-_ ]?(\d{1,2})\b", re.IGNORECASE)


@dataclass(frozen=True)
class Section:
    """A piece of one file with a location a reader can open.

    location keys by kind:
    - pdf: page (1-based)
    - markdown: headings (list, outermost first), line (1-based start line)
    - notebook: cell_id, cell_index (0-based), cell_type
    """

    text: str
    location: dict = field(default_factory=dict)
    # Extra words used only to find this section by keyword search, never
    # shown or cited. Notebook code cells get the notebook title and the
    # Markdown cell above them, since a code cell rarely says what it does.
    context: str = ""


def file_kind(path: Path) -> str | None:
    return SUPPORTED_SUFFIXES.get(path.suffix.lower())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def doc_id_for(relpath: str) -> str:
    """Stable document ID: derived from the path relative to the library root."""
    return hashlib.sha256(relpath.encode("utf-8")).hexdigest()[:12]


def course_and_week(relpath: str, root_name: str) -> tuple[str, int | None]:
    """Course is the top-level folder; week comes from a `week-NN` style folder."""
    parts = Path(relpath).parts
    course = parts[0] if len(parts) > 1 else root_name
    week = None
    for part in parts[:-1]:
        match = _WEEK.match(part)
        if match:
            week = int(match.group(1))
    return course, week


@dataclass
class Scan:
    """What scan_folder found. Paths in `unreadable` and `outside` are relative to root."""

    paths: list[Path] = field(default_factory=list)
    unreadable: dict[str, str] = field(default_factory=dict)  # folder -> error; "." is the root
    outside: list[str] = field(default_factory=list)  # links that lead out of the folder, skipped


def scan(root: Path) -> Scan:
    """Supported files under root, skipping hidden files and folders.

    Folder links are not followed. A file link is kept only if it resolves
    inside root, so a link cannot pull outside files into the library (and
    from there into search results and the MCP server). Folders that cannot
    be read are reported, not silently skipped.
    """
    result = Scan()
    real_root = root.resolve()

    def failed(err: OSError) -> None:
        where = Path(err.filename) if err.filename else root
        try:
            rel = where.relative_to(root).as_posix()
        except ValueError:
            rel = "."
        result.unreadable[rel or "."] = f"{type(err).__name__}: {err.strerror or err}"

    for dirpath, dirnames, filenames in os.walk(root, onerror=failed, followlinks=False):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in filenames:
            if name.startswith("."):
                continue
            path = Path(dirpath) / name
            if not file_kind(path):
                continue
            if path.is_symlink():
                try:
                    target = path.resolve(strict=True)
                except (OSError, RuntimeError):
                    continue  # a broken or looping link
                if not target.is_relative_to(real_root) or any(
                    part.startswith(".") for part in target.relative_to(real_root).parts
                ):
                    # Outside the folder, or into a hidden part of it that the scan skips.
                    result.outside.append(path.relative_to(root).as_posix())
                    continue
            if path.is_file():
                result.paths.append(path)
    result.paths.sort()
    result.outside.sort()
    return result


def scan_folder(root: Path) -> list[Path]:
    """Supported files under root (see scan)."""
    return scan(root).paths


def inside(root: Path, path: Path) -> bool:
    """True if path, after following links, is inside root."""
    try:
        return path.resolve().is_relative_to(root.resolve())
    except (OSError, RuntimeError):
        return False


def parse_file(path: Path) -> list[Section]:
    kind = file_kind(path)
    if kind == "pdf":
        return parse_pdf(path)
    if kind == "markdown":
        return parse_markdown(path.read_text(encoding="utf-8"))
    if kind == "notebook":
        return parse_notebook(path.read_text(encoding="utf-8"))
    raise ValueError(f"Unsupported file type: {path.suffix}")


def parse_pdf(path: Path) -> list[Section]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    sections = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            sections.append(Section(text=text, location={"page": number}))
    return sections


def parse_markdown(text: str) -> list[Section]:
    sections: list[Section] = []
    headings: list[tuple[int, str]] = []
    buffer: list[str] = []
    start_line = 1
    fence = ""  # the opening fence while inside a code block

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append(
                Section(
                    text=body,
                    location={"headings": [h for _, h in headings], "line": start_line},
                )
            )

    for number, line in enumerate(text.splitlines(), start=1):
        opener = _FENCE.match(line)
        if opener:
            marker = opener.group(1)
            if not fence:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence) and not line.strip()[len(marker):].strip():
                fence = ""
            # Lines inside a fence are body text.
            match = None
        else:
            match = None if fence else _HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            headings = [h for h in headings if h[0] < level] + [(level, match.group(2))]
            buffer = [line]
            start_line = number
        else:
            if not buffer:
                start_line = number
            buffer.append(line)
    flush()
    return sections


CONTEXT_CHARS = 400


def _notebook_title(cells: list[dict]) -> str:
    for cell in cells:
        if cell.get("cell_type") != "markdown":
            continue
        source = cell.get("source", "")
        for line in ("".join(source) if isinstance(source, list) else source).splitlines():
            match = _HEADING.match(line)
            if match:
                return match.group(2).strip()
    return ""


def parse_notebook(text: str) -> list[Section]:
    data = json.loads(text)
    sections = []
    cells = data.get("cells", [])
    title = _notebook_title(cells)
    above = ""  # the nearest non-empty Markdown cell so far
    for index, cell in enumerate(cells):
        cell_type = cell.get("cell_type")
        if cell_type not in ("markdown", "code"):
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        source = source.strip()
        if not source:
            continue
        context = ""
        if cell_type == "code":
            parts = [f"Code cell in {title}." if title else "Code cell."]
            lines = above.splitlines()
            heading = _HEADING.match(lines[0]) if lines else None
            if heading and heading.group(2).strip() == title:
                lines = lines[1:]  # the title is already there
            words = " ".join(" ".join(lines).split())[:CONTEXT_CHARS]
            if words:
                parts.append(words)
            context = " ".join(parts)
        else:
            above = source
        sections.append(
            Section(
                text=source,
                location={
                    "cell_id": cell.get("id") or f"cell-{index}",
                    "cell_index": index,
                    "cell_type": cell_type,
                },
                context=context,
            )
        )
    return sections
