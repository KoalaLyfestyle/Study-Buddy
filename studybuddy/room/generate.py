"""Add model-written sections to an extractive study room.

The model (local Ollama by default) gets one chapter at a time: its title
and its passages with their chunk IDs. It answers with JSON sections. Each
reply is merged into the room and checked with the same rules as any room
(schema, known citations, quiz answers). A bad reply is retried once with
the list of problems; if it is still bad, that chapter keeps only its
extractive sections and the report says why. Nothing is written here;
save_room writes the result.

Generated sections are marked `provenance: "generated"`, may cite only
their own chapter's passages, and get IDs from the chapter, kind, and
order, so rebuilding with the same reply keeps progress and edits.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Callable

from .model import RoomValidationError, validate_room

TextModel = Callable[[str], str]

GENERATED_KINDS = ("setup", "worked_example", "common_mistakes", "quiz")
MAX_PASSAGE_CHARS = 1500
MAX_PROMPT_PASSAGE_CHARS = 9000
MAX_SECTIONS = 6

PROMPT = """You are writing study notes for one part of a course. Use only the passages below.

Chapter: {title}

Passages (each starts with its ID in square brackets):
{passages}

Reply with JSON only, no other text, in this shape:
{{"sections": [
  {{"kind": "setup" | "worked_example" | "common_mistakes" | "quiz",
    "title": "short title",
    "cites": ["passage IDs this section is based on"],
    "blocks": [{{"type": "paragraph", "text": "..."}} | {{"type": "code", "text": "..."}} | {{"type": "list", "items": ["..."]}}],
    "quiz": [{{"prompt": "...", "choices": ["...", "..."], "answer_index": 0, "explanation": "...", "cites": ["passage IDs"]}}]
  }}
]}}

Rules:
- At most {max_sections} sections. Cite only IDs from the passages above.
- "setup" explains the chapter in plain words for a beginner.
- "worked_example" works one small example step by step.
- "common_mistakes" lists mistakes a student is likely to make.
- "quiz" has 2 to 5 questions with 2 to 4 distinct choices; "blocks" may be empty. Only quiz sections have "quiz".
- Do not copy code to run; code blocks are shown as text only.
"""

RETRY = """
Your last reply could not be used:
{problems}
Reply again with corrected JSON only.
"""


@dataclass
class GenerateReport:
    model: str
    generated: list[str] = field(default_factory=list)  # chapter titles that got model sections
    fallbacks: dict[str, str] = field(default_factory=dict)  # chapter title -> why it kept only extracts

    def summary(self) -> str:
        text = f"{self.model}: {len(self.generated)} chapters got generated sections"
        for title, why in sorted(self.fallbacks.items()):
            text += f"\n  kept only the extracts for {title!r}: {why}"
        return text


def _hash(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).hexdigest()[:12]


def _passages(chapter: dict, texts: dict[str, str]) -> tuple[str, list[str]]:
    ids: list[str] = []
    for section in chapter["sections"]:
        for chunk_id in section["cites"]:
            if chunk_id not in ids and chunk_id in texts:
                ids.append(chunk_id)
    lines, used = [], 0
    for chunk_id in ids:
        text = texts[chunk_id][:MAX_PASSAGE_CHARS]
        if used + len(text) > MAX_PROMPT_PASSAGE_CHARS:
            break
        used += len(text)
        lines.append(f"[{chunk_id}] {text}")
    return "\n\n".join(lines), ids


def _json_object(reply: str) -> object:
    """The JSON in a model reply, allowing for text or code fences around it."""
    try:
        return json.loads(reply)
    except json.JSONDecodeError as first:
        start, end = reply.find("{"), reply.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(reply[start : end + 1])
            except json.JSONDecodeError:
                pass
        raise RoomValidationError([f"/: not valid JSON: {first.msg} at line {first.lineno} column {first.colno}"]) from None


def _sections_from_reply(reply: str, chapter: dict, allowed: list[str]) -> list[dict]:
    data = _json_object(reply)
    if not isinstance(data, dict) or not isinstance(data.get("sections"), list):
        raise RoomValidationError(['/sections: expected {"sections": [...]}'])
    problems: list[str] = []
    out: list[dict] = []
    counts: dict[str, int] = {}
    allowed_set = set(allowed)
    for i, raw in enumerate(data["sections"][:MAX_SECTIONS]):
        where = f"/sections/{i}"
        if not isinstance(raw, dict):
            problems.append(f"{where}: must be an object")
            continue
        kind = raw.get("kind")
        if kind not in GENERATED_KINDS:
            problems.append(f"{where}/kind: must be one of {', '.join(GENERATED_KINDS)}")
            continue
        n = counts.get(kind, 0)
        counts[kind] = n + 1
        sid = f"gen-{_hash(chapter['id'], kind, n)}"
        cites = raw.get("cites")
        if isinstance(cites, list):
            for j, c in enumerate(cites):
                if not isinstance(c, str) or c not in allowed_set:
                    problems.append(f"{where}/cites/{j}: {c!r} is not one of this chapter's passages")
        section = {
            "id": sid,
            "kind": kind,
            "title": raw.get("title"),
            "provenance": "generated",
            "cites": cites,
            "blocks": raw.get("blocks", []),
        }
        if kind == "quiz":
            questions = raw.get("quiz")
            if isinstance(questions, list):
                for qi, q in enumerate(questions):
                    if isinstance(q, dict):
                        q = {k: q.get(k) for k in ("prompt", "choices", "answer_index", "explanation", "cites")}
                        q["id"] = f"gq-{_hash(sid, qi)}"
                        for j, c in enumerate(q["cites"] if isinstance(q["cites"], list) else []):
                            if c not in allowed_set:
                                problems.append(f"{where}/quiz/{qi}/cites/{j}: {c!r} is not one of this chapter's passages")
                        questions[qi] = q
            section["quiz"] = questions
        elif "quiz" in raw:
            problems.append(f"{where}/quiz: only quiz sections have questions")
        out.append(section)
    if not out and not problems:
        problems.append("/sections: no sections")
    if problems:
        raise RoomValidationError(problems)
    return out


def _with_chapter(room: dict, index: int, sections: list[dict]) -> dict:
    candidate = copy.deepcopy(room)
    chapter = candidate["chapters"][index]
    chapter["sections"] = [s for s in chapter["sections"] if s["provenance"] != "generated"] + sections
    for section in sections:
        for chunk_id in section["cites"] if isinstance(section["cites"], list) else []:
            if not isinstance(chunk_id, str):
                continue
            ids = candidate["coverage"].setdefault(chunk_id, [])
            if section["id"] not in ids:
                ids.append(section["id"])
    return candidate


def generate_room(room: dict, texts: dict[str, str], model: TextModel, model_name: str) -> tuple[dict, GenerateReport]:
    """Return a copy of `room` with model sections added where the model's reply passes the checks.

    texts: chunk ID -> passage text for the room's week (library.outline).
    """
    report = GenerateReport(model=model_name)
    known = set(room["coverage"])
    result = copy.deepcopy(room)
    for index, chapter in enumerate(room["chapters"]):
        passages, allowed = _passages(chapter, texts)
        if not allowed:
            report.fallbacks[chapter["title"]] = "no passages to send"
            continue
        prompt = PROMPT.format(title=chapter["title"], passages=passages, max_sections=MAX_SECTIONS)
        error: RoomValidationError | None = None
        for attempt in range(2):
            ask = prompt if error is None else prompt + RETRY.format(problems="\n".join(f"- {p}" for p in error.problems[:10]))
            try:
                reply = model(ask)
                sections = _sections_from_reply(reply, chapter, allowed)
                candidate = _with_chapter(result, index, sections)
                validate_room(candidate, known)
            except RoomValidationError as exc:
                error = exc
                continue
            except Exception as exc:  # the model call itself failed (not running, timeout)
                error = RoomValidationError([f"the model call failed: {type(exc).__name__}: {exc}"])
                break
            result = candidate
            report.generated.append(chapter["title"])
            error = None
            break
        if error is not None:
            report.fallbacks[chapter["title"]] = "; ".join(error.problems[:3])
    if report.generated:
        result["builder"] = f"{room['builder']}+{model_name}"
    return validate_room(result, known), report


def ollama_model(name: str = "mistral") -> TextModel:
    """The local Ollama model, asked for JSON. Needs `ollama serve` and `ollama pull <name>`."""
    from langchain_ollama import OllamaLLM

    llm = OllamaLLM(model=name, temperature=0, format="json", num_predict=2000, timeout=300)
    return lambda prompt: llm.invoke(prompt)
