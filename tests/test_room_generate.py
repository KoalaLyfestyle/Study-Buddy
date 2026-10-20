"""Model-written room sections, with a fake model (no network, no Ollama)."""

import json
import re
import shutil
from pathlib import Path

import pytest

from studybuddy import HashingEmbedder, Library
from studybuddy import __main__ as cli_module
from studybuddy.room import build_room, generate_room, room_dir_for, save_room, week_chunk_ids, week_texts

SAMPLE = Path(__file__).resolve().parent.parent / "sample_course"
COURSE = "intro-to-optimisation"


@pytest.fixture
def library(tmp_path):
    folder = tmp_path / "course"
    shutil.copytree(SAMPLE, folder)
    lib = Library(folder, tmp_path / "index", HashingEmbedder())
    lib.import_folder()
    yield lib
    lib.close()


def ids_in(prompt):
    passages = prompt.split("Reply with JSON only")[0]
    return re.findall(r"^\[([0-9a-f]{20})\]", passages, flags=re.M)


def good_reply(prompt):
    ids = ids_in(prompt)
    return json.dumps(
        {
            "sections": [
                {"kind": "setup", "title": "The idea", "cites": ids[:1],
                 "blocks": [{"type": "paragraph", "text": "We take small steps downhill."}]},
                {"kind": "quiz", "title": "Quick check", "cites": ids[:1], "blocks": [],
                 "quiz": [{"prompt": "Which way does a step go?", "choices": ["Downhill", "Uphill"],
                           "answer_index": 0, "explanation": "Against the gradient.", "cites": ids[:1]}]},
            ]
        }
    )


class Recorder:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        return reply(prompt) if callable(reply) else reply


def sections(room):
    return [s for c in room["chapters"] for s in c["sections"]]


def run(library, model, week=1):
    room = build_room(library, COURSE, week)
    return room, *generate_room(room, week_texts(library, COURSE, week), model, "fake")


def test_generated_sections_validate_cite_their_chapter_and_are_labelled(library, tmp_path):
    room, generated, report = run(library, Recorder(good_reply))
    assert len(report.generated) == len(room["chapters"]) and not report.fallbacks
    assert generated["builder"] == "extractive-v1+fake"
    for chapter in generated["chapters"]:
        own = {c for s in chapter["sections"] if s["provenance"] == "source" for c in s["cites"]}
        made = [s for s in chapter["sections"] if s["provenance"] == "generated"]
        assert [s["kind"] for s in made] == ["setup", "quiz"]
        assert all(set(s["cites"]) <= own for s in made)
    html = save_room(library, generated, tmp_path / "room").html.read_text()
    assert html.count("badge generated") == 2 * len(room["chapters"])
    assert "We take small steps downhill." in html


def test_a_truncated_reply_is_retried_then_the_chapter_keeps_its_extracts(library):
    model = Recorder(lambda p: good_reply(p)[:40])
    room, generated, report = run(library, model)
    assert report.generated == [] and len(report.fallbacks) == len(room["chapters"])
    assert all("not valid JSON" in why for why in report.fallbacks.values())
    assert generated == room  # nothing half-applied
    assert len(model.prompts) == 2 * len(room["chapters"])
    assert "could not be used" in model.prompts[1] and "not valid JSON" in model.prompts[1]


def test_a_fixed_second_reply_is_used(library):
    model = Recorder('{"sections": [{"kind": "poem"}]}', good_reply)
    _, generated, report = run(library, model, week=2)
    assert report.generated and not report.fallbacks
    assert "/sections/0/kind" in model.prompts[1]


@pytest.mark.parametrize(
    "bad",
    [
        lambda p: good_reply(p).replace('"answer_index": 0', '"answer_index": 5'),
        lambda p: good_reply(p).replace(ids_in(p)[0], "0123456789abcdef0123"),
        lambda p: good_reply(p).replace('"choices": ["Downhill", "Uphill"]', '"choices": ["Same", "Same"]'),
        lambda p: json.dumps({"sections": [{"kind": "setup", "title": "x", "cites": ids_in(p)[:1], "blocks": [],
                                            "quiz": []}]}),
        lambda p: json.dumps({"sections": []}),
        lambda p: "[]",
        lambda p: good_reply(p).replace('"answer_index": 0', '"answer_index": 0.0'),
        lambda p: json.dumps({"sections": [{"kind": "quiz", "title": "q", "cites": ids_in(p)[:1], "blocks": [],
                                            "quiz": [{"prompt": "x", "choices": ["a", "b"], "answer_index": 0,
                                                      "explanation": "e", "cites": [["nested"]]}]}]}),
        lambda p: json.dumps({"sections": ["just text"]}),
        lambda p: json.dumps({"sections": [{"kind": "quiz", "title": "q", "cites": ids_in(p)[:1], "blocks": [],
                                            "quiz": ["not a question"]}]}),
    ],
)
def test_replies_that_break_the_rules_fall_back(library, bad):
    room, generated, report = run(library, Recorder(bad), week=2)
    assert report.fallbacks and not report.generated
    assert generated == room


def test_another_chapters_passage_cannot_be_cited(library):
    room = build_room(library, COURSE, 1)
    other = room["chapters"][1]["sections"][0]["cites"][0]

    def reply(prompt):
        data = json.loads(good_reply(prompt))
        data["sections"][0]["cites"] = [other] if other not in ids_in(prompt) else data["sections"][0]["cites"]
        return json.dumps(data)

    _, report = generate_room(room, week_texts(library, COURSE, 1), Recorder(reply), "fake")
    title = room["chapters"][0]["title"]
    assert title in report.fallbacks and "not one of this chapter's passages" in report.fallbacks[title]


def test_a_model_that_is_not_running_falls_back_without_retrying(library):
    model = Recorder(ConnectionError("connection refused"))
    room, generated, report = run(library, model, week=1)
    assert generated == room and len(report.fallbacks) == len(room["chapters"])
    assert all("model call failed" in why for why in report.fallbacks.values())
    assert len(model.prompts) == 1  # one try in total, not one per chapter


def test_a_failing_chapter_does_not_stop_the_next(library):
    room = build_room(library, COURSE, 1)
    first_ids = set(c for s in room["chapters"][0]["sections"] for c in s["cites"])
    model = Recorder(lambda p: "nope" if set(ids_in(p)) <= first_ids else good_reply(p))
    generated, report = generate_room(room, week_texts(library, COURSE, 1), model, "fake")
    assert report.fallbacks.keys() == {room["chapters"][0]["title"]}
    assert report.generated == [room["chapters"][1]["title"]]
    assert generated["chapters"][0] == room["chapters"][0]


def test_retry_problems_point_at_the_reply(library):
    model = Recorder(lambda p: good_reply(p).replace('"answer_index": 0', '"answer_index": 5'), good_reply)
    run(library, model, week=2)
    assert "/sections/1/quiz/0/answer_index" in model.prompts[1]


def test_the_model_may_cite_only_passages_it_was_shown(library, monkeypatch):
    from studybuddy.room import generate

    monkeypatch.setattr(generate, "MAX_PROMPT_PASSAGE_CHARS", 1)
    room, generated, report = run(library, Recorder(good_reply), week=2)
    assert generated == room and "no passages to send" in next(iter(report.fallbacks.values()))


def test_a_reply_inside_a_code_fence_is_accepted(library):
    _, _, report = run(library, Recorder(lambda p: "Here you go:\n```json\n" + good_reply(p) + "\n```"), week=2)
    assert report.generated


def test_rebuilding_with_the_same_reply_keeps_ids_and_progress(library, tmp_path):
    _, first, _ = run(library, Recorder(good_reply))
    room_dir = tmp_path / "room"
    save_room(library, first, room_dir)
    made = next(s for s in sections(first) if s["provenance"] == "generated")
    question = next(q for s in sections(first) if s["provenance"] == "generated" for q in s.get("quiz", []))
    (room_dir / "progress.json").write_text(json.dumps({"read": [made["id"]], "answers": {question["id"]: 0}}))
    (room_dir / "edits.json").write_text(json.dumps({made["id"]: {"title": "My title"}}))
    _, second, _ = run(library, Recorder(good_reply))
    assert [s["id"] for s in sections(second)] == [s["id"] for s in sections(first)]
    saved = save_room(library, second, room_dir)
    assert saved.orphans == [] and "My title" in saved.html.read_text()


def test_cli_generate_uses_the_model_and_prints_the_report(library, tmp_path, monkeypatch, capsys):
    index = str(library.index_dir)
    library.close()
    monkeypatch.setattr(cli_module, "ollama_model", lambda name: Recorder(good_reply))
    code = cli_module.main(["--embedder", "hashing", "--index-dir", index, "room", COURSE, "2", "--generate"])
    out = capsys.readouterr().out
    assert code == 0 and "ollama:mistral: 1 chapters got generated sections" in out
    room = json.loads((room_dir_for(Path(index), COURSE, 2) / "room.json").read_text())
    assert room["builder"] == "extractive-v1+ollama:mistral"
    assert any(s["provenance"] == "generated" for s in sections(room))


def test_week_texts_cover_the_week(library):
    assert set(week_texts(library, COURSE, 1)) == week_chunk_ids(library, COURSE, 1)
