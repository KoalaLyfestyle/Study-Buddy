"""Study rooms built from the sample course (offline hashing embedder)."""

import copy
import json
import shutil
from pathlib import Path

import pytest

from studybuddy import HashingEmbedder, Library
from studybuddy.__main__ import main as cli
from studybuddy.room import (
    RoomValidationError,
    build_room,
    parse_room,
    render_room,
    room_dir_for,
    save_room,
    validate_room,
    week_chunk_ids,
)

SAMPLE = Path(__file__).resolve().parent.parent / "sample_course"
COURSE = "intro-to-optimisation"
NOTES = "intro-to-optimisation/week-01/notes.md"
LAB = "intro-to-optimisation/week-01/gradient-descent-lab.ipynb"


@pytest.fixture
def course(tmp_path):
    folder = tmp_path / "course"
    shutil.copytree(SAMPLE, folder)
    return folder


@pytest.fixture
def library(course, tmp_path):
    lib = Library(course, tmp_path / "index", HashingEmbedder())
    lib.import_folder()
    yield lib
    lib.close()


def sections(room):
    return [s for c in room["chapters"] for s in c["sections"]]


@pytest.mark.parametrize("week", [1, 2])
def test_sample_weeks_build_valid_rooms_with_working_citations(library, week):
    room = build_room(library, COURSE, week)
    validate_room(room, week_chunk_ids(library, COURSE, week))
    assert all(ids for ids in room["coverage"].values()), "every passage is used by some section"
    for section in sections(room):
        assert section["provenance"] == "source"
        for chunk_id in section["cites"]:
            found = library.read_chunk(chunk_id)
            assert found is not None and library.check_citation(found)


def test_week_one_has_every_kind_of_section(library):
    kinds = {s["kind"] for s in sections(build_room(library, COURSE, 1))}
    assert kinds == {"setup", "symbols", "explanation", "worked_example", "common_mistakes", "summary", "quiz"}


def test_quiz_answers_point_at_the_part_that_has_the_quote(library):
    room = build_room(library, COURSE, 1)
    for chapter in room["chapters"]:
        by_title = {s["title"]: s for s in chapter["sections"] if s["kind"] == "explanation"}
        for section in chapter["sections"]:
            for q in section.get("quiz", []):
                answer = by_title[q["choices"][q["answer_index"]]]
                quote = q["prompt"].split("says this: “", 1)[1].rstrip("”")
                assert any(quote in b.get("text", "") for b in answer["blocks"])
                assert q["cites"] == answer["cites"]


def test_builds_are_byte_identical(library, tmp_path):
    first = save_room(library, build_room(library, COURSE, 1), tmp_path / "a")
    second = save_room(library, build_room(library, COURSE, 1), tmp_path / "b")
    assert first.room_json.read_bytes() == second.room_json.read_bytes()
    assert first.html.read_bytes() == second.html.read_bytes()


def test_editing_the_source_keeps_ids_progress_and_authored_edits(library, course, tmp_path):
    room_dir = room_dir_for(library.index_dir, COURSE, 1)
    before = build_room(library, COURSE, 1)
    save_room(library, before, room_dir)
    stopping = next(s for s in sections(before) if s["title"] == "Stopping")
    rule = next(s for s in sections(before) if s["title"] == "The update rule")
    question = next(q for s in sections(before) for q in s.get("quiz", []))
    progress = {"read": [stopping["id"]], "answers": {question["id"]: 1}}
    edits = {rule["id"]: {"blocks": [{"type": "paragraph", "text": "My own words on the update rule."}]}}
    (room_dir / "progress.json").write_text(json.dumps(progress))
    (room_dir / "edits.json").write_text(json.dumps(edits))

    notes = course / NOTES
    notes.write_text(
        notes.read_text().replace("number of iterations.", "number of iterations, whichever comes first."),
        encoding="utf-8",
    )
    assert library.import_folder().updated == [NOTES]
    after = build_room(library, COURSE, 1)
    assert [s["id"] for s in sections(after)] == [s["id"] for s in sections(before)]

    saved = save_room(library, after, room_dir)
    assert saved.orphans == []
    assert json.loads((room_dir / "progress.json").read_text()) == progress
    html = saved.html.read_text()
    assert "whichever comes first" in html
    assert "My own words on the update rule." in html and "Edited by you" in html
    assert json.dumps(progress["read"]) in html  # the page starts from saved progress


def test_ids_that_disappear_are_kept_and_reported(library, tmp_path):
    room_dir = tmp_path / "room"
    room_dir.mkdir()
    (room_dir / "progress.json").write_text(json.dumps({"read": ["sec-000000000000"], "answers": {}}))
    saved = save_room(library, build_room(library, COURSE, 1), room_dir)
    assert saved.orphans == ["sec-000000000000"]
    assert "no longer has" in saved.html.read_text()


def _break(room, how):
    bad = copy.deepcopy(room)
    quiz = next(s for s in sections(bad) if s["kind"] == "quiz")
    if how == "duplicate id":
        sections(bad)[1]["id"] = sections(bad)[0]["id"]
    elif how == "unknown cite":
        sections(bad)[0]["cites"] = ["not-a-chunk"]
    elif how == "answer out of range":
        quiz["quiz"][0]["answer_index"] = len(quiz["quiz"][0]["choices"])
    elif how == "missing field":
        del sections(bad)[0]["provenance"]
    elif how == "bad block":
        sections(bad)[0]["blocks"] = [{"type": "paragraph"}]
    elif how == "empty quiz":
        quiz["quiz"] = []
    return bad


@pytest.mark.parametrize(
    "how", ["duplicate id", "unknown cite", "answer out of range", "missing field", "bad block", "empty quiz"]
)
def test_malformed_rooms_are_refused_with_a_path(library, how):
    room = build_room(library, COURSE, 1)
    with pytest.raises(RoomValidationError) as err:
        parse_room(json.dumps(_break(room, how)), week_chunk_ids(library, COURSE, 1))
    assert err.value.problems and all(p.startswith(("/", "edits")) for p in err.value.problems)


@pytest.mark.parametrize("text", ["", '{"schema_version": 1, "chapters": [', "[]"])
def test_truncated_or_wrong_json_is_refused(text):
    with pytest.raises(RoomValidationError):
        parse_room(text)


def test_cli_keeps_the_last_good_room_when_model_output_is_malformed(library, tmp_path, capsys):
    index = str(library.index_dir)
    library.close()
    assert cli(["--embedder", "hashing", "--index-dir", index, "room", COURSE, "1"]) == 0
    room_json = room_dir_for(Path(index), COURSE, 1) / "room.json"
    good = room_json.read_bytes()

    bad = tmp_path / "model.json"
    bad.write_text(good.decode()[: len(good) // 2])
    assert cli(["--embedder", "hashing", "--index-dir", index, "room", COURSE, "1", "--from-json", str(bad)]) == 2
    assert "not valid JSON" in capsys.readouterr().err
    assert room_json.read_bytes() == good

    # A valid room for a different week is refused too.
    other = json.loads(good)
    other["week"] = 2
    bad.write_text(json.dumps(other))
    assert cli(["--embedder", "hashing", "--index-dir", index, "room", COURSE, "1", "--from-json", str(bad)]) == 2
    assert room_json.read_bytes() == good


def test_a_bad_edits_file_is_refused_and_keeps_the_previous_room(library, tmp_path):
    room_dir = tmp_path / "room"
    room = build_room(library, COURSE, 1)
    first = save_room(library, room, room_dir)
    page = first.html.read_bytes()
    (room_dir / "edits.json").write_text(json.dumps({sections(room)[0]["id"]: {"blocks": [{"type": "video"}]}}))
    with pytest.raises(RoomValidationError):
        save_room(library, room, room_dir)
    assert first.html.read_bytes() == page


def test_notebook_code_is_shown_as_inert_text(library, course):
    lab = course / LAB
    data = json.loads(lab.read_text())
    data["cells"][1]["source"] = ["print('</code></pre><script>alert(1)</script>')\n"]
    lab.write_text(json.dumps(data))
    library.import_folder()
    room = build_room(library, COURSE, 1)
    html = render_room(room)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert html.count("<script") == 2  # the progress data and the page's own script


def test_an_empty_week_says_what_to_do(library):
    with pytest.raises(ValueError, match="python -m studybuddy courses"):
        build_room(library, COURSE, 9)


def test_citations_open_the_right_section_or_cell(library, tmp_path):
    import re

    saved = save_room(library, build_room(library, COURSE, 1), tmp_path / "room")
    html = saved.html.read_text()
    links = set(re.findall(r'href="(sources/[^"#]+)#([^"]+)"', html))
    assert links, "markdown and notebook citations link to the source view"
    for page, anchor in links:
        target = (tmp_path / "room" / page).read_text()
        start = target.index(f'id="{anchor}"')
        article = target[start : target.index("</article>", start)]
        # The anchored article holds the cited text, escaped.
        assert "&lt;" not in anchor and "<pre><code>" in article
    lab_cell = next(s for s in sections(build_room(library, COURSE, 1)) if s["kind"] == "worked_example")
    chunk = library.read_chunk(lab_cell["cites"][0])
    from studybuddy.room.render import anchor_for, citation_keys

    key = citation_keys(library, COURSE, 1)[(chunk.citation.relpath, json.dumps(chunk.citation.location, sort_keys=True))]
    page = (tmp_path / "room" / f"sources/{chunk.citation.doc_id}.html").read_text()
    start = page.index(f'id="{anchor_for(key)}"')
    assert "def grad(x):" in page[start : page.index("</article>", start)]


def _anchored_text(room_dir, href):
    page, anchor = href.split("#")
    target = (room_dir / page).read_text()
    assert target.count(f'id="{anchor}"') == 1
    start = target.index(f'id="{anchor}"')
    return target[start : target.index("</article>", start)]


def _links(html):
    import re

    return re.findall(r'<a href="(sources/[^"]+)">([^<]+)</a>', html)


def test_repeated_headings_stay_separate_and_link_to_their_own_place(course, tmp_path):
    week = course / COURSE / "week-03"
    week.mkdir()
    (week / "examples.md").write_text(
        "# Topic\n\n## Example\n\nFirst example text.\n\n## Other\n\nOther text here.\n\n"
        "## Example\n\nSecond example text.\n",
        encoding="utf-8",
    )
    lib = Library(course, tmp_path / "idx", HashingEmbedder())
    try:
        lib.import_folder()
        room = build_room(lib, COURSE, 3)
        parts = [s for s in sections(room) if s["kind"] == "explanation"]
        assert [s["title"] for s in parts] == ["Example", "Other", "Example"]
        assert len({s["id"] for s in parts}) == 3
        saved = save_room(lib, room, tmp_path / "room")
        html = saved.html.read_text()
        for part in parts:
            article = html[html.index(f'<article id="{part["id"]}"') :]
            href = _links(article)[0][0]
            text = part["blocks"][0]["text"]
            assert text in _anchored_text(tmp_path / "room", href)
    finally:
        lib.close()


def _notebook(cells):
    return json.dumps({"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 4})


def _md(text, cell_id=None):
    cell = {"cell_type": "markdown", "metadata": {}, "source": [text]}
    return cell | ({"id": cell_id} if cell_id else {})


def _code(text, cell_id=None):
    cell = {"cell_type": "code", "metadata": {}, "source": [text], "outputs": [], "execution_count": None}
    return cell | ({"id": cell_id} if cell_id else {})


def test_cells_without_ids_keep_their_ids_when_a_cell_is_added_elsewhere(course, tmp_path):
    week = course / COURSE / "week-04"
    week.mkdir()
    lab = week / "old-lab.ipynb"
    tail = [_md("## Part two\n\nNow the second part."), _code("y = 2")]
    lab.write_text(_notebook([_md("# Old lab\n\nNo cell ids here."), _code("x = 1"), *tail]))
    lib = Library(course, tmp_path / "idx", HashingEmbedder())
    try:
        lib.import_folder()
        before = {s["title"]: s["id"] for s in sections(build_room(lib, COURSE, 4)) if s["kind"] == "worked_example"}
        lab.write_text(
            _notebook([_md("# Old lab\n\nNo cell ids here."), _code("x = 1"), _code("x2 = 3"), *tail])
        )
        lib.import_folder()
        after = sections(build_room(lib, COURSE, 4))
        code_ids = [s["id"] for s in after if s["kind"] == "worked_example"]
        # "y = 2" moved from cell 4 to cell 5 but is under the same heading as before.
        y_section = next(s for s in after if s["kind"] == "worked_example" and s["blocks"][0]["text"] == "y = 2")
        assert y_section["id"] == before["Code, cell 4"]
        assert len(set(code_ids)) == 3
    finally:
        lib.close()


def test_duplicate_cell_ids_stay_separate(course, tmp_path):
    week = course / COURSE / "week-05"
    week.mkdir()
    (week / "dup.ipynb").write_text(
        _notebook([_md("# Dup lab\n\nTwo cells share an id.", "intro"), _code("a = 1", "same"), _code("b = 2", "same")])
    )
    lib = Library(course, tmp_path / "idx", HashingEmbedder())
    try:
        lib.import_folder()
        room = build_room(lib, COURSE, 5)
        code = [s for s in sections(room) if s["kind"] == "worked_example"]
        assert [s["blocks"][0]["text"] for s in code] == ["a = 1", "b = 2"]
        saved = save_room(lib, room, tmp_path / "room")
        html = saved.html.read_text()
        for part in code:
            article = html[html.index(f'<article id="{part["id"]}"') :]
            assert part["blocks"][0]["text"] in _anchored_text(tmp_path / "room", _links(article)[0][0])
    finally:
        lib.close()


def test_cli_reports_a_missing_file_and_a_room_for_another_week(library, tmp_path, capsys):
    index = str(library.index_dir)
    library.close()
    base = ["--embedder", "hashing", "--index-dir", index, "room", COURSE]
    assert cli([*base, "1", "--from-json", str(tmp_path / "nope.json")]) == 2
    assert "cannot read it" in capsys.readouterr().err
    assert cli([*base, "1"]) == 0
    capsys.readouterr()
    week_one = room_dir_for(Path(index), COURSE, 1) / "room.json"
    assert cli([*base, "2", "--from-json", str(week_one)]) == 2
    assert "the file is for 'intro-to-optimisation' week 1" in capsys.readouterr().err


@pytest.mark.parametrize("answers", [{"q-x": True}, {"q-x": -1}, {"q-x": "1"}])
def test_bad_progress_answers_are_refused(library, tmp_path, answers):
    room_dir = tmp_path / "room"
    room_dir.mkdir()
    (room_dir / "progress.json").write_text(json.dumps({"read": [], "answers": answers}))
    with pytest.raises(RoomValidationError, match="answers"):
        save_room(library, build_room(library, COURSE, 1), room_dir)


def test_answers_past_the_last_choice_are_dropped(library, tmp_path):
    room = build_room(library, COURSE, 1)
    question = next(q for s in sections(room) for q in s.get("quiz", []))
    room_dir = tmp_path / "room"
    room_dir.mkdir()
    (room_dir / "progress.json").write_text(json.dumps({"read": [], "answers": {question["id"]: 9}}))
    html = save_room(library, room, room_dir).html.read_text()
    assert question["id"] not in html.split('id="progress">')[1].split("</script>")[0]


def test_course_names_that_look_alike_get_their_own_folders(tmp_path):
    assert room_dir_for(tmp_path, "a b", 1) != room_dir_for(tmp_path, "a-b", 1)
    assert room_dir_for(tmp_path, COURSE, 1) == tmp_path / "rooms" / COURSE / "week-01"
