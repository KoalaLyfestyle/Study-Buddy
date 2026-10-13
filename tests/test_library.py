"""End-to-end check of import and search on the sample course.

Uses the offline hashing embedder, so it needs no model download and
gives the same results on every run.
"""

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from studybuddy import EmptyLibraryError, HashingEmbedder, Library
from studybuddy import library as library_module

SAMPLE = Path(__file__).resolve().parent.parent / "sample_course"
NOTES = "intro-to-optimisation/week-01/notes.md"
LAB = "intro-to-optimisation/week-01/gradient-descent-lab.ipynb"
PDF = "intro-to-optimisation/week-02/convexity-lecture.pdf"


class CountingEmbedder(HashingEmbedder):
    def __init__(self):
        super().__init__()
        self.documents_embedded = 0

    def embed_documents(self, texts):
        self.documents_embedded += len(texts)
        return super().embed_documents(texts)


@pytest.fixture
def course(tmp_path):
    folder = tmp_path / "course"
    shutil.copytree(SAMPLE, folder)
    return folder


@pytest.fixture
def make_library(tmp_path):
    opened = []

    def make(folder, embedder=None, index="index"):
        lib = Library(folder, tmp_path / index, embedder or CountingEmbedder())
        opened.append(lib)
        return lib

    yield make
    for lib in opened:
        lib.close()


def chunk_rows(lib):
    return {
        row[0]: (row[1], row[2])
        for row in lib.db.execute("SELECT chunk_id, relpath, rowid FROM chunks")
    }


def test_import_twice_does_no_extraction_or_embedding(course, make_library, monkeypatch):
    lib = make_library(course)
    first = lib.import_folder()
    assert sorted(first.added) == sorted([NOTES, LAB, PDF])
    assert first.failed == {}
    before = chunk_rows(lib)

    parsed = []
    monkeypatch.setattr(
        library_module.sources, "parse_file", lambda p: parsed.append(p) or pytest.fail("parsed")
    )
    embedded_before = lib.embedder.documents_embedded
    second = lib.import_folder()

    assert sorted(second.unchanged) == sorted([NOTES, LAB, PDF])
    assert second.files_parsed == 0 and second.chunks_embedded == 0
    assert parsed == []
    assert lib.embedder.documents_embedded == embedded_before
    assert chunk_rows(lib) == before


def test_changing_one_file_updates_only_its_records(course, make_library):
    lib = make_library(course)
    lib.import_folder()
    before = chunk_rows(lib)

    notes = course / NOTES
    notes.write_text(notes.read_text() + "\n## Momentum\n\nMomentum averages past gradients.\n")
    lib.embedder.documents_embedded = 0
    report = lib.import_folder()

    assert report.updated == [NOTES]
    assert sorted(report.unchanged) == sorted([LAB, PDF])
    assert report.files_parsed == 1
    after = chunk_rows(lib)
    assert lib.embedder.documents_embedded == sum(1 for _, (rel, _) in after.items() if rel == NOTES)
    others_before = {k: v for k, v in before.items() if v[0] != NOTES}
    others_after = {k: v for k, v in after.items() if v[0] != NOTES}
    assert others_before == others_after  # same IDs and same rows: untouched
    # Unchanged sections of the edited file keep their chunk IDs.
    assert {k for k, v in before.items() if v[0] == NOTES} < {k for k, v in after.items() if v[0] == NOTES}
    top = lib.search("momentum averages past gradients", k=1)[0]
    assert top.citation.relpath == NOTES
    assert top.citation.location["headings"][-1] == "Momentum"


def test_deleting_a_file_removes_its_results(course, make_library):
    lib = make_library(course)
    lib.import_folder()
    assert any(r.citation.relpath == PDF for r in lib.search("Hessian positive semidefinite", k=3))

    (course / PDF).unlink()
    report = lib.import_folder()

    assert report.removed == [PDF]
    assert all(r.citation.relpath != PDF for r in lib.search("Hessian positive semidefinite", k=10))
    assert lib.db.execute("SELECT COUNT(*) FROM chunks WHERE relpath=?", (PDF,)).fetchone()[0] == 0


def test_empty_folder_gives_an_actionable_message(tmp_path, make_library):
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "readme.txt").write_text("not a supported type")
    lib = make_library(empty)

    with pytest.raises(EmptyLibraryError) as raised:
        lib.import_folder()
    message = str(raised.value)
    assert str(empty.resolve()) in message
    assert ".pdf" in message and ".md" in message and ".ipynb" in message
    assert "python -m studybuddy import" in message

    with pytest.raises(EmptyLibraryError):
        lib.search("anything")


def test_missing_folder_gives_an_actionable_message(tmp_path, make_library):
    lib = make_library(tmp_path / "nope")
    with pytest.raises(EmptyLibraryError, match="does not exist"):
        lib.import_folder()


@pytest.mark.parametrize(
    "query, relpath, location_key",
    [
        ("learning rate too large loss diverges", NOTES, "headings"),
        ("every local minimum is a global minimum", PDF, "page"),
        ("def grad return 2 x minus 3", LAB, "cell_id"),
    ],
)
def test_every_result_points_to_a_valid_source_location(course, make_library, query, relpath, location_key):
    lib = make_library(course)
    lib.import_folder()
    results = lib.search(query, k=5)

    assert results[0].citation.relpath == relpath
    assert location_key in results[0].citation.location
    for result in results:
        assert lib.check_citation(result), result.citation
        assert lib.read_chunk(result.chunk_id).text == result.text
        c = result.citation
        assert c.course == "intro-to-optimisation"
        assert c.week == (2 if c.relpath == PDF else 1)


def test_filters_apply_before_ranking(course, make_library):
    lib = make_library(course)
    lib.import_folder()
    results = lib.search("gradient descent learning rate", k=10, week=2)
    assert results and all(r.citation.week == 2 for r in results)
    assert lib.search("gradient", course="no-such-course") == []
    assert lib.courses() == [{"course": "intro-to-optimisation", "files": 3, "weeks": [1, 2]}]


def test_notebook_outputs_are_not_indexed(course, make_library):
    lib = make_library(course)
    lib.import_folder()
    texts = [row[0] for row in lib.db.execute("SELECT text FROM chunks")]
    assert not any("OUTPUT-ONLY-MARKER" in t for t in texts)
    cell = lib.search("def grad return 2 x minus 3", k=1)[0].citation.location
    assert cell == {"cell_id": "descent-loop", "cell_index": 1, "cell_type": "code"}


def test_chunk_ids_are_stable_across_fresh_indexes(course, make_library):
    a = make_library(course, index="a")
    b = make_library(course, index="b")
    a.import_folder()
    b.import_folder()
    assert set(chunk_rows(a)) == set(chunk_rows(b))


def test_bad_file_is_reported_and_keeps_its_last_good_records(course, make_library):
    lib = make_library(course)
    lib.import_folder()
    lab_chunks = {k for k, v in chunk_rows(lib).items() if v[0] == LAB}

    (course / LAB).write_text("{ this is not json")
    report = lib.import_folder()

    assert LAB in report.failed and "JSONDecodeError" in report.failed[LAB]
    assert lib.failures() == {LAB: report.failed[LAB]}
    assert {k for k, v in chunk_rows(lib).items() if v[0] == LAB} == lab_chunks

    # Fixing the file clears the failure.
    shutil.copy(SAMPLE / LAB, course / LAB)
    assert lib.import_folder().updated == [LAB]
    assert lib.failures() == {}


def test_failed_rebuild_keeps_the_last_good_index(course, make_library):
    lib = make_library(course)
    lib.import_folder()
    before = chunk_rows(lib)

    class BrokenEmbedder(HashingEmbedder):
        def embed_documents(self, texts):
            raise RuntimeError("model unavailable")

    (course / NOTES).write_text("# Changed\n\nNew text.\n")
    lib.embedder = BrokenEmbedder()
    with pytest.raises(RuntimeError):
        lib.import_folder()

    assert chunk_rows(lib) == before


def test_import_never_writes_to_the_source_folder(course, make_library):
    def snapshot():
        return {
            p.relative_to(course).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in course.rglob("*")
            if p.is_file()
        }

    notes = course / NOTES
    notes.write_text(notes.read_text() + "\nA line I added by hand.\n")
    before = snapshot()
    lib = make_library(course)
    lib.import_folder()
    lib.import_folder()
    assert snapshot() == before


def test_cli_import_and_search(course, tmp_path, capsys):
    from studybuddy.__main__ import main

    index = str(tmp_path / "cli-index")
    assert main(["--index-dir", index, "--embedder", "hashing", "import", str(course)]) == 0
    assert "3 added" in capsys.readouterr().out
    assert main(["--index-dir", index, "--embedder", "hashing", "import", str(course)]) == 0
    assert "3 unchanged" in capsys.readouterr().out
    code = main(
        ["--index-dir", index, "--embedder", "hashing", "search", "convex epigraph", "--folder", str(course)]
    )
    assert code == 0
    assert "convexity-lecture.pdf, page 1" in capsys.readouterr().out


def test_sample_pdf_matches_its_generator():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "make_sample_pdf", SAMPLE.parent / "scripts" / "make_sample_pdf.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert (SAMPLE / PDF).read_bytes() == module.build_pdf(module.PAGES)


def test_sample_notebook_is_valid_json():
    data = json.loads((SAMPLE / LAB).read_text())
    assert data["nbformat"] == 4
