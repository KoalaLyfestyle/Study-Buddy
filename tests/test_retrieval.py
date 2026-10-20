"""Lexical, hybrid, and evaluation checks on the sample course plus eval/extra."""

import json
import shutil
from pathlib import Path

import pytest

from studybuddy import HashingEmbedder, Library
from studybuddy import evaluate, retrieval

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def corpus(tmp_path):
    course = evaluate.build_corpus(tmp_path, ROOT / "sample_course", ROOT / "eval" / "extra")
    lib = Library(course, tmp_path / "index", HashingEmbedder())
    lib.import_folder()
    lex = retrieval.LexicalIndex(lib)
    labels = evaluate.chunk_labels(lib)
    yield lib, lex, labels
    lib.close()


def test_bm25_finds_exact_terms(corpus):
    lib, lex, labels = corpus
    hits = lex.bm25("Armijo condition", k=3)
    assert labels[hits[0].chunk_id] == "week-04/second-order-methods.pdf#p3"
    assert hits == sorted(hits, key=lambda h: -h.score)


FILTERS = [({"week": 5}, "week-05/"), ({"kind": "notebook"}, ".ipynb#"), ({"kind": "pdf", "week": 2}, "week-02/convexity")]
QUERY = "momentum learning rate Hessian"


@pytest.mark.parametrize("method", ["bm25", "tfidf", "dense"])
def test_filters_apply_before_ranking(corpus, method):
    """A filtered search equals the full ranking restricted to eligible chunks.

    Filtering after taking the top k would return fewer hits here, because
    most of the top 10 unfiltered hits are outside each filter.
    """
    lib, lex, labels = corpus
    search = {
        "bm25": lambda k, **f: lex.bm25(QUERY, k, **f),
        "tfidf": lambda k, **f: lex.tfidf(QUERY, k, **f),
        "dense": lambda k, **f: retrieval.dense(lib, QUERY, k, **f),
    }[method]
    full = [h.chunk_id for h in search(1000)]
    for filters, marker in FILTERS:
        eligible = [c for c in full if marker in labels[c]]
        got = [h.chunk_id for h in search(3, **filters)]
        assert got == eligible[:3], (method, filters)
        assert len(got) == min(3, len(eligible)) > 0


def test_hybrid_respects_filters(corpus):
    lib, lex, labels = corpus
    for filters, marker in FILTERS:
        hits = retrieval.hybrid(lib, lex, QUERY, 10, **filters)
        assert hits and all(marker in labels[h.chunk_id] for h in hits), filters


def test_bm25_treats_query_syntax_as_text(corpus):
    _, lex, _ = corpus
    for query in ['AND OR NOT', 'theta*', '"unbalanced', 'week:3', 'x^2 + y^2', 'NEAR(a b)']:
        lex.bm25(query, 5)  # must not raise sqlite3.OperationalError
    assert lex.bm25("the of and", 5) == []  # stop words only: no query, no results


def test_tfidf_returns_nothing_without_shared_terms(corpus):
    _, lex, _ = corpus
    assert lex.tfidf("zebra xylophone", 5) == []


def test_rrf_keeps_candidates_found_by_only_one_method():
    dense_only = [retrieval.Hit("semantic-match", 0.9), retrieval.Hit("both", 0.8)]
    lexical_only = [retrieval.Hit("both", 5.0), retrieval.Hit("keyword-match", 4.0)]
    fused = retrieval.rrf([dense_only, lexical_only], k=3, c=60)
    assert [h.chunk_id for h in fused] == ["both", "semantic-match", "keyword-match"]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)


def test_metrics():
    m = evaluate.metrics(["a", "x", "b"], {"a", "b"})
    assert m["recall@1"] == 0.5 and m["recall@3"] == 1.0
    assert m["mrr@10"] == 1.0
    assert m["ndcg@10"] == pytest.approx((1 + 1 / 2) / (1 + 1 / 1.584962500721156))


def test_every_label_matches_one_chunk(corpus):
    _, _, labels = corpus
    known = set(labels.values())
    assert len(known) == len(labels), "chunk labels must be unique"
    for q in evaluate.load_queries(ROOT / "eval" / "queries.jsonl"):
        assert set(q.relevant) <= known, q.id
    queries = evaluate.load_queries(ROOT / "eval" / "queries.jsonl")
    assert 30 <= len(queries) <= 50
    assert {q.split for q in queries} == {"tune", "heldout"}
    assert {"exact", "paraphrase", "equation", "notebook", "cross-week", "unanswerable"} <= {q.type for q in queries}


def test_evaluate_command_writes_report_and_failures(tmp_path, capsys):
    out = tmp_path / "results"
    assert evaluate.main(["--embedder", "hashing", "--out", str(out), "--repeat", "1"]) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["semantic"] is False
    # The hashing stand-in can never justify adopting hybrid, and the report says it is undecided.
    assert summary["decision"]["adopt_hybrid"] is False
    assert "Not decided" in (out / "report.md").read_text()
    assert all(v["cold_ms"] > 0 for v in summary["latency"].values())
    assert set(summary["results"]["heldout"]) == {"dense", "bm25", "tfidf", "hybrid"}
    assert (out / "report.md").read_text().startswith("# Retrieval comparison")
    for line in (out / "failures.jsonl").read_text().splitlines():
        assert json.loads(line)["kind"] in {"miss@5", "answered-unanswerable"}


def test_bad_label_fails_clearly(tmp_path):
    queries = tmp_path / "q.jsonl"
    queries.write_text(json.dumps({"id": "x", "type": "exact", "split": "tune", "query": "q", "relevant": ["week-09/none.md#Nope"]}) + "\n")
    with pytest.raises(SystemExit, match="labels match no chunk"):
        evaluate.main(["--embedder", "hashing", "--queries", str(queries), "--out", str(tmp_path / "o"), "--repeat", "1"])


def test_code_cells_are_found_by_the_words_around_them(corpus):
    """q25 (a tune query): the code cell, not only the Markdown intro, ranks near the top."""
    lib, lex, labels = corpus
    top = [labels[h.chunk_id] for h in lex.bm25("code for the gradient of the Rosenbrock function", k=3)]
    assert "week-03/momentum-lab.ipynb#rosenbrock-grad" in top


def test_code_cell_context_is_indexed_but_never_cited(corpus):
    lib, _, labels = corpus
    rows = lib.db.execute(
        "SELECT c.chunk_id, c.text, c.index_text, c.location FROM chunks c WHERE c.relpath LIKE '%momentum-lab.ipynb'"
    ).fetchall()
    by_label = {labels[r[0]]: r for r in rows}
    chunk_id, text, index_text, location = by_label["week-03/momentum-lab.ipynb#rosenbrock-grad"]
    assert text.startswith("import numpy as np") and "Rosenbrock" not in text
    assert index_text.startswith("Code cell in Lab 3: momentum on the Rosenbrock valley.")
    assert index_text.endswith(text)
    # The ID depends on the cited text only, as before parser version 2.
    from studybuddy.chunking import make_chunk_id
    from studybuddy.sources import doc_id_for

    relpath = next(r for r in lib.db.execute("SELECT relpath FROM chunks WHERE chunk_id=?", (chunk_id,)))[0]
    assert chunk_id == make_chunk_id(doc_id_for(relpath), json.loads(location), 0, text)
    for r in rows:
        assert lib.check_citation(lib.read_chunk(r[0]))
    markdown = by_label["week-03/momentum-lab.ipynb#intro"]
    assert markdown[2] == markdown[1]  # Markdown cells get no extra context


def test_code_cell_embeddings_use_the_cell_text_only(corpus):
    """The context is for keyword search; with the real model it made dense ranking worse."""
    import numpy as np

    lib, _, labels = corpus
    rows = lib.db.execute(
        "SELECT chunk_id, text, index_text, embedding FROM chunks WHERE relpath LIKE '%.ipynb'"
    ).fetchall()
    assert any(index_text != text for _, text, index_text, _ in rows)
    for chunk_id, text, index_text, blob in rows:
        expected = np.asarray(lib.embedder.embed_documents([text])[0], dtype=np.float32)
        assert np.allclose(np.frombuffer(blob, dtype=np.float32), expected), labels[chunk_id]


def test_a_heading_only_cell_is_not_repeated_in_the_context():
    from studybuddy.sources import parse_notebook

    nb = {"cells": [
        {"cell_type": "markdown", "source": "# Lab 3"},
        {"cell_type": "code", "source": "x = 1"},
        {"cell_type": "markdown", "source": "## Part two\n\nNow fit the model."},
        {"cell_type": "code", "source": "y = 2"},
    ]}
    first, second = [s.context for s in parse_notebook(json.dumps(nb)) if s.location["cell_type"] == "code"]
    assert first == "Code cell in Lab 3."
    assert second == "Code cell in Lab 3. ## Part two Now fit the model."
