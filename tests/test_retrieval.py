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


@pytest.mark.parametrize("method", ["bm25", "tfidf", "dense", "hybrid"])
def test_filters_apply_before_ranking(corpus, method):
    lib, lex, labels = corpus
    for filters, prefix in [({"week": 5}, "week-05/"), ({"kind": "notebook"}, ".ipynb#"), ({"kind": "pdf", "week": 2}, "week-02/convexity")]:
        if method == "bm25":
            hits = lex.bm25("momentum learning rate Hessian", 10, **filters)
        elif method == "tfidf":
            hits = lex.tfidf("momentum learning rate Hessian", 10, **filters)
        elif method == "dense":
            hits = retrieval.dense(lib, "momentum learning rate Hessian", 10, **filters)
        else:
            hits = retrieval.hybrid(lib, lex, "momentum learning rate Hessian", 10, **filters)
        assert hits, (method, filters)
        assert all(prefix in labels[h.chunk_id] for h in hits), (method, filters, [labels[h.chunk_id] for h in hits])


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
    assert 0 < m["ndcg@10"] < 1


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
    # The hashing stand-in can never justify adopting hybrid.
    assert summary["decision"]["adopt_hybrid"] is False
    assert set(summary["results"]["heldout"]) == {"dense", "bm25", "tfidf", "hybrid"}
    assert (out / "report.md").read_text().startswith("# Retrieval comparison")
    for line in (out / "failures.jsonl").read_text().splitlines():
        assert json.loads(line)["kind"] in {"miss@5", "answered-unanswerable"}


def test_bad_label_fails_clearly(tmp_path):
    queries = tmp_path / "q.jsonl"
    queries.write_text(json.dumps({"id": "x", "type": "exact", "split": "tune", "query": "q", "relevant": ["week-09/none.md#Nope"]}) + "\n")
    with pytest.raises(SystemExit, match="labels match no chunk"):
        evaluate.main(["--embedder", "hashing", "--queries", str(queries), "--out", str(tmp_path / "o"), "--repeat", "1"])
