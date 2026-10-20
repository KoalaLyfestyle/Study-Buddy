"""Compare dense, BM25, TF-IDF, and hybrid retrieval on the sample course.

    python -m studybuddy.evaluate                       # real embedder (BAAI/bge-small-en)
    python -m studybuddy.evaluate --embedder hashing    # offline stand-in, not semantic

The corpus is `sample_course/` plus the extra weeks in `eval/extra/`,
copied to a temp folder. Queries and relevance labels are in
`eval/queries.jsonl`. Results go to `eval/results/<embedder>/`:
`report.md`, `summary.json`, and `failures.jsonl` (one line per query
and method where no relevant chunk is in the top 5).

Hybrid settings (RRF constant and candidate depth) are tuned on the
`tune` split only. Decisions use the `heldout` split.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from . import retrieval
from .embeddings import Embedder, HashingEmbedder, default_embedder
from .library import Library

ROOT = Path(__file__).resolve().parent.parent
KS = (1, 3, 5, 10)
METHODS = ("dense", "bm25", "tfidf", "hybrid")
RRF_C = (10, 30, 60)
RRF_DEPTH = (10, 20, 40)
# Adopt hybrid only if it beats the best single method on held-out MRR by at
# least this much and does not lose Recall@5. Fixed before looking at results.
ADOPT_MARGIN = 0.05
TIMING_KEYS = ("build_seconds", "latency")  # vary run to run; kept out of tracked files


@dataclass
class Query:
    id: str
    type: str
    split: str
    query: str
    relevant: list[str]
    filters: dict


def load_queries(path: Path) -> list[Query]:
    queries = []
    for line in path.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            queries.append(
                Query(d["id"], d["type"], d["split"], d["query"], d["relevant"], d.get("filters", {}))
            )
    return queries


def build_corpus(dest: Path, sample: Path, extra: Path) -> Path:
    course = dest / "course"
    shutil.copytree(sample, course, ignore=shutil.ignore_patterns("*.txt"))
    if extra.is_dir():
        shutil.copytree(extra, course, dirs_exist_ok=True)
    return course


def chunk_label(relpath: str, location: dict) -> str:
    """`week-01/notes.md#Stopping`, `week-02/x.pdf#p2`, `week-03/lab.ipynb#cell-id`."""
    short = relpath.split("/", 1)[1] if "/" in relpath else relpath
    if "page" in location:
        anchor = f"p{location['page']}"
    elif "cell_id" in location:
        anchor = location["cell_id"]
    else:
        anchor = (location.get("headings") or [""])[-1]
    return f"{short}#{anchor}"


def chunk_labels(library: Library) -> dict[str, str]:
    rows = library.db.execute("SELECT chunk_id, relpath, location FROM chunks").fetchall()
    return {cid: chunk_label(rel, json.loads(loc)) for cid, rel, loc in rows}


def metrics(ranked: list[str], relevant: set[str]) -> dict[str, float]:
    out = {}
    for k in KS:
        out[f"recall@{k}"] = len(relevant & set(ranked[:k])) / len(relevant)
    rr = 0.0
    for rank, cid in enumerate(ranked[:10], start=1):
        if cid in relevant:
            rr = 1.0 / rank
            break
    out["mrr@10"] = rr
    dcg = sum(1.0 / math.log2(r + 1) for r, cid in enumerate(ranked[:10], start=1) if cid in relevant)
    ideal = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(relevant), 10) + 1))
    out["ndcg@10"] = dcg / ideal
    return out


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def run(args) -> dict:
    queries = load_queries(Path(args.queries))
    embedder: Embedder = HashingEmbedder() if args.embedder == "hashing" else (
        default_embedder() if args.embedder is None else _named(args.embedder)
    )

    with tempfile.TemporaryDirectory() as tmp:
        course = build_corpus(Path(tmp), Path(args.sample), Path(args.extra))
        library = Library(course, Path(tmp) / "index", embedder)

        t0 = time.perf_counter()
        report = library.import_folder()
        dense_build = time.perf_counter() - t0
        if report.failed:
            raise SystemExit(f"import failed: {report.failed}")

        t0 = time.perf_counter()
        lexical = retrieval.LexicalIndex(library)
        lexical_build = time.perf_counter() - t0

        labels = chunk_labels(library)
        by_label: dict[str, str] = {}
        for cid, label in labels.items():
            if label in by_label:
                raise SystemExit(f"label {label} is not unique; add a more specific anchor")
            by_label[label] = cid
        for q in queries:
            missing = [r for r in q.relevant if r not in by_label]
            if missing:
                raise SystemExit(f"{q.id}: labels match no chunk: {missing}")

        def retrieve(method: str, q: Query, k: int, c: int = 60, depth: int = 20) -> list[str]:
            f = q.filters
            if method == "dense":
                hits = retrieval.dense(library, q.query, k, f.get("course"), f.get("week"), f.get("kind"))
            elif method == "bm25":
                hits = lexical.bm25(q.query, k, f.get("course"), f.get("week"), f.get("kind"))
            elif method == "tfidf":
                hits = lexical.tfidf(q.query, k, f.get("course"), f.get("week"), f.get("kind"))
            else:
                hits = retrieval.hybrid(
                    library, lexical, q.query, k, f.get("course"), f.get("week"), f.get("kind"), depth=depth, c=c
                )
            return [h.chunk_id for h in hits]

        # Cold: a fresh embedder, library handle, and lexical index, then one query,
        # before anything else has run. Includes model load and index build.
        cold = {}
        for method in METHODS:
            t0 = time.perf_counter()
            fresh_embedder = _fresh(embedder)
            fresh = Library(course, Path(tmp) / "index", fresh_embedder)
            fresh_lexical = retrieval.LexicalIndex(fresh) if method != "dense" else None
            q = queries[0]
            if method == "dense":
                retrieval.dense(fresh, q.query, 10)
            elif method == "bm25":
                fresh_lexical.bm25(q.query, 10)
            elif method == "tfidf":
                fresh_lexical.tfidf(q.query, 10)
            else:
                retrieval.hybrid(fresh, fresh_lexical, q.query, 10)
            cold[method] = round((time.perf_counter() - t0) * 1000, 2)
            fresh.close()

        # Tune hybrid on the tune split only.
        tune = [q for q in queries if q.split == "tune" and q.relevant]
        grid = []
        for c in RRF_C:
            for depth in RRF_DEPTH:
                score = mean(
                    metrics(retrieve("hybrid", q, 10, c, depth), {by_label[r] for r in q.relevant})["mrr@10"]
                    for q in tune
                )
                grid.append({"c": c, "depth": depth, "tune_mrr@10": round(score, 4)})
        best = max(grid, key=lambda g: (g["tune_mrr@10"], -g["c"], g["depth"]))
        hybrid_c, hybrid_depth = best["c"], best["depth"]

        # Warm: median over repeated passes with everything loaded.
        latency = {}
        for method in METHODS:
            times = []
            for q in queries * args.repeat:
                t0 = time.perf_counter()
                retrieve(method, q, 10, hybrid_c, hybrid_depth)
                times.append(time.perf_counter() - t0)
            latency[method] = {
                "cold_ms": cold[method],
                "warm_median_ms": round(statistics.median(times) * 1000, 3),
            }

        per_query = []
        for q in queries:
            relevant = {by_label[r] for r in q.relevant}
            row = {"id": q.id, "type": q.type, "split": q.split, "query": q.query, "filters": q.filters, "methods": {}}
            for method in METHODS:
                ranked = retrieve(method, q, 10, hybrid_c, hybrid_depth)
                entry = {"top": [labels[c] for c in ranked[:5]], "n_results": len(ranked)}
                if relevant:
                    entry.update(metrics(ranked, relevant))
                row["methods"][method] = entry
            per_query.append(row)

        dense_bytes = library.db.execute("SELECT COALESCE(SUM(LENGTH(embedding)), 0) FROM chunks").fetchone()[0]
        sizes = {"dense": dense_bytes, **lexical.size_bytes()}
        sizes["hybrid"] = sizes["dense"] + sizes["bm25"]
        n_chunks = library.chunk_count()
        library.close()

    summary = {
        "embedder": embedder.id,
        "semantic": not isinstance(embedder, HashingEmbedder),
        "corpus": {"files": len(report.added), "chunks": n_chunks},
        "queries": {
            split: {
                "answerable": sum(1 for q in queries if q.split == split and q.relevant),
                "unanswerable": sum(1 for q in queries if q.split == split and not q.relevant),
            }
            for split in ("tune", "heldout")
        },
        "build_seconds": {
            "dense": round(dense_build, 3),
            "bm25+tfidf": round(lexical_build, 3),
        },
        "index_bytes": sizes,
        "latency": latency,
        "hybrid": {"chosen": {"c": hybrid_c, "depth": hybrid_depth}, "grid": grid},
        "results": {},
    }
    for split in ("tune", "heldout"):
        rows = [r for r in per_query if r["split"] == split]
        answerable = [r for r in rows if r["type"] != "unanswerable"]
        unanswerable = [r for r in rows if r["type"] == "unanswerable"]
        summary["results"][split] = {}
        for method in METHODS:
            m = {
                key: round(mean(r["methods"][method][key] for r in answerable), 4)
                for key in [f"recall@{k}" for k in KS] + ["mrr@10", "ndcg@10"]
            }
            m["by_type_mrr@10"] = {
                t: round(mean(r["methods"][method]["mrr@10"] for r in answerable if r["type"] == t), 4)
                for t in sorted({r["type"] for r in answerable})
            }
            m["unanswerable_no_result_rate"] = (
                round(mean(1.0 if r["methods"][method]["n_results"] == 0 else 0.0 for r in unanswerable), 4)
                if unanswerable
                else None
            )
            summary["results"][split][method] = m

    held = summary["results"]["heldout"]
    singles = [m for m in METHODS if m != "hybrid" and (m != "dense" or summary["semantic"])]
    best_single = max(singles, key=lambda m: (held[m]["mrr@10"], held[m]["recall@5"]))
    gain = held["hybrid"]["mrr@10"] - held[best_single]["mrr@10"]
    adopt = (
        summary["semantic"]
        and gain >= ADOPT_MARGIN
        and held["hybrid"]["recall@5"] >= held[best_single]["recall@5"]
    )
    wins = losses = 0
    for r in per_query:
        if r["split"] == "heldout" and r["type"] != "unanswerable":
            h, s = r["methods"]["hybrid"]["mrr@10"], r["methods"][best_single]["mrr@10"]
            wins += h > s
            losses += h < s
    summary["decision"] = {
        "best_single_method": best_single,
        "hybrid_minus_best_mrr@10": round(gain, 4),
        "hybrid_wins_losses_vs_best": [wins, losses],
        "adopt_hybrid": adopt,
        "rule": f"adopt hybrid only with a semantic embedder, held-out MRR@10 gain >= {ADOPT_MARGIN}, "
        "and no loss in held-out Recall@5",
    }

    out = Path(args.out) if args.out else ROOT / "eval" / "results" / re.sub(r"[^A-Za-z0-9.-]+", "_", embedder.id)
    out.mkdir(parents=True, exist_ok=True)
    # Timings change on every run, so they go to an untracked timings.json and the
    # tracked files keep only what is reproducible (#12).
    timings = {k: summary[k] for k in TIMING_KEYS}
    stable = {k: v for k, v in summary.items() if k not in TIMING_KEYS}
    (out / "summary.json").write_text(json.dumps(stable, indent=2) + "\n")
    (out / "timings.json").write_text(json.dumps(timings, indent=2) + "\n")
    with open(out / "failures.jsonl", "w") as fh:
        for r in per_query:
            for method, e in r["methods"].items():
                if r["type"] == "unanswerable":
                    if e["n_results"]:
                        fh.write(json.dumps({"id": r["id"], "method": method, "kind": "answered-unanswerable",
                                             "query": r["query"], "top": e["top"][:3]}) + "\n")
                elif e["recall@5"] == 0:
                    expected = next(q.relevant for q in queries if q.id == r["id"])
                    fh.write(json.dumps({"id": r["id"], "method": method, "kind": "miss@5", "split": r["split"],
                                         "type": r["type"], "query": r["query"], "expected": expected,
                                         "top": e["top"][:3]}) + "\n")
    (out / "per_query.json").write_text(json.dumps(per_query, indent=1) + "\n")
    (out / "report.md").write_text(render_report(summary, timings=False))
    print(render_report(summary, timings=True))
    print(f"Wrote {out}")
    return summary


def _fresh(embedder: Embedder) -> Embedder:
    """A new instance of the same embedder, so a cold run reloads the model."""
    if isinstance(embedder, HashingEmbedder):
        return HashingEmbedder(embedder.dim)
    return _named(embedder.model_name)


def _named(name: str) -> Embedder:
    from .embeddings import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(name)


def render_report(s: dict, timings: bool = True) -> str:
    """The Markdown report. With timings=False the Cost table keeps only index sizes,
    so the tracked report.md is the same on every run."""
    lines = [f"# Retrieval comparison: {s['embedder']}", ""]
    if not s["semantic"]:
        lines += [
            "> The dense column uses the hashing embedder, an offline lexical stand-in. It says nothing",
            "> about semantic search, so dense and hybrid results here do not count toward the decision.",
            "> Run `python -m studybuddy.evaluate` with the real model for that.",
            "",
        ]
    lines += [
        "Labels are model-generated and provisional until a person checks them.",
        "",
        f"Corpus: {s['corpus']['files']} files, {s['corpus']['chunks']} chunks. "
        f"Queries: tune {s['queries']['tune']}, held-out {s['queries']['heldout']}.",
        "",
    ]
    for split in ("heldout", "tune"):
        lines += [f"## {'Held-out' if split == 'heldout' else 'Tune'} split", "",
                  "| Method | R@1 | R@3 | R@5 | R@10 | MRR@10 | nDCG@10 | Unanswerable with no result |",
                  "|---|---|---|---|---|---|---|---|"]
        for method, m in s["results"][split].items():
            nores = "n/a" if m["unanswerable_no_result_rate"] is None else f"{m['unanswerable_no_result_rate']:.2f}"
            lines.append(
                f"| {method} | {m['recall@1']:.2f} | {m['recall@3']:.2f} | {m['recall@5']:.2f} | "
                f"{m['recall@10']:.2f} | {m['mrr@10']:.3f} | {m['ndcg@10']:.3f} | {nores} |"
            )
        types = sorted(next(iter(s["results"][split].values()))["by_type_mrr@10"])
        lines += ["", "MRR@10 by query type:", "", "| Method | " + " | ".join(types) + " |",
                  "|---|" + "---|" * len(types)]
        for method, m in s["results"][split].items():
            lines.append(f"| {method} | " + " | ".join(f"{m['by_type_mrr@10'][t]:.2f}" for t in types) + " |")
        lines.append("")
    if timings:
        lines += ["## Cost", "", "| Method | Build (s) | Index size (bytes) | Cold query (ms) | Warm median (ms) |",
                  "|---|---|---|---|---|"]
        build = {"dense": s["build_seconds"]["dense"], "bm25": s["build_seconds"]["bm25+tfidf"],
                 "tfidf": s["build_seconds"]["bm25+tfidf"],
                 "hybrid": s["build_seconds"]["dense"] + s["build_seconds"]["bm25+tfidf"]}
        for method, lat in s["latency"].items():
            lines.append(f"| {method} | {build[method]:.3f} | {s['index_bytes'][method]} | {lat['cold_ms']} | {lat['warm_median_ms']} |")
    else:
        lines += ["## Cost", "",
                  "Build and query times change on every run, so they are printed to the console and saved",
                  "next to this file in `timings.json`, which git ignores.", "",
                  "| Method | Index size (bytes) |", "|---|---|"]
        for method in s["index_bytes"]:
            lines.append(f"| {method} | {s['index_bytes'][method]} |")
    d = s["decision"]
    lines += [
        "",
        "Notes on the columns:",
        "",
        "- Build: dense includes parsing and importing; BM25 and TF-IDF are built together from the imported chunks.",
        "- Index size: dense counts the stored float32 vectors; BM25 counts the whole in-memory FTS5 database,",
        "  which holds its own copy of the chunk text; TF-IDF is an estimate for sparse storage (8 bytes per",
        "  non-zero entry). Treat them as orders of magnitude.",
        "- Cold: a fresh embedder, library handle, and lexical index plus one query, measured before any other",
        "  query runs. With the real model this includes loading the weights, but not importing torch,",
        "  which the run has already done. Warm: median over repeated passes.",
        "- Unanswerable with no result: dense and hybrid always return their top k, so they score 0 here",
        "  by design. Lexical methods return nothing only when no query term appears in the corpus.",
        "  None of the methods has a calibrated score threshold yet.",
        "",
        "## Decision",
        "",
        f"- Hybrid settings chosen on the tune split: RRF c = {s['hybrid']['chosen']['c']}, depth = {s['hybrid']['chosen']['depth']}.",
        f"- Best single method on held-out: **{d['best_single_method']}**.",
        f"- Hybrid minus best single, held-out MRR@10: {d['hybrid_minus_best_mrr@10']:+.3f} "
        f"(hybrid better on {d['hybrid_wins_losses_vs_best'][0]} queries, worse on {d['hybrid_wins_losses_vs_best'][1]}).",
        f"- Rule: {d['rule']}.",
        (
            f"- **Adopt hybrid: {'yes' if d['adopt_hybrid'] else 'no'}.**"
            if s["semantic"]
            else "- **Not decided: this run has no semantic embedder.** The BM25 and TF-IDF rows do not depend on the embedder."
        ),
        "",
        "Per-query misses are in `failures.jsonl`; every ranking is in `per_query.json`.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m studybuddy.evaluate", description=__doc__.split("\n")[0])
    p.add_argument("--embedder", default=None, help="'hashing' or a sentence-transformers model; default from STUDYBUDDY_EMBEDDER")
    p.add_argument("--queries", default=str(ROOT / "eval" / "queries.jsonl"))
    p.add_argument("--sample", default=str(ROOT / "sample_course"))
    p.add_argument("--extra", default=str(ROOT / "eval" / "extra"))
    p.add_argument("--out", default=None, help="default: eval/results/<embedder>")
    p.add_argument("--repeat", type=int, default=5, help="passes over the queries for warm latency")
    run(p.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
