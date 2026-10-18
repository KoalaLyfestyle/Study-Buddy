# Retrieval comparison: hashing-v1-512

> The dense column uses the hashing embedder, an offline lexical stand-in. It says nothing
> about semantic search, so dense and hybrid results here do not count toward the decision.
> Run `python -m studybuddy.evaluate` with the real model for that.

Labels are model-generated and provisional until a person checks them.

Corpus: 10 files, 42 chunks. Queries: tune {'answerable': 20, 'unanswerable': 3}, held-out {'answerable': 18, 'unanswerable': 3}.

## Held-out split

| Method | R@1 | R@3 | R@5 | R@10 | MRR@10 | nDCG@10 | Unanswerable with no result |
|---|---|---|---|---|---|---|---|
| dense | 0.50 | 0.69 | 0.81 | 0.86 | 0.662 | 0.698 | 0.00 |
| bm25 | 0.75 | 0.92 | 1.00 | 1.00 | 0.884 | 0.912 | 0.67 |
| tfidf | 0.75 | 0.86 | 0.86 | 1.00 | 0.877 | 0.900 | 1.00 |
| hybrid | 0.64 | 0.86 | 0.92 | 0.97 | 0.818 | 0.845 | 0.00 |

MRR@10 by query type:

| Method | cross-week | equation | exact | filtered | notebook | paraphrase |
|---|---|---|---|---|---|---|
| dense | 0.75 | 1.00 | 0.88 | 1.00 | 0.17 | 0.32 |
| bm25 | 1.00 | 1.00 | 1.00 | 1.00 | 0.29 | 0.87 |
| tfidf | 1.00 | 1.00 | 1.00 | 1.00 | 0.15 | 0.90 |
| hybrid | 1.00 | 1.00 | 1.00 | 1.00 | 0.24 | 0.65 |

## Tune split

| Method | R@1 | R@3 | R@5 | R@10 | MRR@10 | nDCG@10 | Unanswerable with no result |
|---|---|---|---|---|---|---|---|
| dense | 0.50 | 0.63 | 0.68 | 0.83 | 0.693 | 0.688 | 0.00 |
| bm25 | 0.70 | 0.85 | 0.90 | 0.97 | 0.858 | 0.890 | 0.00 |
| tfidf | 0.60 | 0.87 | 0.87 | 0.89 | 0.799 | 0.814 | 0.33 |
| hybrid | 0.66 | 0.83 | 0.88 | 0.95 | 0.861 | 0.856 | 0.00 |

MRR@10 by query type:

| Method | cross-week | equation | exact | filtered | notebook | paraphrase |
|---|---|---|---|---|---|---|
| dense | 0.78 | 0.83 | 1.00 | 0.56 | 0.67 | 0.38 |
| bm25 | 0.75 | 1.00 | 1.00 | 1.00 | 1.00 | 0.58 |
| tfidf | 0.71 | 1.00 | 1.00 | 1.00 | 0.83 | 0.47 |
| hybrid | 1.00 | 1.00 | 1.00 | 1.00 | 0.83 | 0.55 |

## Cost

| Method | Build (s) | Index size (bytes) | Cold query (ms) | Warm median (ms) |
|---|---|---|---|---|
| dense | 0.082 | 86016 | 1.31 | 0.265 |
| bm25 | 0.003 | 40960 | 2.64 | 0.036 |
| tfidf | 0.003 | 8016 | 2.58 | 0.024 |
| hybrid | 0.085 | 126976 | 3.1 | 0.58 |

Notes on the columns:

- Build: dense includes parsing and importing; BM25 and TF-IDF are built together from the imported chunks.
- Index size: dense counts the stored float32 vectors; BM25 counts the whole in-memory FTS5 database,
  which holds its own copy of the chunk text; TF-IDF is an estimate for sparse storage (8 bytes per
  non-zero entry). Treat them as orders of magnitude.
- Cold: a fresh embedder, library handle, and lexical index plus one query, measured before any other
  query runs. With the real model this includes loading the weights, but not importing torch,
  which the run has already done. Warm: median over repeated passes.
- Unanswerable with no result: dense and hybrid always return their top k, so they score 0 here
  by design. Lexical methods return nothing only when no query term appears in the corpus.
  None of the methods has a calibrated score threshold yet.

## Decision

- Hybrid settings chosen on the tune split: RRF c = 30, depth = 40.
- Best single method on held-out: **bm25**.
- Hybrid minus best single, held-out MRR@10: -0.066 (hybrid better on 2 queries, worse on 3).
- Rule: adopt hybrid only with a semantic embedder, held-out MRR@10 gain >= 0.05, and no loss in held-out Recall@5.
- **Not decided: this run has no semantic embedder.** The BM25 and TF-IDF rows do not depend on the embedder.

Per-query misses are in `failures.jsonl`; every ranking is in `per_query.json`.
