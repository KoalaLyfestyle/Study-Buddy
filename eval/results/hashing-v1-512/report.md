# Retrieval comparison: hashing-v1-512

> The dense column uses the hashing embedder, an offline lexical stand-in. It says nothing
> about semantic search, so dense and hybrid results here do not count toward the decision.
> Run `python -m studybuddy.evaluate` with the real model for that.

Labels are model-generated and provisional until a person checks them.

Corpus: 10 files, 42 chunks. Queries: tune {'answerable': 20, 'unanswerable': 3}, held-out {'answerable': 18, 'unanswerable': 3}.

## Held-out split

| Method | R@1 | R@3 | R@5 | R@10 | MRR@10 | nDCG@10 | Unanswerable with no result |
|---|---|---|---|---|---|---|---|
| dense | 0.50 | 0.64 | 0.75 | 0.86 | 0.650 | 0.692 | 0.00 |
| bm25 | 0.75 | 0.92 | 1.00 | 1.00 | 0.884 | 0.912 | 0.67 |
| tfidf | 0.75 | 0.86 | 0.86 | 1.00 | 0.877 | 0.900 | 1.00 |
| hybrid | 0.58 | 0.81 | 0.94 | 1.00 | 0.786 | 0.835 | 0.00 |

MRR@10 by query type:

| Method | cross-week | equation | exact | filtered | notebook | paraphrase |
|---|---|---|---|---|---|---|
| dense | 0.75 | 1.00 | 0.88 | 1.00 | 0.06 | 0.32 |
| bm25 | 1.00 | 1.00 | 1.00 | 1.00 | 0.29 | 0.87 |
| tfidf | 1.00 | 1.00 | 1.00 | 1.00 | 0.15 | 0.90 |
| hybrid | 1.00 | 1.00 | 1.00 | 1.00 | 0.20 | 0.55 |

## Tune split

| Method | R@1 | R@3 | R@5 | R@10 | MRR@10 | nDCG@10 | Unanswerable with no result |
|---|---|---|---|---|---|---|---|
| dense | 0.50 | 0.60 | 0.65 | 0.80 | 0.693 | 0.673 | 0.00 |
| bm25 | 0.70 | 0.85 | 0.90 | 0.97 | 0.858 | 0.890 | 0.00 |
| tfidf | 0.60 | 0.87 | 0.87 | 0.89 | 0.799 | 0.814 | 0.33 |
| hybrid | 0.61 | 0.80 | 0.88 | 0.90 | 0.829 | 0.815 | 0.00 |

MRR@10 by query type:

| Method | cross-week | equation | exact | filtered | notebook | paraphrase |
|---|---|---|---|---|---|---|
| dense | 0.78 | 1.00 | 1.00 | 0.56 | 0.42 | 0.43 |
| bm25 | 0.75 | 1.00 | 1.00 | 1.00 | 1.00 | 0.58 |
| tfidf | 0.71 | 1.00 | 1.00 | 1.00 | 0.83 | 0.47 |
| hybrid | 1.00 | 1.00 | 1.00 | 1.00 | 0.67 | 0.52 |

## Cost

| Method | Build (s) | Index size (bytes) | Cold query (ms) | Warm median (ms) |
|---|---|---|---|---|
| dense | 0.070 | 86016 | 1.01 | 0.269 |
| bm25 | 0.002 | 40960 | 2.62 | 0.036 |
| tfidf | 0.002 | 8016 | 2.62 | 0.019 |
| hybrid | 0.072 | 126976 | 3.19 | 0.362 |

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

- Hybrid settings chosen on the tune split: RRF c = 30, depth = 10.
- Best single method on held-out: **bm25**.
- Hybrid minus best single, held-out MRR@10: -0.099 (hybrid better on 1 queries, worse on 4).
- Rule: adopt hybrid only with a semantic embedder, held-out MRR@10 gain >= 0.05, and no loss in held-out Recall@5.
- **Not decided: this run has no semantic embedder.** The BM25 and TF-IDF rows do not depend on the embedder.

Per-query misses are in `failures.jsonl`; every ranking is in `per_query.json`.
