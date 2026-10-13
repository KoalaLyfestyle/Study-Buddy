# Retrieval comparison: hashing-v1-512

> The dense column uses the hashing embedder, an offline lexical stand-in. It says nothing
> about semantic search, so dense and hybrid results here do not count toward the decision.
> Run `python -m studybuddy.evaluate` with the real model for that.

Labels are model-generated and provisional until a person checks them.

Corpus: 10 files, 42 chunks. Queries: tune {'answerable': 20, 'unanswerable': 3}, held-out {'answerable': 18, 'unanswerable': 3}.

## Held-out split

| Method | R@1 | R@3 | R@5 | R@10 | MRR@10 | nDCG@10 | Unanswerable with no result |
|---|---|---|---|---|---|---|---|
| dense | 0.47 | 0.64 | 0.75 | 0.86 | 0.651 | 0.689 | 0.00 |
| bm25 | 0.72 | 0.83 | 0.92 | 0.97 | 0.871 | 0.879 | 0.67 |
| tfidf | 0.72 | 0.83 | 0.83 | 0.92 | 0.869 | 0.859 | 1.00 |
| hybrid | 0.64 | 0.81 | 0.86 | 0.97 | 0.835 | 0.847 | 0.00 |

MRR@10 by query type:

| Method | cross-week | equation | exact | filtered | notebook | paraphrase |
|---|---|---|---|---|---|---|
| dense | 0.75 | 1.00 | 0.88 | 1.00 | 0.07 | 0.32 |
| bm25 | 1.00 | 1.00 | 1.00 | 1.00 | 0.17 | 0.87 |
| tfidf | 1.00 | 1.00 | 1.00 | 1.00 | 0.07 | 0.90 |
| hybrid | 1.00 | 1.00 | 1.00 | 1.00 | 0.14 | 0.75 |

## Tune split

| Method | R@1 | R@3 | R@5 | R@10 | MRR@10 | nDCG@10 | Unanswerable with no result |
|---|---|---|---|---|---|---|---|
| dense | 0.55 | 0.64 | 0.72 | 0.86 | 0.718 | 0.720 | 0.00 |
| bm25 | 0.65 | 0.82 | 0.89 | 0.91 | 0.814 | 0.824 | 0.00 |
| tfidf | 0.60 | 0.82 | 0.82 | 0.86 | 0.774 | 0.784 | 0.33 |
| hybrid | 0.62 | 0.79 | 0.89 | 0.91 | 0.823 | 0.811 | 0.00 |

MRR@10 by query type:

| Method | cross-week | equation | exact | filtered | notebook | paraphrase |
|---|---|---|---|---|---|---|
| dense | 0.78 | 1.00 | 1.00 | 0.56 | 0.42 | 0.53 |
| bm25 | 0.75 | 1.00 | 1.00 | 1.00 | 0.67 | 0.61 |
| tfidf | 0.71 | 1.00 | 1.00 | 1.00 | 0.67 | 0.47 |
| hybrid | 1.00 | 1.00 | 1.00 | 1.00 | 0.50 | 0.59 |

## Cost

| Method | Build (s) | Index size (bytes) | Cold query (ms) | Warm median (ms) |
|---|---|---|---|---|
| dense | 0.108 | 86016 | 0.35 | 0.303 |
| bm25 | 0.004 | 36864 | 0.1 | 0.039 |
| tfidf | 0.004 | 7168 | 0.16 | 0.02 |
| hybrid | 0.112 | 122880 | 0.74 | 0.616 |

Dense build time includes parsing and importing; BM25 and TF-IDF are built together from the imported chunks.

## Decision

- Hybrid settings chosen on the tune split: RRF c = 30, depth = 40.
- Best single method on held-out: **bm25**.
- Hybrid minus best single, held-out MRR@10: -0.036 (hybrid better on 2 queries, worse on 3).
- Rule: adopt hybrid only with a semantic embedder, held-out MRR@10 gain >= 0.05, and no loss in held-out Recall@5.
- **Adopt hybrid: no.**

Per-query misses are in `failures.jsonl`; every ranking is in `per_query.json`.
