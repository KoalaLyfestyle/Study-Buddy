# Comparing retrieval methods

`python -m studybuddy.evaluate` compares four ways to search the same chunks:
dense (the embedding model), BM25 (SQLite FTS5), TF-IDF, and hybrid
(reciprocal rank fusion of dense and BM25). Course, week, and file-type
filters apply before ranking in all four.

```bash
python -m studybuddy.evaluate                       # real model
python -m studybuddy.evaluate --embedder hashing    # offline; dense is then not semantic
```

- Corpus: `sample_course/` plus the extra weeks in `eval/extra/` (original material).
- Queries: `eval/queries.jsonl`, 44 queries of seven types, split into `tune` and `heldout`.
  The relevance labels were written by a model and are provisional until a person checks them.
- Output: `eval/results/<embedder>/report.md`, `summary.json`, `per_query.json`, and
  `failures.jsonl` (each miss in the top 5, and each unanswerable query that got results).
  Build and query timings go to `timings.json` in the same folder, which git ignores, so
  rerunning the comparison leaves the committed results unchanged.
- Hybrid's settings are tuned on `tune` only. It is adopted only if, with a semantic
  embedder, it beats the best single method on held-out MRR@10 by at least 0.05 without
  losing Recall@5. "BM25 is enough" is a valid outcome.

## Notebook code cells (parser version 2)

Code cells get index-only context for keyword search (see step 1 of [How retrieval works](how-it-works.md)). Dense vectors use the cell text alone: an earlier
version of this change also embedded the context, and with the real model
that lowered held-out dense MRR@10 from 0.782 to 0.708 (notebook queries
0.67 to 0.50, equation queries 0.50 to 0.22).

With the real model (`BAAI/bge-small-en` at the pinned commit), checked on
a Mac: dense is unchanged from before (held-out MRR@10 0.782, notebook
0.67), BM25's held-out notebook MRR@10 rises to 0.29, and hybrid's
held-out MRR@10 is 0.889.

Offline comparison (`--embedder hashing`), held-out MRR@10 on notebook
queries:

| Method | Before | After |
|---|---|---|
| BM25 | 0.17 | 0.29 |
| Dense (hashing stand-in) | 0.06 | 0.06 |
| TF-IDF | 0.07 | 0.15 |
| Hybrid | 0.14 | 0.20 |

Offline, no other held-out query type dropped for BM25 or TF-IDF, and dense
is unchanged. Hybrid with the hashing stand-in got worse overall on
held-out (0.835 to 0.786), mostly paraphrase (0.75 to 0.55), so judge
hybrid by the real-model numbers above. On the tune split, paraphrase fell
for BM25 (0.61 to 0.58) and hybrid (0.59 to 0.52). Held-out has only two
notebook queries, so treat the notebook gain as a direction, not a size.

The parser version change makes the next import re-read and re-embed
every file once, not only notebooks. With the real model that takes
longer than a normal refresh.
