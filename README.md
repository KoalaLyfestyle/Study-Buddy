# Study Buddy

A local study assistant. It imports your course files, searches them, and
answers with the source of each passage: the PDF page, the Markdown
heading, or the notebook cell.

## What works today

- **Import** a folder of PDF, Markdown, and Jupyter notebook files.
  Notebooks are read as JSON. Their cells are never run and their outputs
  are ignored.
- **Incremental updates.** Importing again skips unchanged files. A changed
  file is re-read and re-embedded on its own. A deleted file drops out of
  search.
- **Search with citations.** Every result carries the file path, course,
  week, location (page, heading path, or cell ID), the character offsets
  of the chunk inside that page, section, or cell, and the hash of the
  file it came from.
- **Study rooms.** `python -m studybuddy room <course> <week>` turns one
  imported week into a single HTML page: setup, symbols, the source
  passages, worked examples, common mistakes, a summary, and a quiz, each
  linked to the passage it comes from. See [Study rooms](#study-rooms).
- **Chat UI** (Streamlit) with a LangChain ReAct agent on a local Ollama
  model. The agent calls the search tool and is told to cite sources.

Placeholders: the agent's `Quiz_Generator` and `Summarizer` tools return a
fixed template and the first 200 characters of the input. They do not use
the model or your notes yet.

## How retrieval works

1. `studybuddy/sources.py` finds supported files and splits them into
   sections: one per PDF page, one per Markdown heading, one per notebook
   cell. The course is the top-level folder; the week comes from a
   `week-NN` folder name. A notebook code cell also gets index-only
   context: the notebook title and the Markdown cell above it. Keyword
   search (BM25 and TF-IDF) uses it, since a code cell rarely says in words
   what it does. Embeddings, citations, and chunk IDs use the cell alone.
2. `studybuddy/chunking.py` cuts sections into chunks of up to 1,200
   characters at paragraph breaks. Each chunk ID is a hash of the file
   path, location (without line numbers), position, and text, so it stays
   the same across runs and when other sections of the file change.
3. `studybuddy/embeddings.py` embeds chunks with `BAAI/bge-small-en`
   through sentence-transformers, run locally, at a recorded model commit.
4. `studybuddy/library.py` stores chunks and vectors in SQLite at
   `.studybuddy/library.sqlite`. Each file has a cache key made from its
   SHA-256 and the parser, chunker, and embedder versions. If any of them
   changes, that file is rebuilt.
5. Search ranks chunks by cosine similarity after applying course and week
   filters.

If a file fails to parse, the import reports it and keeps that file's last
good records. Search still returns them, labelled "out of date". If the
import fails as a whole, for example because the embedding model cannot
load, nothing is written and the previous index stays. If the folder has
no supported files at all, the import stops with advice and leaves the
index as it was, since that is usually a wrong path or an unmounted drive.
A folder inside the course that cannot be read is reported as a failure,
and its files keep their last good records instead of being removed.
Links to folders are not followed, and a link to a file outside the course
folder is skipped and reported, so it cannot pull other files into search.
A renamed file is treated as a deleted file plus a new one. The importer
never writes to your course folder, so your own `notes.md` files are never
replaced.

## Setup

Requires Python 3.12 (the lock file targets it) and [Ollama](https://ollama.com) for the chat.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock     # exact versions; requirements.txt lists the direct ones
ollama pull mistral
```

The lock covers macOS, Linux, and Windows; the CUDA packages install on
Linux only. After changing `requirements.txt`, regenerate it with
`uv pip compile requirements.txt --universal --python-version 3.12 -o requirements.lock`.

## Use

Put your files in `data/`, one folder per course:

```
data/
  linear-algebra/
    week-01/lecture.pdf
    week-01/notes.md
    week-02/lab.ipynb
```

Import and search from the command line:

```bash
python -m studybuddy import data
python -m studybuddy search "eigenvalues of a symmetric matrix" --week 1
```

Or start the chat app. Before each search it refreshes the index from
`data/`, so files you add while it runs are picked up. If the refresh
fails (for example the folder is missing), it searches the last good index
and says so:

```bash
streamlit run app.py
```

The first import downloads the embedding model once. An index is tied
to the embedder that built it; if you switch, import again. To try everything
offline, set `STUDYBUDDY_EMBEDDER=hashing`. That uses a lexical stand-in
that needs no download; results are worse.

A small original sample course lives in `sample_course/`:

```bash
python -m studybuddy import sample_course
python -m studybuddy search "why does convexity matter"
```

## Use it from Claude or another MCP client

`python -m studybuddy.mcp_server` serves the library over MCP (stdio). It has four tools:

| Tool | Returns |
|---|---|
| `list_courses` | Courses with their weeks and file counts, and files that failed to import |
| `search_materials` | Up to 10 passages: chunk ID, score, a 300-character snippet, and a citation (path, page or cell, label). Filters: `course`, `week`, `kind` (`pdf`, `markdown`, `notebook`) |
| `read_source` | The full text of one passage by chunk ID, 4,000 characters per call |
| `get_week_context` | Every passage in one course week, in reading order, with citations and short snippets |

The tools only read and search; none calls a generation model. `search_materials` uses the
local embedder and refreshes the index from `STUDYBUDDY_DATA_DIR` first; if that refresh fails,
it answers from the last good index and adds a `warning`. The other tools read the index as it
is (they import only when it is empty), so run a search or `python -m studybuddy import` after
adding files.

Example for Claude Desktop (`claude_desktop_config.json`); use absolute paths:

```json
{
  "mcpServers": {
    "study-buddy": {
      "command": "/path/to/Study-Buddy/.venv/bin/python",
      "args": ["-m", "studybuddy.mcp_server"],
      "env": {
        "PYTHONPATH": "/path/to/Study-Buddy",
        "STUDYBUDDY_DATA_DIR": "/path/to/your/courses",
        "STUDYBUDDY_INDEX_DIR": "/path/to/Study-Buddy/.studybuddy"
      }
    }
  }
}
```

All three front ends share `studybuddy/api.py`: the MCP tools return its JSON, the CLI prints it
(`python -m studybuddy courses`, `read <chunk_id>`, `week <course> <week>`, and `search`), and the
agent's `Search_Notes` tool formats its search results for the chat.

## Comparing retrieval methods

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

## Study rooms

```bash
python -m studybuddy import sample_course
python -m studybuddy room intro-to-optimisation 1
# Open .studybuddy/rooms/intro-to-optimisation/week-01/index.html
```

- **Format.** A room is JSON checked against `studybuddy/room/schema.json`:
  chapters (one per file) made of sections. Each section has a kind
  (setup, symbols, explanation, worked example, common mistakes, quiz, or
  summary), the chunk IDs it cites, and a provenance: `source` (quoted
  from the cited passages) or `generated` (written by a model and only
  supported by them). `coverage` maps every passage of the week to the
  sections that use it, so you can see what a room leaves out.
- **Builder.** The built-in builder (`extractive-v1`) uses no model. It
  only quotes your files, so every section is `source`. Its quiz asks
  which part of a file says a quoted sentence.
- **Model-written sections.** `python -m studybuddy room <course> <week> --generate`
  (needs `ollama serve` and `ollama pull mistral`; pick another with
  `--model`) sends each chapter's passages to the local model and adds
  the setup, worked example, common mistakes, and quiz sections it writes.
  They are marked "Generated" and may cite only that chapter's passages.
  A reply that fails the checks is retried once with the problems listed;
  if it still fails, that chapter keeps only its quoted sections and the
  command says why. Generated section IDs come from the chapter and the
  section's kind and order, so progress and edits survive a rebuild.
- **Your own room JSON**, for example from a local model:
  `python -m studybuddy room <course> <week> --from-json room.json`. It is
  checked against the schema and the library (unknown citations,
  duplicate IDs, quiz answers out of range, passages missing from
  `coverage`). If it fails, the command lists each problem with its JSON
  path, exits with code 2, and keeps the last good room.
- **Stable IDs, progress, and edits.** Section IDs come from where a
  passage lives (file, page, heading path, or notebook cell), not from its
  text, so rebuilding after you edit your notes keeps them. A repeated
  heading or cell ID is numbered by order. A notebook cell without an ID
  (older notebooks) is placed by the nearest heading above it, so adding a
  cell under another heading does not move it; adding one under the same
  heading does. Progress lives
  in `progress.json` and your own text in `edits.json`, next to the room;
  a rebuild never rewrites them. The page saves ticks and quiz answers in
  the browser, and "Export progress" downloads them as `progress.json`.
  `edits.json` maps a section ID to a new `title` or `blocks`.
- **Safety.** All course text is escaped. Notebook code is shown as text
  and never run.
- **Screenshots** of the sample course's week 1 room:
  [phone](docs/images/room-phone.png), [desktop](docs/images/room-desktop.png).
- **Browser check** (needs Playwright with Chromium):
  `python scripts/check_room_page.py <room>/index.html` checks phone and
  desktop widths, keyboard access, that progress survives a reload and
  exports, and that citations open their section. `--shots DIR` saves
  screenshots; use it on the public sample course only.

### Notebook code cells (parser version 2)

Code cells get index-only context for keyword search (see step 1 of "How
retrieval works"). Dense vectors use the cell text alone: an earlier
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

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `STUDYBUDDY_DATA_DIR` | `./data` | Course folder the app imports |
| `STUDYBUDDY_INDEX_DIR` | `./.studybuddy` | Where the index is stored |
| `STUDYBUDDY_EMBEDDER` | `BAAI/bge-small-en` | A sentence-transformers model name, optionally `name@<commit sha>`, or `hashing` |

The model's commit is part of each file's cache key, so new weights
re-embed the library rather than mixing old and new vectors. The default
model is pinned to commit `2275a7bd` by `DEFAULT_REVISION` in
`studybuddy/embeddings.py`. Another model without `@<sha>` uses the commit
already in the local Hugging Face cache (or, before the first download,
the one the Hub names); `python -m studybuddy.embeddings` prints the
cached commit of the default model. After
upgrading from a version without this, run `import` once: the new model ID
re-embeds the library.

## Tests

The tests use the sample course and the offline embedder. They need no
model download and no Ollama.

```bash
pip install -r requirements-dev.txt
python -m pytest
```

## Layout

- `studybuddy/` - import, chunking, embeddings, library, retrieval methods, evaluation, the CLI, and the MCP server (`api.py` holds the functions the CLI, MCP server, and agent call)
- `studybuddy/room/` - study rooms: the schema, the extractive builder, checks, progress and edits, and the HTML renderer
- `scripts/check_room_page.py` - browser check for a built study room
- `rag.py` - opens the library for the agent and the app
- `agent/` - LangChain ReAct agent and its tools
- `app.py` - Streamlit UI
- `config.py` - local Ollama model settings
- `sample_course/` - original sample material for tests and demos
- `scripts/make_sample_pdf.py` - regenerates the sample PDF
- `eval/` - retrieval comparison: queries, extra course weeks, and results
- `scripts/make_eval_pdf.py` - regenerates the evaluation PDF

## Acknowledgements

LangChain, Streamlit, Ollama, sentence-transformers, pypdf, and the
BAAI bge embedding models.
