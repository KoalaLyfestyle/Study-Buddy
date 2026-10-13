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
- **Chat UI** (Streamlit) with a LangChain ReAct agent on a local Ollama
  model. The agent calls the search tool and is told to cite sources.

Placeholders: the agent's `Quiz_Generator` and `Summarizer` tools return a
fixed template and the first 200 characters of the input. They do not use
the model or your notes yet.

## How retrieval works

1. `studybuddy/sources.py` finds supported files and splits them into
   sections: one per PDF page, one per Markdown heading, one per notebook
   cell. The course is the top-level folder; the week comes from a
   `week-NN` folder name.
2. `studybuddy/chunking.py` cuts sections into chunks of up to 1,200
   characters at paragraph breaks. Each chunk ID is a hash of the file
   path, location (without line numbers), position, and text, so it stays
   the same across runs and when other sections of the file change.
3. `studybuddy/embeddings.py` embeds chunks with `BAAI/bge-small-en`
   through sentence-transformers, run locally.
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
- Hybrid's settings are tuned on `tune` only. It is adopted only if, with a semantic
  embedder, it beats the best single method on held-out MRR@10 by at least 0.05 without
  losing Recall@5. "BM25 is enough" is a valid outcome.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `STUDYBUDDY_DATA_DIR` | `./data` | Course folder the app imports |
| `STUDYBUDDY_INDEX_DIR` | `./.studybuddy` | Where the index is stored |
| `STUDYBUDDY_EMBEDDER` | `BAAI/bge-small-en` | A sentence-transformers model name, or `hashing` |

## Tests

The tests use the sample course and the offline embedder. They need no
model download and no Ollama.

```bash
pip install -r requirements-dev.txt
python -m pytest
```

## Layout

- `studybuddy/` - import, chunking, embeddings, library, retrieval methods, evaluation, the CLI, and the MCP server (`api.py` holds the functions the CLI, MCP server, and agent call)
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
