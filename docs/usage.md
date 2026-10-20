# Setup and use

The chat UI (Streamlit) runs a LangChain ReAct agent on a local Ollama model. The agent calls the search tool and is told to cite sources. Its `Quiz_Generator` and `Summarizer` tools are placeholders: they return a fixed template and the first 200 characters of the input, and do not use the model or your notes yet.

## Install with the real model and the chat

Requires Python 3.12 (the lock file targets it) and [Ollama](https://ollama.com) for the chat.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock     # exact versions; requirements.txt lists the direct ones
ollama pull mistral
```

The lock covers macOS, Linux, and Windows; the CUDA packages install on
Linux only. After changing `requirements.txt`, regenerate it with
`uv pip compile requirements.txt --universal --python-version 3.12 -o requirements.lock`.

## Import and search your own courses

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
