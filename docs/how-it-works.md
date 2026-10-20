# How retrieval works

Notebooks are read as JSON: their cells are never run and their outputs are ignored. Importing again skips unchanged files, re-reads and re-embeds a changed file on its own, and drops a deleted file from search.

Every search result carries the file path, course, week, location (page, heading path, or cell ID), the character offsets of the chunk inside that page, section, or cell, and the hash of the file it came from.

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

## When things fail

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
