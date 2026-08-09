# Notebooks

[Marimo](https://marimo.io/) notebooks for interactive exploration. Marimo stores notebook state in
source-friendly Python files, so diffs and reviews work like ordinary code.

Run a notebook with `uv run marimo edit notebooks/<name>.py`.

## Templates

| Notebook | What it shows |
|---|---|
| [`00_chat_starter.py`](00_chat_starter.py) | Single `LLMClient` call with a dropdown for the model, a text area for the prompt, and a temperature slider; prints route / tokens / cost / latency. |
| [`01_eval_iterate.py`](01_eval_iterate.py) | Tiny in-process dataset, multi-model picker, runs `run_experiment` and renders the outcomes as a Marimo table. |
| [`02_rag_prototype.py`](02_rag_prototype.py) | Inline documents, recursive chunker + in-memory vector store + dense retriever; type a query, see ranked results. |

All three are starting points — replace the inline data with your own and swap pieces (different
model, real `DatasetStore`, Qdrant store, hybrid retrieval, reranker) as you iterate.

## Getting marimo

Marimo is in this repo's `dev` dependency group, so a plain `uv sync` at the repo root installs it
alongside the test and lint tooling — there is nothing extra to add. If you are working outside a
clone of this repo, `pip install marimo` (or `uv pip install marimo`) is enough; the notebooks only
need marimo plus `strata-forge` itself.

Every notebook calls a provider, so the relevant credential (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
…) has to be in the environment marimo starts in. None of these files load a `.env` themselves the
way `examples/_common.py` does, so exporting the variable in the shell you launch from is the
reliable route.

## If you have never used marimo

`uv run marimo edit notebooks/00_chat_starter.py` starts a local server and opens the editor in
your browser; Ctrl-C in the terminal stops it. `uv run marimo run notebooks/00_chat_starter.py`
serves the same file as a read-only app with the code hidden, which is the shape to reach for when
showing a result to someone else.

Four things differ from a Jupyter notebook and explain how these files are written:

- Each cell is a function of the variables it reads. Marimo works out the dependency graph itself,
  so editing a cell re-runs everything downstream and nothing else. Execution order follows the
  graph, not the position on screen.
- There is no stale state. Deleting a cell deletes its variables, and a variable can only be
  defined by one cell — which is why the cells here return their values as tuples and take their
  inputs as arguments.
- Cells that `await` are ordinary `async def` cells; marimo runs them on its own event loop, so no
  `asyncio.run` appears in these files.
- The file on disk is real Python. `ruff` lints it, and `tests/e2e/test_notebook_imports.py` imports
  every numbered notebook in this directory to catch templates drifting away from the library API.
  Keep new notebooks importable without side effects, and prefix them with a number so the harness
  picks them up.
