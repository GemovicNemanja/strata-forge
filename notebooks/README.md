# Notebooks

[Marimo](https://marimo.io/) notebooks for interactive exploration. Marimo stores notebook state in source-friendly Python files, so diffs and reviews work like ordinary code.

Run a notebook with `uv run marimo edit notebooks/<name>.py`.

## Templates

| Notebook | What it shows |
|---|---|
| [`00_chat_starter.py`](00_chat_starter.py) | Single `LLMClient` call with a dropdown for the model, a text area for the prompt, and a temperature slider; prints route / tokens / cost / latency. |
| [`01_eval_iterate.py`](01_eval_iterate.py) | Tiny in-process dataset, multi-model picker, runs `run_experiment` and renders the outcomes as a Marimo table. |
| [`02_rag_prototype.py`](02_rag_prototype.py) | Inline documents, recursive chunker + in-memory vector store + dense retriever; type a query, see ranked results. |

All three are starting points — replace the inline data with your own and swap pieces (different model, real `DatasetStore`, Qdrant store, hybrid retrieval, reranker) as you iterate.
