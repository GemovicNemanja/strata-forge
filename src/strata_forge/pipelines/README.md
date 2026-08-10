# strata_forge.pipelines

Runnable entrypoints for machines you provision. Every other `strata_forge` module is a set of
primitives you import and call; a pipeline is a program you *launch* on a compute target. It takes
no command-line arguments, reads an inert JSON run spec from an environment variable, composes the
library's serving, batch-inference and storage primitives into one job, and appends structured
progress records to a file so whatever started it can follow along and know how it ended.

That shape exists because a remote job needs a contract. Submitting a shell command to a GPU host
is easy; knowing what it is doing while it runs, and what it produced when it stops, is not. A
pipeline module fixes all four sides of that: a stable entrypoint, a validated spec, a defined
progress format, and defined exit codes — so the thing driving the job never has to scrape log
output to find out what happened.

`inference_runner` is the entrypoint that ships today. Point it at a Hugging Face dataset split
and a model; it renders one prompt per row, serves the model with vLLM on loopback, runs the
prompts through a concurrency-bounded batch, and writes the results as a parquet file — pushed to
a private Hugging Face dataset repo when you supply a write token and a destination, and kept on
the machine for you to collect otherwise.

```bash
export STRATA_RUN_CONFIG='{"model_id": "Qwen/Qwen2.5-7B-Instruct",
                           "dataset_id": "owner/prompts", "split": "train",
                           "template": "Summarize: {doc}",
                           "column_mapping": {"doc": "text"}}'
export FORGE_PROGRESS_PATH=progress.jsonl
python -m strata_forge.pipelines.inference_runner
```

The spec is data, never code: it is parsed as JSON, validated against a Pydantic model that
rejects unknown fields, and rendered into prompts by a substitution that executes nothing. A write
token travels in its own environment variable rather than in the spec, is passed explicitly to the
Hub client, and is scrubbed from anything the runner reports.

Importing this module is always safe — `datasets` and `huggingface_hub` are imported only inside
the functions that use them, so you can build and validate a run spec anywhere. Running a job
needs the `[hf]` and `[storage]` extras, and vLLM already installed on the target machine.

Full reference:
[`docs/modules/pipelines.md`](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/modules/pipelines.md).
