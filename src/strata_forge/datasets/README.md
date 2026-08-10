# strata_forge.datasets

Typed dataset shapes plus pluggable storage and synthetic-data primitives. The frozen Pydantic
models `DatasetItem` and `Dataset` are the canonical in-memory shape every other piece converts
into and out of; `dataset_version` derives a content hash so runs can name exactly what they
evaluated, and `diff` reports what changed between two versions. See
[ADR 0009](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/architecture/adr/0009-datasets-langfuse-canonical-hf-exchange.md)
for why Langfuse is the persistent store and Hugging Face Datasets the exchange format.

`DatasetStore` is the async storage interface, with `InMemoryDatasetStore` and
`LangfuseDatasetStore` (`[langfuse]` extra) as backends. `to_hf_dataset` / `from_hf_dataset` bridge
to Hugging Face Datasets (`[hf]` extra), and `self_instruct` / `distill` generate datasets from an
`LLMClient`.

Reference:
[docs/modules/datasets.md](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/modules/datasets.md).
