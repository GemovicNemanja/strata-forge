# strata_forge.core

Cross-cutting utilities every other module depends on: the exception hierarchy (`ForgeError` and
its subclasses, including the normalized `ProviderError` family), a tenacity-backed `@retry`
decorator, structlog configuration that injects a `correlation_id` contextvar into every record,
`BudgetContext` for cost and token ceilings, reproducibility helpers (`set_seed`, `content_hash`,
`env_snapshot`), a hand-rolled RFC 9562 `uuid7`, and shared type aliases.

Strictly upstream: this module never imports from another `strata_forge.*` module, and it needs no
optional extra.

Reference:
[docs/modules/core.md](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/modules/core.md).
