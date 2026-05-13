# forge.core

Cross-cutting utilities every other module depends on: exception hierarchy (`ForgeError` and subclasses), `@retry` decorator, structlog setup with `trace_id` correlation via contextvars, `BudgetContext` for cost/token ceilings, reproducibility helpers (seeds, content hashing, environment snapshots), UUIDv7-based correlation IDs, and shared type aliases. Strictly upstream — never imports from other `forge.*` modules.

> Implementation pending. See `docs/roadmap.md` for current status.
