# Examples

Runnable example scripts that double as end-to-end smoke tests for the documented public API.

Each script is self-contained, prints structured output, is environment-gated (skips with a clear message when required keys are missing), and is referenced from the relevant module's `docs/modules/<name>.md` guide.

Run an example directly with `uv run python examples/<name>.py [--help]`.
