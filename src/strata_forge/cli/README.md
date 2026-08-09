# strata_forge.cli

The Typer application behind the `strata-forge` console script, which is installed with the
package. `strata-forge doctor` is the first thing to run: it prints resolved settings, tracked
package versions, and reachability probes for Langfuse, Redis and Qdrant. The remaining command
groups expose each module's operator workflows — `chat`, `prompts`, `datasets`, `eval`,
`experiments`, `compute`, `train` and `serve` — with `--help` on every one.

The module's only public symbol is `app`, the Typer instance, so the CLI can be embedded in another
Typer program or driven from tests. Commands that reach a module behind an optional extra need that
extra installed; the base install covers the rest.

Reference:
[docs/modules/cli.md](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/modules/cli.md).
