## Summary

<!-- What changed, in a few sentences. Describe the end state, not the journey. -->

## Why

<!-- The problem this solves, or the behaviour that was wrong. Link the issue if there is one
     (`Closes #123`). If the change is architectural, link the ADR that motivates it. -->

## Test plan

<!-- How you know this works: the commands you ran, the cases you covered, what a reviewer should
     re-run. "CI is green" is not a test plan. -->

---

### Checklist

- [ ] PR title follows [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) —
      `<type>(<scope>): <imperative summary>`, lowercase, under 70 characters.
- [ ] Title, commits, and branch name use time-stable language — no phase numbers, step numbers, or
      plan-file slugs.
- [ ] Branched off `dev`. (`main` only ever receives `dev`, via a promotion PR.)
- [ ] `make check` passes — ruff lint plus pyright strict on `src/strata_forge`.
- [ ] `make test` passes, and every new code path has a test in this PR.
- [ ] `docs/modules/<name>.md` updated in this PR if the public API changed; `README.md` too if the
      change is user-facing.
- [ ] The module's `src/strata_forge/<module>/README.md` still describes the module accurately (it
      ships inside the wheel).
- [ ] `CHANGELOG.md` has an entry under `## [Unreleased]` if the change is user-visible.
- [ ] An ADR under `docs/architecture/adr/` accompanies any architecturally significant change, or
      the existing one it follows is linked above.
- [ ] Rule files (`CLAUDE.md`, `src/strata_forge/<module>/CLAUDE.md`, `.cursor/rules/*.mdc`) updated
      if this change makes any of them stale.
- [ ] No secrets, live credentials, or unscrubbed VCR cassettes are included in the diff.

House conventions are in
[CONTRIBUTING.md](https://github.com/GemovicNemanja/strata-forge/blob/main/CONTRIBUTING.md) (the
machine-readable version agents load is
[AGENTS.md](https://github.com/GemovicNemanja/strata-forge/blob/main/CLAUDE.md)). Never report a
vulnerability in a pull request — see
[SECURITY.md](https://github.com/GemovicNemanja/strata-forge/blob/main/SECURITY.md).
