## Problem
<!-- What issue does this PR address? Link the relevant issue. -->

## Solution
<!-- What change does this PR introduce? Summarize the approach. -->

## Testing
<!-- How was this change tested? List the exact commands and any manual verification. -->

Contributor quick path:

```sh
make setup PROFILE=core
make check-changed BASE=origin/main
```

If this change crosses a named boundary, add the explicitly relevant specialist
lane(s) and list them below. Specialist lanes are troubleshooting/boundary
work, not a routine gate; CI owns the full Lean/provider, coverage,
compatibility, packaging, security, and exhaustive-matrix surface. See
[CONTRIBUTING.md](../CONTRIBUTING.md) and the
[testing strategy](../docs/reference/testing-strategy.md) for lane ownership.

- Specialist validation run (if any): <!-- e.g. make test-lean TESTS=..., make harbor-validate-task DATASET=... TASKS="..." -->

## Trust & Compatibility Impact
<!-- Does this change affect the verification kernel, checker registry, artifact format, or public API? -->

## Checklist
- [ ] `make check-changed BASE=origin/main` passes
- [ ] Explicitly relevant specialist validation is listed above (boundary, Lean, provider, Harbor/Oracle)
- [ ] Harbor task or verifier changes ran `make harbor-prepare-task` then `make harbor-validate-task` (if applicable)
