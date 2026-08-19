## Problem
<!-- What issue does this PR address? Link the relevant issue. -->

## Solution
<!-- What change does this PR introduce? Summarize the approach. -->

## Testing
<!-- How was this change tested? List the exact commands and any manual verification. -->

Contributor quick path:

```sh
make setup
make check
```

If this change crosses a named boundary, add the explicitly relevant specialist
lane(s) and list them below. Specialist lanes are troubleshooting/boundary
work, not a routine gate; CI owns Lean on merge-group/main, coverage,
compatibility, packaging, and the ordinary Python surface. See
[CONTRIBUTING.md](../CONTRIBUTING.md) and the
[testing strategy](../docs/reference/testing-strategy.md) for lane ownership.

- Specialist validation run (if any): <!-- e.g. make test-lean TESTS=..., make harbor-validate-task DATASET=... TASKS="..." -->

## Public contract impact
<!-- Does this change alter an operation ID, request/result schema, native API,
MCP contract, or mathematical semantics? State "none" when it does not. -->

## Public operation admission
<!-- Complete only when adding or materially changing a public operation. -->

- Concrete gap:
- Why existing operations or typed values are insufficient:
- Stable mathematical result:
- Admission decision:

## Checklist
- [ ] `make check` passes
- [ ] Explicitly relevant specialist validation is listed above (boundary, Lean, backend, Harbor/Oracle)
- [ ] Harbor task or verifier changes ran `make harbor-prepare-task` then `make harbor-validate-task` (if applicable)
- [ ] Public operation changes include an owner-local admission decision (if applicable)
- [ ] Result semantics distinguish exact, approximate, incomplete, unknown, and unavailable outcomes where applicable
- [ ] New shared abstractions replace duplication in at least two surviving production paths (if applicable)
