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

## Canonical value audit
<!-- Complete when this change adds or changes a mathematical value, request,
result, producer, or consumer. -->
- [ ] Existing canonical values searched
- [ ] Owner or intentional distinction documented
- [ ] Producer→consumer serialization tested

## Public operation admission
<!-- Complete only when adding or materially changing a public operation. -->

- Concrete gap:
- Why existing operations or typed values are insufficient:
- Stable mathematical result:
- Semantic domain and admitted execution envelope:
- Controlling work, intermediate, memory, and output quantities:
- Exact representation, algorithm regimes, and maintained backend:
- Remaining fixed caps and their classification:
- Admission decision:

## Closure matrix
<!-- Complete when this PR closes a broad parent issue. List each operation the
parent proposed and its outcome, so deferred or rejected candidates stay
visible instead of disappearing with the parent. Leave "none" for a single
admitted operation with no residual surface. -->

| Candidate | Outcome | Operation ID or follow-up issue |
| --- | --- | --- |
| <!-- e.g. graphicality with obstruction --> | <!-- delivered / deferred / rejected --> | <!-- operation ID or child issue --> |

## Checklist
- [ ] `make check` passes
- [ ] Explicitly relevant specialist validation is listed above (boundary, Lean, backend, Harbor/Oracle)
- [ ] Harbor task or verifier changes ran `make harbor-prepare-task` then `make harbor-validate-task` (if applicable)
- [ ] Public operation changes include an owner-local admission decision (if applicable)
- [ ] Result semantics distinguish exact, approximate, incomplete, unknown, and unavailable outcomes where applicable
- [ ] New or changed bounds include boundary, algorithm-crossover, and realistic source-backed scale evidence (if applicable)
- [ ] New shared abstractions replace duplication in at least two surviving production paths (if applicable)
