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
work, not a routine gate; CI owns coverage, compatibility, packaging, and the
ordinary Python surface. See
[CONTRIBUTING.md](../CONTRIBUTING.md) and the
[testing strategy](../docs/reference/testing-strategy.md) for lane ownership.

- Specialist validation run (if any): <!-- e.g. make test-mcp TESTS=..., make harbor-validate-task DATASET=... TASKS="..." -->

## Public contract impact
<!-- Does this change alter an operation ID, request/result schema, native API,
MCP contract, or mathematical semantics? State "none" when it does not. -->

## Operation boundary ownership
<!-- Complete for any operation, request/result model, backend adapter,
native API, dispatch, or MCP change. Use "not applicable" when this PR does
not touch an operation boundary. -->
- Changed stage: <!-- request parsing / request bounds / kernel adapter / result construction / transport projection / other -->
- Request-bounds owner and controlling quantities:
- Backend or kernel path:
- Result construction: <!-- canonical conversion; malformed-backend handling belongs to the adapter -->
- Independent-result verifier (only if the public contract accepts independently supplied result data): <!-- none, or state its explicit replay bound -->
- Native/MCP parity: <!-- same semantic admission/results, with transport-only differences stated -->
- Serialized-result and round-trip evidence:

## Canonical value audit
<!-- Complete when this change adds or changes a mathematical value, request,
result, producer, or consumer. -->
- [ ] Existing canonical values searched
- [ ] Owner or intentional distinction documented
- [ ] Producer→consumer serialization tested

## Catalog admission
<!-- Complete only when adding, removing, or materially changing catalog
membership. This is publication admission, not per-request runtime bounds. -->

- Concrete gap:
- Why existing operations or typed values are insufficient:
- Stable mathematical result:
- Semantic domain and admitted execution envelope:
- Controlling work, intermediate, memory, and output quantities:
- Exact representation, algorithm regimes, and maintained backend:
- Remaining fixed caps and their classification:
- Motivating source request exercised through the public boundary:
- Final-tree outcome for that request:
- Effect of private normalization or presolve on admission:
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
- [ ] Explicitly relevant specialist validation is listed above (boundary, backend, Harbor/Oracle)
- [ ] Harbor task or verifier changes ran `make harbor-prepare-task` then `make harbor-validate-task` (if applicable)
- [ ] Catalog changes include an owner-local catalog-admission decision (if applicable)
- [ ] Runtime-bound changes name the request-bounds owner and execute the same semantic path for native and MCP callers (if applicable)
- [ ] Result semantics distinguish exact, approximate, incomplete, unknown, and unavailable outcomes where applicable
- [ ] Public operation changes include a behavioral regression copied from a motivating parent-gap request, or explain why no source request exists (if applicable)
- [ ] New or changed bounds include boundary, algorithm-crossover, and realistic source-backed scale evidence (if applicable)
- [ ] New shared abstractions replace duplication in at least two surviving production paths (if applicable)
