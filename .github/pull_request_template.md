## Problem
<!-- What observable problem does this PR address? Link the relevant issue or
explain why no issue exists. -->

## Outcome
<!-- What behavior or contract is different after this change? State what is
intentionally unchanged. -->

## Validation
<!-- List exact commands, relevant CI checks or links, and any manual
verification. Name the final head SHA that the evidence covers. -->

- Commands run:
- Final head SHA tested:
- Relevant CI checks or links:

Contributor quick path:

```sh
make setup
make handoff LANE=math TESTS=tests/math/graphs/test_graph_distance_matrix.py
```

If this change crosses a named boundary, list the explicitly relevant specialist
lane(s) in the validation section. Specialist lanes are troubleshooting or
boundary work, not a routine gate; CI owns coverage, compatibility, packaging,
and the ordinary Python surface. See [CONTRIBUTING.md](../CONTRIBUTING.md) and
the [testing strategy](../docs/reference/testing-strategy.md) for lane ownership.

## Contract impact
<!-- State "none" when this PR changes no operation ID, request/result schema,
native API, MCP contract, or mathematical semantics. Check only applicable
items; write N/A with a reason for the rest. -->

- Public contract impact:
- [ ] Schema and runtime accept/reject the same requests
- [ ] Cheap malformed or over-budget input is rejected before expensive work
- [ ] Exact results fit the complete canonical output boundary
- [ ] All mandatory phases share one request deadline
- [ ] Native and MCP behavior agree, where both are public
- [ ] Producer → serialization → consumer composition was tested
- [ ] Defining invariant and adversarial regression are covered

## Review closure
<!-- Complete or update this section when responding to review. For a new PR,
the review-thread item is not applicable until review begins. -->

- [ ] Every substantive review thread has a root-cause fix, regression proof,
      and reply
- [ ] Merge conflicts were resolved against the intended base
- [ ] Validation was rerun after the final commit or autofix
- [ ] CI evidence matches the final head SHA
- [ ] The appropriate repository validation target and any specialist lane are
      listed above (normally `make handoff ...`; docs use `make docs-linkcheck`)
- [ ] Repository conventions were checked: no compatibility or shadow paths
      were added; if a shared abstraction was introduced, two existing paths
      already share its mechanics and contract

## Conditional detail
<!-- Complete only the subsections relevant to this change. The compact
sections above are the required summary; these fields preserve deeper evidence
for operation, catalog, backend, and parent-issue changes. -->

### Operation boundary
<!-- Complete for any operation, request/result model, backend adapter,
native API, dispatch, or MCP change. Use "not applicable" when this PR does
not touch an operation boundary. -->
- Changed stage and owner: <!-- request parsing / request bounds / kernel adapter / result construction / transport projection / other -->
- Semantic admission and controlling quantities:
- Execution envelope: <!-- work, intermediate, memory, exact output, deadline, cleanup/reaping grace, and caller-selectable range; state actual enforcing limits, not an unexplained safety reserve -->
- Backend or kernel path:
- Result construction: <!-- canonical conversion; malformed-backend handling belongs to the adapter -->
- Independent-result verifier (only if the public contract accepts independently supplied result data): <!-- none, or state its explicit replay bound -->
- Native/MCP parity: <!-- same semantic admission/results, with transport-only differences stated -->
- Serialized-result and round-trip evidence:

### Canonical value audit
<!-- Complete when this change adds or changes a mathematical value, request,
result, producer, or consumer. -->
- [ ] Existing canonical value searched; owner or intentional distinction recorded
- [ ] Advertised codomain is closed under the result types, including required
      parent, embedding, branch, orientation, basis, or coordinate data
- [ ] Producer→consumer serialization tested
- [ ] Theorem-dependent preconditions and validated input subtype are covered, or marked not applicable with a reason
- Structurally valid but mathematically invalid fixture and outcome: <!-- rejection or typed non-applicability -->
- Serialized-subtype trust boundary: <!-- constructor validation / source-bound bounded verifier / consumer recognition, plus forged-payload evidence -->
- Exact-success defining invariant: <!-- reconstruction equation, preservation law, optimum/certificate relation, or not applicable -->
- Discriminated result schema and impossible combinations: <!-- public branch schema plus contradictory status/diagnostic/witness rejection -->

### Catalog admission (publication changes only)
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

### Residual scope (parent issue only)
<!-- Complete when this PR closes a broad parent issue. List each operation the
parent proposed and its outcome, so deferred or rejected candidates stay
visible instead of disappearing with the parent. Leave "none" when there is no
residual surface. -->

| Candidate | Outcome | Operation ID or follow-up issue |
| --- | --- | --- |
| <!-- e.g. graphicality with obstruction --> | <!-- delivered / deferred / rejected --> | <!-- operation ID or child issue --> |
