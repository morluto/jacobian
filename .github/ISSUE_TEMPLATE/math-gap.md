---
name: Mathematical vocabulary gap
about: Report a missing mathematical result and diagnose what kind of gap it is
labels: ["status: needs-triage"]
assignees: []
---

## Observed mathematical task(s)
<!-- Which real problems required this result? -->

## Missing mathematical result
<!-- State the result the agent could not obtain. No API proposal here yet. -->

## Operation failure record (when an existing operation was attempted)
<!--
Preserve the exact evidence rather than only summarizing the failure. Write
"not retained" for unavailable fields instead of reconstructing them later.
-->
- Operation ID and version:
- Exact request or minimal reproducible fixture:
- Runtime and catalog revision (keep installed MCP and local source distinct
  unless their identity is verified):
- Exact typed result, validation error, or transport error:
- Timeout owner, configured deadline, and observed elapsed time (if applicable):
- Failure stage and contract owner (if applicable):
- Replay/verification outcome (only if the public contract has one):

## Bespoke escape used
<!-- What custom code, external library call, solver encoding, or manual work was needed? -->

## Gap classification
<!-- Check one. An operation gap records a missing postcondition; it does not pre-admit a public operation. -->
- [ ] representation
- [ ] interoperability
- [ ] discovery
- [ ] contract
- [ ] scale/backend
- [ ] operation
- [ ] reasoning

## Domain, bounds, and acceleration evidence
<!--
If a current operation has the right postcondition, separate its semantic
mathematical domain from the admitted execution envelope. Which limit is
mathematical, representation-specific, backend-specific, or merely
uninvestigated? Name the quantities that actually control work, intermediate
growth, memory, and output. Record any result-sensitive admission, compact
exact representation, maintained backend, algorithm-regime switch, batching,
or deterministic partitioning considered before proposing a smaller domain or
a new operation. A timeout or UNKNOWN result is not evidence of mathematical
nonexistence.
-->

## Catalog disposition (operation gaps only)
<!-- State whether the evidence supports public-catalog consideration, suggests
native-only support, or leaves disposition unresolved. This is catalog
publication; request bounds for an existing operation belong to its domain. -->

## Existing math.find queries
<!-- Queries already tried and what they returned. Search misses alone do not
establish absence; inspect plausible operation contracts and actual composition. -->

## Existing operations inspected
<!-- Closest current operations and why a clean composition still fails. -->

## Why composition was insufficient
<!-- The indispensable mathematical postcondition not already available. -->

## Evidence and downstream uses
<!--
Traces, benchmarks, or unrelated tasks needing the same result. Include
realistic source-backed cases and relevant algorithm or representation
crossover points when scale is part of the gap.
-->
