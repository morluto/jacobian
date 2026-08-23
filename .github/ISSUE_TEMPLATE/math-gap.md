---
name: Mathematical vocabulary gap
about: Report a missing mathematical result and diagnose what kind of gap it is
labels: []
assignees: []
---

## Observed mathematical task(s)
<!-- Which real problems required this result? -->

## Missing mathematical result
<!-- State the result the agent could not obtain. No API proposal here yet. -->

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

## Admission posture (operation gaps only)
<!-- State whether the evidence supports public-catalog consideration, suggests native-only support, or leaves disposition unresolved. Public admission is a later decision. -->

## Existing math.find queries
<!-- Queries already tried and what they returned. -->

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
