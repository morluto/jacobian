# jacobian/well-total-domination-counterexample

Check a finite graph counterexample to a well-total-domination claim.

## Field

graph-theory

## Provenance

- case_version: mathematical-benchmarks-v1
- contamination_class: public-known-counterexample
- fixture_digest: sha256:93f08e9913d479fe672a9d8ca918c74ce0bc96c7557d02f2fd44c88d2334047e
- upstream: https://github.com/google-deepmind/formal-conjectures/issues/4133
- upstream_license: Apache-2.0
- derivation: The five-vertex path counterexample is independently re-encoded
  as a finite graph fixture; no upstream proof code is copied.
- derivation_note: The public source states the same P5 graph, exact degree
  calculation, and two minimal total-dominating sets of different sizes.

## Contract

- schema_version: 1.4
- difficulty: medium
- maximum_assurance: COMPUTED
- agent-visible verification record schema: no
- timeout_sec (agent): 600.0
- timeout_sec (verifier): 120.0
- environment_mode: separate

The task is self-contained and offline. The instruction names no tool,
operation, or invocation order. The clean-room verifier independently checks
the graph, connectivity, exact average degree, pendant vertices, and both
minimal total-dominating sets. A wrong result forces reward to zero.
