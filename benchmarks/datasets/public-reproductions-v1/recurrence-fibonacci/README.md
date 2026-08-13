# jacobian/recurrence-fibonacci

Select the most specific operation for a recurrence-series query.

## Field

combinatorics

## Provenance

- case_version: public-reproductions-v1
- contamination_class: public-answer-visible-reproduction
- fixture_digest: sha256:e5dc32d34d869c775cd4b54008e96dab0cf55ebb66ac53766da405ddfbf8cd24
- derivation: Discovery regression for 'compute the Fibonacci number at index n'; expected first operation combinatorics.compute.fibonacci.

## Contract

- schema_version: 1.4
- difficulty: easy
- maximum_assurance: COMPUTED
- agent-visible verification record schema: no
- timeout_sec (agent): 600.0
- timeout_sec (verifier): 120.0
- environment_mode: separate

The task is self-contained and offline. The instruction names no tool,
operation, or invocation order. The verifier is a separate clean-room Python
script that scores correctness, evidence validity, scope accuracy, assurance
calibration, and aggregate reward; a wrong result or an unsupported VERIFIED
claim forces the reward to zero.
