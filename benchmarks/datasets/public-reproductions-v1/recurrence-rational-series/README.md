# jacobian/recurrence-rational-series

Select the most specific operation for a recurrence-series query.

## Field

combinatorics

## Provenance

- case_version: public-reproductions-v1
- contamination_class: public-answer-visible-reproduction
- fixture_digest: sha256:c7eee3b3ce008f7765cf499d8068e8991645789b16481d8aeb5a99190e1176f0
- derivation: Discovery regression for 'exact coefficient prefix of a rational generating function at zero'; expected first operation combinatorics.generating_function.coefficients.compute.

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
