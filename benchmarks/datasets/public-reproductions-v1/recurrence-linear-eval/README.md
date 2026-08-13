# jacobian/recurrence-linear-eval

Select the most specific operation for a recurrence-series query.

## Field

combinatorics

## Provenance

- case_version: public-reproductions-v1
- contamination_class: public-answer-visible-reproduction
- fixture_digest: sha256:5dda8dc82cd6e29003e68160dd0d028bd6d610fa1618cf07e5118e166a1fc0fc
- derivation: Discovery regression for 'evaluate requested indices of a constant coefficient linear recurrence'; expected first operation combinatorics.recurrence.linear.evaluate.

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
