# jacobian/recurrence-lucas

Select the most specific operation for a recurrence-series query.

## Field

combinatorics

## Provenance

- case_version: public-reproductions-v1
- contamination_class: public-answer-visible-reproduction
- fixture_digest: sha256:98bbb0e35e00f3d0a54ce04eafdfdf3f4879a992c71565c49f8284ccbc8b4881
- derivation: Discovery regression for 'compute the Lucas number at index n'; expected first operation combinatorics.compute.lucas.

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
