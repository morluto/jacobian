# jacobian/jacobian-negative-control

Negative control: a mutated collision image must remain an UNKNOWN non-conclusion.

## Field

algebra

## Provenance

- case_version: public-reproductions-v1
- contamination_class: public-answer-visible-reproduction
- fixture_digest: sha256:321b50efad411db9a7e38c3ad4ad22c7c0c4f5a4df91d1f9280d72f548c931d3
- derivation: Mutated collision image; fail-closed UNKNOWN non-conclusion replay.

## Contract

- schema_version: 1.4
- difficulty: medium
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
