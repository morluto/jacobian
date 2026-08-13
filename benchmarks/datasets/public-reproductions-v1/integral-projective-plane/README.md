# jacobian/integral-projective-plane

Compute unreduced integral homology of a six-vertex triangulation of RP^2.

## Field

algebraic-topology

## Provenance

- case_version: public-reproductions-v1
- contamination_class: public-answer-visible-reproduction
- fixture_digest: sha256:22d1152268b00b5e81611c781f55624475a7ae56c96a493186ee76ff8333175c
- derivation: Six-vertex RP^2 triangulation; unreduced integral homology replay.

## Contract

- schema_version: 1.4
- difficulty: hard
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
