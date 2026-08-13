# jacobian/integral-circle

Compute unreduced integral homology of a triangle boundary model of S^1.

## Field

algebraic-topology

## Provenance

- case_version: public-reproductions-v1
- contamination_class: public-answer-visible-reproduction
- fixture_digest: sha256:4d014e0acb89a4a6839adb6b8b411db6dd988d614f305fa9501f7983510f2c6b
- derivation: Triangle boundary simplicial complex; unreduced integral homology replay.

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
