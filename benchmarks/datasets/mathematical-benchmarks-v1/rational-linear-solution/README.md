# jacobian/rational-linear-solution

Solve an exact rational linear system.

## Field

linear-algebra

## Provenance

- case_version: mathematical-benchmarks-v1
- contamination_class: hand-designed-structural-variant
- fixture_digest: sha256:7ca7f46fa89c2b4f7260fd01698dfe5c2a7414116f45c0b00541666270f5816e
- derivation: Fixed two-variable system with a non-integral unique solution.
- derivation_note: Hand-designed exact system; no floating-point arithmetic is needed.

## Contract

- schema_version: 1.4
- difficulty: medium
- maximum_assurance: COMPUTED
- agent-visible verification record schema: no
- timeout_sec (agent): 600.0
- timeout_sec (verifier): 120.0
- environment_mode: separate

The task is self-contained and offline. The instruction names no tool,
capability, or invocation order. The verifier is a separate clean-room Python
script that scores correctness, evidence validity, scope accuracy, assurance
calibration, and aggregate reward; a wrong result or an unsupported VERIFIED
claim forces the reward to zero.
