# jacobian/finite-partition

Partition a finite universe into exact residue classes.

## Field

number-theory

## Provenance

- case_version: mathematical-benchmarks-v1
- contamination_class: hand-designed-structural-variant
- fixture_digest: sha256:058a953f55dfbb537df91e94244944488ef78c6a4ecf9add0fef3b140aaeb5b2
- derivation: Fixed twelve-element universe and modulo-three partition relation.
- derivation_note: Hand-designed finite coverage contract; no external source is loaded at runtime.

## Contract

- schema_version: 1.4
- difficulty: medium
- maximum_assurance: VERIFIED
- agent-visible verification record schema: yes
- timeout_sec (agent): 600.0
- timeout_sec (verifier): 120.0
- environment_mode: separate

The task is self-contained and offline. The instruction names no tool,
capability, or invocation order. The verifier is a separate clean-room Python
script that scores correctness, evidence validity, scope accuracy, assurance
calibration, and aggregate reward; a wrong result or an unsupported VERIFIED
claim forces the reward to zero.
