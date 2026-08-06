# jacobian/sat-witness

Decide a bounded CNF instance and provide a satisfying witness.

## Field

logic

## Provenance

- case_version: mathematical-benchmarks-v1
- contamination_class: hand-designed-structural-variant
- fixture_digest: sha256:adc92d0febe95c5950692e8dd7d53f39124978551732bb57bb4bf0a5be87e876
- derivation: Fixed four-variable CNF with a nontrivial satisfying assignment.
- derivation_note: Hand-designed bounded CNF; no solver or external proof backend is loaded at runtime.

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
