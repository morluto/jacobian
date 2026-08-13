# jacobian/hermite-normal-form

Compute a row Hermite normal form with an integral transformation.

## Field

linear-algebra

## Provenance

- case_version: mathematical-benchmarks-v1
- contamination_class: hand-designed-structural-variant
- fixture_digest: sha256:c8b594b227372622fb69167b5182f136eab90a5550fc25cd77eac711a72ef798
- derivation: Fixed full-rank integer matrix with a unimodular row reduction certificate.
- derivation_note: Hand-designed full-rank matrix with exact unimodular certificate.

## Contract

- schema_version: 1.4
- difficulty: hard
- maximum_assurance: VERIFIED
- agent-visible verification record schema: yes
- timeout_sec (agent): 600.0
- timeout_sec (verifier): 120.0
- environment_mode: separate

The task is self-contained and offline. The instruction names no tool,
operation, or invocation order. The verifier is a separate clean-room Python
script that scores correctness, evidence validity, scope accuracy, assurance
calibration, and aggregate reward; a wrong result or an unsupported VERIFIED
claim forces the reward to zero.
