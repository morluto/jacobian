# jacobian/smith-rectangular

Compute the Smith normal form of a rectangular integer matrix.

## Field

linear-algebra

## Provenance

- case_version: public-reproductions-v1
- contamination_class: public-answer-visible-reproduction
- fixture_digest: sha256:36852b839bffcba6da9c474cc2c409e99a9688d459fb7a0f35e9fa98d2ff6e3f
- derivation: Fixed 2x3 integer matrix; canonical Smith normal form replay.

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
