# jacobian/jacobian-inverse-obstruction

Verify a two-point collision witnessing non-invertibility of a polynomial map over Q.

## Field

algebra

## Provenance

- case_version: public-reproductions-v1
- contamination_class: public-answer-visible-reproduction
- fixture_digest: sha256:c1463d2778f12ed1c2bcc8417182406d7cbdaaf31366944dfbb1364dbbd27ad6
- derivation: Two-point rational collision witnessing non-invertibility; exact replay.

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
