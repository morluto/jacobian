# jacobian/jacobian-keller

Verify the Keller condition and Jacobian determinant of a three-variable polynomial map over Q.

## Field

algebra

## Provenance

- case_version: public-reproductions-v1
- contamination_class: public-answer-visible-reproduction
- fixture_digest: sha256:580784458d5dbda3c9f8ac7f0b7bc7daa512e992980a74716a37157e93b30fcc
- derivation: Three-variable Keller-map polynomial; exact Jacobian determinant replay.

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
