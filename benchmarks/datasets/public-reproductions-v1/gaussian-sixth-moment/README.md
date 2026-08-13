# jacobian/gaussian-sixth-moment

Compute the exact order-6 complex Gaussian moment of a polynomial.

## Field

probability

## Provenance

- case_version: public-reproductions-v1
- contamination_class: public-answer-visible-reproduction
- fixture_digest: sha256:56aaae5a7c0244c529e91e47c33dfa2d09f18aedc17eff04a28ed6c111cee3d4
- derivation: Standard complex Gaussian sixth moment; exact rational replay.

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
