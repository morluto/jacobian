# jacobian/gaussian-complex-cancellation

Compute the exact order-2 complex Gaussian moment of a polynomial.

## Field

probability

## Provenance

- case_version: public-reproductions-v1
- contamination_class: public-answer-visible-reproduction
- fixture_digest: sha256:c4479f2a152004c1ff1c5b480b6ed46200c405e10d4c0a8cc62cb5ae05559e17
- derivation: Complex cancellation second moment; exact rational replay.

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
