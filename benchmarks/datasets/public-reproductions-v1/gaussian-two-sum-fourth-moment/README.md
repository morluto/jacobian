# jacobian/gaussian-two-sum-fourth-moment

Compute the exact order-4 complex Gaussian moment of a polynomial.

## Field

probability

## Provenance

- case_version: public-reproductions-v1
- contamination_class: public-answer-visible-reproduction
- fixture_digest: sha256:13dfb30bfb7a94bad2c34340aa64ed9469776b3f704562a85037ac41d58295c6
- derivation: Two-variable Gaussian sum fourth moment; exact rational replay.

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
