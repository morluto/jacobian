# jacobian/smith-rank-deficient

Compute the Smith normal form of a rank-deficient integer matrix.

## Field

linear-algebra

## Provenance

- case_version: public-reproductions-v1
- contamination_class: public-answer-visible-reproduction
- fixture_digest: sha256:89c380ca5ad1b5769f08d32f24a6a481c9136e70a1c0de1d3b3e90e5baf585fd
- derivation: Fixed 3x2 rank-deficient integer matrix; canonical Smith normal form replay.

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
