# jacobian/reduced-point

Compute reduced integral homology of a one-point simplicial complex.

## Field

algebraic-topology

## Provenance

- case_version: public-reproductions-v1
- contamination_class: public-answer-visible-reproduction
- fixture_digest: sha256:536bcaaf096e412463cc5d18050e5743d4ea7d29b1ba67beb0a492443834706e
- derivation: One-point complex; reduced integral homology boundary replay.

## Contract

- schema_version: 1.4
- difficulty: easy
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
