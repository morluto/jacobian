# jacobian/symmetry-cycle-rotation

Compute declared-generator vertex and edge orbits of a small graph.

## Field

graph-theory

## Provenance

- case_version: public-reproductions-v1
- contamination_class: public-answer-visible-reproduction
- fixture_digest: sha256:c01264555b33184b80c7fb29bf74a2dc8b43dc4596ccff6306e50b1f7a9bf83e
- derivation: Four-cycle graph under quarter-turn; orbit replay.

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
