# jacobian/graph-artifact-composition

Compose a graph distance artifact with a maximum-degree vertex set.

## Field

graph-theory

## Provenance

- case_version: mathematical-benchmarks-v1
- contamination_class: hand-designed-structural-variant
- fixture_digest: sha256:4fb641fa7ebbad8b497f422242b90de883fb574226bbc60c16571b41225a923c
- derivation: Fixed connected six-vertex graph with a non-singleton distance calculation.
- derivation_note: Hand-designed finite graph; no external source is loaded at runtime.

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
