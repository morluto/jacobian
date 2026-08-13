# jacobian/lean-retrieval

Retrieve a candidate Lean tactic for a Mathlib statement.

## Field

proof-assistants

## Provenance

- case_version: public-reproductions-v1
- contamination_class: public-answer-visible-reproduction
- fixture_digest: sha256:c31079466f2b60c6a47d66ca6cec20f393cb6e713ea33b794cc676532bb4ac7b
- derivation: Lean premise retrieval reproduction.

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
