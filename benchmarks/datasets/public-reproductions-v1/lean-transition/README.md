# jacobian/lean-transition

Apply a Lean tactic to a proof state and report the resulting goal count.

## Field

proof-assistants

## Provenance

- case_version: public-reproductions-v1
- contamination_class: public-answer-visible-reproduction
- fixture_digest: sha256:be2b43e35b0fff3d2f40c061f37e5bfc7b6ec41d9f21ec4ebd068ed1d0d02ce1
- derivation: Lean proof-state transition reproduction.

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
