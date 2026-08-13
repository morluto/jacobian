# jacobian/jcb-postdoc-018

Research diagnostic: WOWII Conjecture 18: detect a semantic mismatch between two graph invariants

## Field

extremal-graph-theory

## Provenance

- case_version: research-diagnostics-v1
- contamination_class: public-answer-visible-diagnostic
- fixture_digest: sha256:2f9ee00f9b80a6a1f26f418da0e0190519b7e2aa5b595448ac6e5cf0dc67a56e
- derivation: A solved graph conjecture was formalized with set eccentricity in place of the source's maximum distance among maximum-degree vertices. A six-vertex graph falsifies the formalized statement while satisfying the intended one. This tests whether operation descriptions and outputs preserve the exact invariant being computed.

## Portfolio status

- historical_fit: `PARTIAL`
- current_status: `PARTIAL`
- evaluation_status: `BLOCKED_ON_INTERVENTION`
- next_action: Implement and independently replay one restricted-set distance profile before evaluating semantic-confusion reduction.

## Contract

- schema_version: 1.4
- difficulty: hard
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
