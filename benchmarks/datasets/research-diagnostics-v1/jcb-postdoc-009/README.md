# jacobian/jcb-postdoc-009

Research diagnostic: Irreducible vertices in positive-definite weighted tree lattices

## Field

lattice-theory

## Provenance

- case_version: research-diagnostics-v1
- contamination_class: public-answer-visible-diagnostic
- fixture_digest: sha256:10bea127b1944406b2613fb3242daef82d1874ab08098bcfe342e4a657e36720
- derivation: This unpublished expert problem requires a structural argument about decompositions in an integral lattice. It is suitable for testing theorem-level reasoning and highlights the absence of lattice-element decomposition and irreducibility operations.

## Portfolio status

- historical_fit: `MISSING`
- current_status: `OPEN_GAP`
- evaluation_status: `BLOCKED_ON_INTERVENTION`
- next_action: Mine a second independent weighted-lattice workflow before proposing a domain contract.

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
