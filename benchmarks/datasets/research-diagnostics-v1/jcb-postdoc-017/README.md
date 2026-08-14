# jacobian/jcb-postdoc-017

Research diagnostic: Exact LP certificate for the pairwise-independent correlation-gap counterexample

## Field

submodular-optimization

## Provenance

- case_version: research-diagnostics-v1
- contamination_class: public-answer-visible-diagnostic
- fixture_digest: sha256:2eb4dae47798fc00527f4a79d212aca79039776cf24791ca839e0a5127278c7c
- derivation: A five-element coverage function and rational marginal vector violate a proposed 4/3 upper bound. The counterexample has a tiny primal witness and a 32-constraint exact dual certificate, making it suitable for current rational optimization and finite-coverage operations.

## Portfolio status

- historical_fit: `DIRECT`
- current_status: `PARTIAL`
- evaluation_status: `BLOCKED_ON_INTERVENTION`
- next_action: Implement the finite coverage-semantic dual checker, then compare it with generic LP-only traces on generated held-out instances.

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
