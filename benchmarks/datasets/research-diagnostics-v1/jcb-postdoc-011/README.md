# jacobian/jcb-postdoc-011

Research diagnostic: A planar counterexample to the bunkbed conjecture

## Field

percolation-theory

## Provenance

- case_version: research-diagnostics-v1
- contamination_class: public-answer-visible-diagnostic
- fixture_digest: sha256:5e14c1720158cd9ed68837035bea4bdc2872eaf3deb6b5d05752e89482b0351f
- derivation: The first planar counterexample is an explicit graph with thousands of vertices. It tests whether Jacobian can represent probabilistic graph events and verify a compressed exact witness without pretending current small-graph tools scale to it.

## Portfolio status

- historical_fit: `MISSING`
- current_status: `OPEN_GAP`
- evaluation_status: `BLOCKED_ON_INTERVENTION`
- next_action: Use small exact graph reliability as a later bounded candidate; do not represent the public counterexample with brute-force subset enumeration.

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
