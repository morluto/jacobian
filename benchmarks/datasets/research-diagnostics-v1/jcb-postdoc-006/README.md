# jacobian/jcb-postdoc-006

Research diagnostic: Written on the Wall II Conjecture 59 at portfolio scale limits

## Field

extremal-graph-theory

## Provenance

- case_version: research-diagnostics-v1
- contamination_class: public-answer-visible-diagnostic
- fixture_digest: sha256:a1e81261ab4aa9deab5eb74c120d912f32b7890925ba2d45f07e8370a268f7a5
- derivation: The explicit graph is much larger than current bounded exact graph inputs, but its invariants admit short structural proofs. This probes whether the agent can avoid forcing a large instance through an unsupported contract and instead compose small exact calculations with transparent mathematics.

## Portfolio status

- historical_fit: `PARTIAL`
- current_status: `PARTIAL`
- evaluation_status: `RUNNABLE_PUBLIC_REPRODUCTION`
- next_action: Use the public run to audit structural decomposition and fail-closed scale handling; do not raise a JSON bound without algorithmic and certificate evidence.

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
