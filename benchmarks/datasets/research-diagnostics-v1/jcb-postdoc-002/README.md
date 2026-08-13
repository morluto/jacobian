# jacobian/jcb-postdoc-002

Research diagnostic: Written on the Wall II Conjecture 200: a 14-vertex exact counterexample

## Field

extremal-graph-theory

## Provenance

- case_version: research-diagnostics-v1
- contamination_class: public-answer-visible-diagnostic
- fixture_digest: sha256:7a70b36e43e03fdf55e966f8bfec9863e47ce312d086c103957c1c3463f3e905
- derivation: This small graph has a short structural proof and independent finite certificates. It probes whether an agent can combine graph construction, neighborhood invariants, exact maximum induced-tree computation, and a Hamiltonian obstruction.

## Portfolio status

- historical_fit: `DIRECT`
- current_status: `COVERED`
- evaluation_status: `RUNNABLE_PUBLIC_REPRODUCTION`
- next_action: Add the exact 14-vertex public object as a stable regression only if its redistribution and fixture size remain appropriate.

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
