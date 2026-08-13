# jacobian/jcb-postdoc-007

Research diagnostic: Equation 834 does not imply Equation 10: an order-eight scale probe

## Field

universal-algebra

## Provenance

- case_version: research-diagnostics-v1
- contamination_class: public-answer-visible-diagnostic
- fixture_digest: sha256:0e2bde58a1e379c3a521361282210ed7cbfcb118e4910a36210f47979f7b098e
- derivation: A public order-eight countermodel lies beyond the present countermodel search order. This is a deliberate fail-closed test: the agent should discover and report the bounded-search limit, not infer an implication from failure to find a smaller model.

## Portfolio status

- historical_fit: `PARTIAL`
- current_status: `PARTIAL`
- evaluation_status: `RUNNABLE_PUBLIC_REPRODUCTION`
- next_action: Reproduce the exact public law pair under the current order-eight contract before changing this status to covered.

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
