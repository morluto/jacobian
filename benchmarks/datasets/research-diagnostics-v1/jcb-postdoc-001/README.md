# jacobian/jcb-postdoc-001

Research diagnostic: Exact verification of the three-variable Keller-map collision

## Field

commutative-algebra

## Provenance

- case_version: research-diagnostics-v1
- contamination_class: public-answer-visible-diagnostic
- fixture_digest: sha256:6640c15efcc9a557682a94c49e4433eb5025f71ba785710bbd19832c4cfe94dc
- derivation: A recently announced explicit three-variable polynomial map supplies a finite disproof witness for the characteristic-zero Jacobian conjecture. This reproduction isolates the two exact obligations that Jacobian should already handle well.

## Portfolio status

- historical_fit: `DIRECT`
- current_status: `COVERED`
- evaluation_status: `REGRESSION_COVERED`
- next_action: Keep the public collision as a no-retrieval regression and preserve producer/checker separation.

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
