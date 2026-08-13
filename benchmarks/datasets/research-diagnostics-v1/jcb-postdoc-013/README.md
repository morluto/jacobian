# jacobian/jcb-postdoc-013

Research diagnostic: Erdős Problem 397: a parametric family of central-binomial collisions

## Field

enumerative-combinatorics

## Provenance

- case_version: research-diagnostics-v1
- contamination_class: public-answer-visible-diagnostic
- fixture_digest: sha256:1fbc7f3e3469aaebc634f220c051ca738d6d4b1db35252f7a480f1c0479bc0a0
- derivation: Erdős asked whether products of distinct central binomial coefficients can coincide in only finitely many ways. An explicit one-parameter family gives infinitely many collisions and is a sharp test of whether finite exact checks are kept separate from a proof for every parameter.

## Portfolio status

- historical_fit: `PARTIAL`
- current_status: `PARTIAL`
- evaluation_status: `RUNNABLE_PUBLIC_REPRODUCTION`
- next_action: Use the no-retrieval runner to measure whether agents keep finite checks separate from the all-parameter proof; do not add a contract until recurrent symbolic workflows are observed.

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
