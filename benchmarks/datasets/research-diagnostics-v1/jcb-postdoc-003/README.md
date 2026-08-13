# jacobian/jcb-postdoc-003

Research diagnostic: Equational implication 4155 to 4658: finite countermodel discovery

## Field

universal-algebra

## Provenance

- case_version: research-diagnostics-v1
- contamination_class: public-answer-visible-diagnostic
- fixture_digest: sha256:975d1718e2799d05d9817d9772a14dfe3fa12486df44327a5dbd4515f349a027
- derivation: A small countermodel tests Jacobian's domain-owned universal-algebra search and law evaluator without requiring the model to know the public operation in advance.

## Portfolio status

- historical_fit: `DIRECT`
- current_status: `COVERED`
- evaluation_status: `REGRESSION_COVERED`
- next_action: Keep the order-two oracle and bounded no-witness behavior as separate regression obligations.

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
