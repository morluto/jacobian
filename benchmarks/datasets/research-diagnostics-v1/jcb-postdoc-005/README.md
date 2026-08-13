# jacobian/jcb-postdoc-005

Research diagnostic: A three-dimensional quartic counterexample to the Gaussian Moments Conjecture

## Field

probability-theory

## Provenance

- case_version: research-diagnostics-v1
- contamination_class: public-answer-visible-diagnostic
- fixture_digest: sha256:544de7516e45fb567eedbacbe3f8f6bc4535d10f8cac49c8b7a24d41a36b8ee3
- derivation: This all-exponents identity is far beyond checking finitely many moments. It tests whether an agent can combine exact polynomial algebra with a human-owned generating-function proof and clearly identify the missing symbolic-expectation operation.

## Portfolio status

- historical_fit: `PARTIAL`
- current_status: `PARTIAL`
- evaluation_status: `BLOCKED_ON_INTERVENTION`
- next_action: Implement and evaluate finite-distribution foundations separately; retain bounded Gaussian polynomial moments and all-m symbolic identities as later candidates.

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
