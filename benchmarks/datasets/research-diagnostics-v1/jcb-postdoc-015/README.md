# jacobian/jcb-postdoc-015

Research diagnostic for Erdős Problem 707. The task keeps the public universal
counterexample answer visible while requiring structural evidence for the
integer Sidon property and complete fixed-order extension decisions at orders
5, 6, and 7.

## Provenance

- case_version: research-diagnostics-v1
- contamination_class: public-answer-visible-diagnostic
- fixture_digest: sha256:c1f3f9159027b5c49d4467e67f3bf4ccb27c4df73553b7d10130ad021aff4014
- derivation: The public universal counterexample is paired with independently replayable Sidon differences and complete fixed-order extension searches for orders 5, 6, and 7; those finite searches are explicitly not the universal obstruction.

## Portfolio status

- historical_fit: `MISSING`
- current_status: `PARTIAL`
- evaluation_status: `RUNNABLE_PUBLIC_REPRODUCTION`
- next_action: Run public Oracle validation, then evaluate capability discovery on protected transformed fixed-order cases.

## Contract

- schema_version: 1.4
- difficulty: hard
- maximum_assurance: COMPUTED
- agent-visible verification record schema: no
- timeout_sec (agent): 600.0
- timeout_sec (verifier): 120.0
- environment_mode: separate

The clean-room verifier recomputes all ordered differences and enumerates every
fixed-order candidate using only the Python standard library. It intentionally
does not certify the public universal obstruction. This public Oracle is a
regression and workflow diagnostic, not held-out causal evidence.
