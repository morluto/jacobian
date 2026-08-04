# jacobian/jcb-postdoc-015

Research diagnostic for Erdős Problem 707. The task keeps the public universal
counterexample answer visible while requiring structural evidence for the
integer Sidon property and complete fixed-order extension decisions at orders
5, 6, and 7.

## Provenance and status

- case version: `research-diagnostics-v1`
- contamination class: `public-answer-visible-diagnostic`
- fixture digest: `sha256:c1f3f9159027b5c49d4467e67f3bf4ccb27c4df73553b7d10130ad021aff4014`
- current status: `BOUNDED_CAPABILITY_AVAILABLE`
- evaluation status: `PUBLIC_REPRODUCTION_READY`
- maximum assurance: `COMPUTED`

The clean-room verifier recomputes all ordered differences and enumerates every
fixed-order candidate using only the Python standard library. It intentionally
does not certify the public universal obstruction. This public Oracle is a
regression and workflow diagnostic, not held-out causal evidence.
