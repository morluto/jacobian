# Audit a zero outside a series definition's domain

A frozen API defines `F(s)` by `Σ n^{-s}` when the series is summable and returns `0` otherwise. Choose `s=1/q` with `3≤q≤7` and certify nonsummability using all dyadic blocks `2^k≤n<2^(k+1)` for the frozen levels.

Represent the exact real part as an integer `numerator` and positive integer
`denominator` in lowest terms.

First report the affine exponent of the general q-th-power lower bound as a coefficient of `k` and a constant term. For each frozen block, report its term count and the corresponding integer lower bound obtained from `n<2^(k+1)`. Include the divergence and returned-value classifications in the typed result.

The verifier recomputes the symbolic exponent and all frozen powers and accepts any allowed q. The nine blocks replay instances of the general bound; finite data alone is not treated as a proof of divergence.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
