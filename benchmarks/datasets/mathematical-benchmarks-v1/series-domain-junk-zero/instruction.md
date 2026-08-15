# Audit a zero outside a series definition's domain

A frozen API defines `F(s)` by `Σ n^{-s}` when the series is summable and returns `0` otherwise. Choose `s=1/q` with `3≤q≤7` and certify nonsummability using all dyadic blocks `2^k≤n<2^(k+1)` for the frozen levels.

First report the affine exponent of the general q-th-power lower bound as a coefficient of `k` and a constant term. For each frozen block, report its term count and the corresponding integer lower bound obtained from `n<2^(k+1)`. Explain why the general exponent proves that the block sums do not tend to zero. Then classify the returned zero as a fallback artifact and show that `Re(s)≠1/2`.

The verifier recomputes the symbolic exponent and all frozen powers and accepts any allowed q. The nine blocks replay instances of the general bound; finite data alone is not treated as a proof of divergence. A hard-coded point, a bare `p`-series citation, or an analytic-zero claim is insufficient. Do not claim verification of a genuine analytic continuation.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates a witness only where the task needs one.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `text/plain`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
