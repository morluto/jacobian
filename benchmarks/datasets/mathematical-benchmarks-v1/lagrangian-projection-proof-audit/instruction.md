Audit the frozen projection step from the supplied research-proof correction.

Submit an exact rational full-rank matrix `D` for which `L = D^T J D` is nonzero, together with nonzero coefficient matrices `P` and `Q`. Reconstruct `W = D P + J D N Q`, where `N = (D^T D)^{-1}`. Report the Gram matrix, its inverse, the Lagrangian defect, both naive projections, and both corrected projection expressions.

Your submitted matrices must make each naive projection differ from its intended coefficient while both corrected identities hold exactly. Rational entries are `{numerator, denominator}` objects; equivalent encodings such as `2/2` and `1` are accepted after exact `Fraction` normalization.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
