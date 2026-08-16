# Polynomial-tail monotonicity counterexample

Decide the polynomial claim in the offline input. If it is false, return
rational coefficient lists for `P` and `Q`, their distinct real roots, and an
exact rational pair `x1 < x2` at or beyond every root for which
`(P-Q)(x1) >= (P-Q)(x2)`.

Represent every rational, including every listed root, as a structured
`{numerator, denominator}` object. Equivalent encodings such as `2/6` and
`1/3` are accepted after exact `Fraction` normalization. This benchmark
intentionally restricts roots to rational values; algebraic irrational roots
are outside the advertised artifact contract. Write `submission.json` to the
exact agent-visible `submission_schema.json`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
