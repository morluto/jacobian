# Polynomial normalization

Combine like terms in the exact sparse polynomial from `input.json`. Return
rational coefficients as integer `numerator`/positive integer
`denominator` objects and exponent vectors, omitting zero terms.
Equivalent encodings such as `6/2` and `3/1` are accepted after exact
`Fraction` normalization.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
