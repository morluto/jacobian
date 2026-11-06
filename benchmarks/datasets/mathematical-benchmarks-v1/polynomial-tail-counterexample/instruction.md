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

The verifier replays the task-specific mathematical predicate from the submitted result. The finite certificate envelope permits polynomial degree at most 32 (at most 33 coefficients and 32 roots per polynomial), and rational integer components of absolute value at most 2^256-1 with positive denominators. This bounds at most 64 root evaluations of degree 32, with exact intermediate bit lengths bounded by the submitted component widths and degree; it is a construction task within this public envelope, not an enumeration of all counterexamples. Integer fields use exact JSON-number values: 2, 2.0, and 2e0 are equivalent. Numeric tokens are limited to 4300 characters, expanded integers to 4300 digits, and decimal exponents to [-4300,4300] before conversion, in addition to the tighter magnitude bounds above. The complete UTF-8 submission file, including whitespace, is limited to 16777216 bytes. Exact-number parsing also admits at most 67108864 cumulative numeric-storage charge bytes per JSON document, charging 256 + 8*(2*token characters + absolute decimal exponent) before each numeric conversion.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
