# Construct a closed subspace with nonclosed projected image

Give an explicit Hilbert space `H`, closed linear subspace `M`, and orthogonal
projection `P` such that `P(M)` is not closed.

Certify every link of the counterexample: the mechanism establishing that `M`
is closed, identification of the projection image `P(M)`, a sequence inside
that image converging to a declared limit, a tail bound proving convergence,
and a proof that the limit has no `ell2` preimage.  For every prefix through
`prefix_length`, submit the exact rational weight, limit coordinate, forced
preimage coordinate, and partial squared norms of the limit and its forced
preimage.

Put the five structural claims in the agent-visible `result.proof_obligations`
fields (`boundedness`, `closedness`, `range_identification`, `convergence`, and
`absent_preimage`).

A finite-dimensional matrix, a bare theorem citation, or a single approximate
sequence is insufficient. Submit every exact rational as a
`{"numerator": integer, "denominator": positive integer}` object. Equivalent encodings such as `2/2` and `1/1` are accepted after exact `Fraction` normalization.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result. The finite replay envelope contains exactly 12 prefixes and 100 through 128 limit coordinates, matching tail_bound.verification_terms. Rational numerators have absolute value at most 2^1024-1 and denominators lie in [1, 2^1024-1]. Tail and growth exponents lie in [1,100]. These limits bound at most 128 rational squares and suffix sums; a suffix numerator/denominator has at most 2*1024*128+8 bits before reduction. Integer fields use exact JSON-number values: 2, 2.0, and 2e0 are equivalent. Numeric tokens are limited to 4300 characters, expanded integers to 4300 digits, and decimal exponents to [-4300,4300] before conversion, in addition to the tighter magnitude bounds above. The complete UTF-8 submission file, including whitespace, is limited to 16777216 bytes.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
