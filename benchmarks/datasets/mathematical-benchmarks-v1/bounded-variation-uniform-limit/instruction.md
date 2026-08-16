# Separate uniform convergence from variation convergence

On `[0,2*pi]`, choose an integer `q` with `2 <= q <= 9` and use

`f_n(x) = sin(q*n*x)/(q*n)`, for `n >= 1`.

Submit a certificate that `f_n` converges uniformly to zero while every
`f_n` has total variation exactly four. State the general sup-norm bound and
the exact monotone-segment accounting: two endpoint segments and all interior
segments. Include at least four distinct freely chosen positive indices with
their frequency, amplitude, segment counts, endpoint contribution, interior
contribution, and total variation.
Represent every exact rational as a numerator/positive-denominator object.
Encode the sequence descriptor `argument_exponents` by the exponents of `q`,
`n`, and `x` in the sine argument and of `q` and `n` in the denominator.
Encode the interior segment count as an affine function of the frequency
`q*n`.
The verifier recomputes every integer and rational identity. Sampling, a graph,
or a conclusion label alone is insufficient.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
