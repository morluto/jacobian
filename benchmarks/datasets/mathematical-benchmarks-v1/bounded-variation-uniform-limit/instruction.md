# Separate uniform convergence from variation convergence

On `[0,2*pi]`, choose an integer `q` with `2 <= q <= 9` and use

`f_n(x) = sin(q*n*x)/(q*n)`, for `n >= 1`.

Submit a certificate that `f_n` converges uniformly to zero while every
`f_n` has total variation exactly four. State the general sup-norm bound and
the exact monotone-segment accounting: two endpoint segments and all interior
segments. Include at least four distinct freely chosen positive indices with
their frequency, amplitude, segment counts, endpoint contribution, interior
contribution, and total variation.
In `result.argument`, use the three typed values
`SUP_NORM_1_OVER_QN_TENDS_TO_ZERO`, `TOTAL_VARIATION_IS_CONSTANTLY_FOUR`,
and `UNIFORM_CONVERGENCE_DOES_NOT_FORCE_VARIATION_CONVERGENCE` to record the
mathematical explanation.

The verifier recomputes every integer and rational identity. Sampling, a graph,
or a conclusion label alone is insufficient; the typed `result.argument` values
carry the mathematical separation.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
