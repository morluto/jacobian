# Factor the nonempty-intersection matrix

For the matrix indexed by nonempty subsets of `[n]`, with entry one exactly
when two subsets intersect, derive its determinant for every `n>=1`.

Submit an exact incidence factorization.  Choose and describe a valid
`sample_n`, provide any inclusion-linear order of all nonzero bitmasks and
the diagonal weights in that basis. One privileged listing of the same masks is not required.  Also provide the determinant parity trace
for every `n` through `trace_max_n`, including the number of nonempty even-cardinality
subsets and the resulting determinant.
Encode the general even-count expression by its power base, exponent offset,
and constant offset, and encode the general determinant by its value at `n=1`
and its value otherwise.

The public determinant values alone are insufficient.  The verifier rebuilds
the intersection matrix and checks the submitted factorization exactly, then
recomputes the parity formula.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

Choose an integer `sample_n` from 1 through 8. For that choice, `mask_order` contains every nonempty subset mask exactly once: it has `2^sample_n - 1` entries, each an integer from 1 through `2^sample_n - 1`. `diagonal_weights` has the same length, with each weight equal to -1 or 1. Any inclusion-linear mask order is permitted. The verifier checks the factorization, weights, parity trace, and general formulas mathematically. The trace contains exactly `trace_max_n` entries (8 in the public input), one for each n from 1 through `trace_max_n`, in any order.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
