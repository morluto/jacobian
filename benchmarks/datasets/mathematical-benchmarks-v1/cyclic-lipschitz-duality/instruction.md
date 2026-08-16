# Certify a cyclic Lipschitz optimum

Read `/app/input.json`. Maximize the sum at the marked positions over real cyclic sequences satisfying the zero-sum and adjacent-difference constraints.

Give a rational feasible sequence and a rational edge circulation `q`, representing each entry as a `{"numerator": integer, "denominator": positive integer}` object. Equivalent encodings such as `2/4` and `1/2` are accepted after exact `Fraction` normalization. With indices modulo the frozen cycle size, require `q_i-q_(i-1)=w_i`, where `w_i=1-m/n` at a marked position and `-m/n` otherwise, for cycle size `n` and `m` marked positions. Its `L1` norm is the dual value.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
