# Factor and certify reciprocal symmetry

The frozen input gives a degree-16 integer polynomial by coefficients in constant-to-leading order. Recover a complete factorization

`leading_coefficient * product(Phi_order(x)^multiplicity)`

using cyclotomic orders from 2 through 30. Submit the ordered factor list, the independently expanded coefficient vector, its reciprocal vector, `P(1)`, and the reciprocal scalar.

The verifier reconstructs cyclotomic polynomials and the entire product; coefficient-pattern recognition alone is insufficient.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
