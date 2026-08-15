# Certify reversal symmetry of a variable-coefficient recurrence

Let `u_0=u_1=v_0=v_1=1`, `u_(k+1)=u_k+a_k*u_(k-1)`, and `v_(k+1)=v_k+a_(n-k)*v_(k-1)`. Produce a symbolic tiling certificate for `u_n=v_n` with arbitrary commuting coefficients.

For the frozen board length `n=10`, list every square-free monomial support with no adjacent indices that occurs in each final recurrence polynomial. Then give the complete reflection pairing induced by `i -> n-i`, together with the general recurrence contract.

The verifier independently enumerates the monomials and reflection map. Numeric
coefficient substitutions or an answer-only equality are insufficient.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
