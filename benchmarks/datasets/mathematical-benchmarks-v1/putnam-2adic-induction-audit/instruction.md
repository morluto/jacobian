# Audit a 2-adic doubling induction

For the frozen recurrence `b_0=0` and
`b_(n+1)=2*b_n^2+b_n+1`, set `u_n=2*b_n` and let
`P_n=product_(0<=j<n)(2*u_j+1)`.

Submit an exact symbolic certificate that:

1. checks the recurrence and the base values at `n=2`;
2. gives the difference factorization underlying translation congruence;
3. records the Taylor and product-doubling identities used in the proof;
4. propagates the three simultaneous exact affine 2-adic valuations from `k`
   to `k+1`, while recording intermediate remainder-term estimates explicitly
   as lower bounds rather than exact valuations;
5. derives the valuation of
   `b_(2^(k+1))-2*b_(2^k)` and both divisibility conclusions; and
6. distinguishes this universal certificate from finite numerical testing.

Affine functions of `k` are represented as `[coefficient, constant]` for
`coefficient*k+constant`. Fields ending in `_lower_bounds` assert `v_2(term)`
is at least that affine value; `hypotheses`, `successor`, `u_difference`, and
`b_difference` assert exact valuations. The doubling identities are frozen source premises;
your certificate must use them to establish the required affine valuation
relations and strict gaps symbolically, not only for selected values of `k`.
Encode the difference identity using ascending coefficients of `f` and two
linear factors in `a,b`. Encode each doubling identity as coefficients of its
named monomials after moving every term to the left-hand side. Encode the
add-one valuation implication by its premise lower bound and exact conclusion.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
