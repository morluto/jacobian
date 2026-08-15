# Repair an inseparability proof at the minimal-polynomial boundary

Let `K` have prime characteristic `p`, let `t` be transcendental over `K`,
and write `u=t^p`.  The frozen proof observes
`X^p-u=(X-t)^p` and immediately declares the minimal polynomial inseparable.
That inference is incomplete: a repeated-root annihilating polynomial need not
itself be minimal.

Submit a symbolic certificate that diagnoses the gap and repairs it by proving
`X^p-u` irreducible over `K(u)` by any independently checkable argument.  For
example, the `u`-adic valuation shows every `p`-th power in `K(u)` has
valuation divisible by `p`, whereas `v_u(u)=1`; an Eisenstein argument at the
prime `u` works equally well.  Then bind the resulting minimal-polynomial
degree, zero derivative, and repeated-root conclusion.  Include any prime
`2 <= p <= 97` as a freely chosen sanity instance; the verifier must accept
alternatives.

A factorization alone, a numeric instance alone, or the source conclusion
alone is insufficient. Write `/app/submission.json`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
