# Audit local density factors for square-free digit walks

For a prime `p`, base `b >= 2`, and digit set `T`, define the forbidden residue
set modulo `p^2` by

`F = {r : p^2 divides r or p^2 divides b*r+d for some d in T}`.

The local density factor is `(p^2-|F|)/p^2`.

Audit every frozen case in `/app/input.json`. For each case, classify the
arithmetic branch as `INVERTIBLE`, `SINGLY_DIVISIBLE`, or `SQUARE_DIVISIBLE`;
submit the complete sorted set of forbidden residues, the valid residue count,
and the density as a reduced numerator and denominator. The checker establishes
only the four finite local computations; it does not establish Euler-product
convergence, the global asymptotic-density formula, or the upstream Lean
development.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
