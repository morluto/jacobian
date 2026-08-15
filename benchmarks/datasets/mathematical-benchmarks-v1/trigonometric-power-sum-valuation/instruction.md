# Certify a trigonometric power-sum divisibility theorem

For positive integers `n`, let `S_n = sum_(k=1)^3 (2 sin(k*pi/7))^(2n)`.

Produce an exact symbolic certificate that `7^floor(n/3)` divides `S_n` for every positive `n`. Report the monic cubic for the three squared sine values, the initial power sums, the exact recurrence data (or an equivalent independently checkable identity), and a finite replay through `n=24` with exact values and 7-adic valuations. The derivation is agent-owned: a recurrence, Newton identities, or another exact decomposition is acceptable when it establishes the same checked artifacts.

Finally give the three residue-class cases (or an equivalent exact valuation argument). For each `n mod 3`, report the valuation offsets, relative to `floor(n/3)`, obtained from the checked certificate. The verifier recomputes the full table and the symbolic divisibility obligation.

Numerical trigonometric approximations or a finite table without the general induction step are insufficient.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
