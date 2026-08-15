# Audit consecutive power-of-two divisor minimizers

For the frozen value `k = 12`, determine the least positive integers having
exactly `2^k` and `2^(k+1)` divisors. Prove their minimality with complete exact
candidate tables: one row for every integer partition of `k` (respectively
`k+1`), recording the partition and the smallest integer realizing that
exponent shape. Submit the prime-exponent factorizations, divisor counts,
minimizers, and their integer quotient.

Write `/app/submission.json` using the public schema and bind explanatory prose
in the typed result. Candidate-table order is free, but coverage must
be exact. Claim `COMPUTED`; no proof-assistant verification is available.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier independently enumerates every exponent partition and recomputes its optimal prime assignment.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
