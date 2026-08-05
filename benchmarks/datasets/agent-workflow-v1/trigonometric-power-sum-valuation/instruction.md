# Certify a trigonometric power-sum divisibility theorem

For positive integers `n`, let `S_n = sum_(k=1)^3 (2 sin(k*pi/7))^(2n)`.

Produce an exact symbolic certificate that `7^floor(n/3)` divides `S_n` for every positive `n`. Report the monic cubic for the three squared sine values, the initial power sums, the exact recurrence data (or an equivalent independently checkable identity), and a finite replay through `n=24` with exact values and 7-adic valuations. The derivation is agent-owned: a recurrence, Newton identities, or another exact decomposition is acceptable when it establishes the same checked artifacts.

Finally give the three residue-class cases (or an equivalent exact valuation argument). For each `n mod 3`, report the valuation offsets, relative to `floor(n/3)`, obtained from the checked certificate. The verifier recomputes the full table and the symbolic divisibility obligation.

Numerical trigonometric approximations, a finite table without the general induction step, or an unsupported `VERIFIED` claim are insufficient.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** one of `DIVISIBILITY_CERTIFIED`, `INSUFFICIENT_EVIDENCE`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** a string value
- **Completeness:** one of `COMPLETE`, `PARTIAL`, `UNKNOWN`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
