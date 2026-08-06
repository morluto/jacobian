# Audit local density factors for square-free digit walks

For a prime `p`, base `b >= 2`, and digit set `T`, define the forbidden residue
set modulo `p^2` by

`F = {r : p^2 divides r or p^2 divides b*r+d for some d in T}`.

The local density factor is `(p^2-|F|)/p^2`.

Audit every frozen case in `/app/input.json`. For each case, classify the
arithmetic branch as `INVERTIBLE`, `SINGLY_DIVISIBLE`, or `SQUARE_DIVISIBLE`;
submit the complete sorted set of forbidden residues, the valid residue count,
and the density as a reduced numerator and denominator. Your concise evidence
must explain why the three divisibility branches require different reasoning
and must identify any collision or vacuous digit condition present in the
cases.

Write `/app/submission.json` and bind `/app/evidence/answer.txt` by SHA-256.
Do not claim that the global density formula, Euler-product convergence, or the
upstream Lean development has been verified. The checker establishes only the
four finite local computations and therefore permits at most `COMPUTED`.
Include this checked limitation exactly: `The checker does not verify
Euler-product convergence, the global asymptotic-density formula, or the
upstream Lean development.`

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** one of `ALL_LOCAL_FACTORS_AUDITED`, `INCOMPLETE`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** a string value
- **Completeness:** one of `COMPLETE`, `PARTIAL`, `UNKNOWN`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
