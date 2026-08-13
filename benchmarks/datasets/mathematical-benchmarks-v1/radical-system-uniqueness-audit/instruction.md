# Audit a radical system and certify its unique real solution

The frozen input contains a real system involving square, cube, and fourth
roots, together with a claim that it has at least two solutions. Determine the
actual real solution set and audit that claim.

Submit a certificate that derives and independently checks an exact
univariate elimination polynomial, classifies every real root against the
principal-root domain constraints, reconstructs every surviving `(a,b,c)`
triple, and checks all three original equations exactly. You may choose the
valid elimination parameterization and algebraic route.
Your evidence must explain why rejected algebraic roots cannot represent real
solutions. Write `submission.json` according to `submission_schema.json` and
bind `evidence/answer.txt` by SHA-256.

This task has no external proof-assistant replay, so claim at most `COMPUTED`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** one of `UNIQUE_REAL_SOLUTION`, `MULTIPLE_REAL_SOLUTIONS`, `INSUFFICIENT_EVIDENCE`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** one of `COMPLETE`, `PARTIAL`, `UNKNOWN`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
