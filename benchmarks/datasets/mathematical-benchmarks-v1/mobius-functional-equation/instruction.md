# Solve a functional equation through its Möbius orbit

Find every function `F : R \ {0,1} -> R` satisfying

`F(x) + F((x-1)/x) = 1+x`.

Submit the three rational functions in the orbit of `x` under
`T(x)=(x-1)/x`, the corresponding right-hand sides, the three rational
functions giving `F` on that orbit, and the exact integer coefficient matrix
whose nonsingularity establishes uniqueness. Rational functions are encoded
as primitive integer coefficient arrays in ascending degree order, with a
positive leading denominator coefficient. The verifier independently replays
the Möbius cycle and every rational-function equation. A copied final formula
without the orbit certificate cannot pass.

Write `submission.json` and digest-bind
`evidence/functional-equation-certificate.json`, which must copy `result`
and `limitations` exactly. Claim at most `COMPUTED`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** one of `UNIQUE_SOLUTION_CERTIFIED`, `INSUFFICIENT_EVIDENCE`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** one of `COMPLETE`, `PARTIAL`, `UNKNOWN`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/functional-equation-certificate.json`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `application/json`.
- **Required artifact filenames:** `evidence/functional-equation-certificate.json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
