# Audit a Chebotarev density solution

Audit the frozen solution for `f(x)=x^4-4x+1`. Produce the exact mod-2 factorization certificate, the actual integer discriminant, and a complete `S4` cycle-type table with class size, fixed-point count, and whether the class contributes to the root-mod-p density. Compute the corrected fixed-point proportion and encoded answer.

Write `/app/submission.json` and `/app/evidence/chebotarev-audit.json` according to the schema. The density calculation is explicitly conditional on the frozen premise `Gal(f)=S4`; do not claim to prove that classification or Chebotarev. Claim only `COMPUTED`.

The digest-bound evidence file `evidence/chebotarev-audit.json` must be a JSON object with exactly four keys: `schema_version` (the string `"1"`), `task_id` (the task identifier), `result` (the same result object placed in `submission.json`), and `limitations` (the same limitations list placed in `submission.json`). The evidence file must not exceed 16 MiB.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** one of `PUBLISHED_SOLUTION_CONTAINS_MULTIPLE_FATAL_ERRORS`, `PUBLISHED_SOLUTION_CORRECT`, `INSUFFICIENT_EVIDENCE`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** one of `COMPLETE`, `PARTIAL`, `UNKNOWN`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/chebotarev-audit.json`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `application/json`.
- **Required artifact filenames:** `evidence/chebotarev-audit.json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
