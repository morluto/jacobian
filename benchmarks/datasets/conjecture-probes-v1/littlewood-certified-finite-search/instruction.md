# Certify a finite Littlewood-product minimum

For every integer `1 <= n <= 2000`, consider
`n ||n sqrt(2)|| ||n sqrt(3)||`. Use rigorous rational lower and upper
enclosures obtained from the frozen 80-digit integer-square-root scale, never
floating point. Submit the complete strict record-minimum sequence. Each row
must include `n`, the two floor values, the two nearest integers, and canonical
rational lower/upper bounds for the product. Identify the final finite argmin.

The verifier independently reconstructs all 2000 enclosures and the complete
record sequence. Evidence is a matching JSON object (`application/json`) with
exactly `schema_version` (the string `"1"`), `task_id`, `result`, and
`limitations`.

This certifies only the frozen finite range for `(sqrt(2),sqrt(3))`; it does not
establish a liminf or any case of Littlewood's conjecture.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

Rigorous rational interval replay for a finite range only; no Littlewood conclusion.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `LITTLEWOOD_FINITE_MINIMUM_CERTIFICATE`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED`, `CHECKED` (ceiling `CHECKED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `application/json`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
