# Audit an exact rational Farkas-certificate slice

The frozen input contains scalar values and a 4×4 principal submatrix extracted
from the canonical exact rational Farkas certificate in the Bandeira 0.2a
repository. Establish the two scalar sign checks and certify that this matrix
is positive definite using either:

- `LDL`: an exact unit-lower-triangular `L` and positive diagonal `D`
  satisfying `Q=L D L^T`; or
- `SYLVESTER`: all four exact positive leading principal determinants.

Use reduced rational objects throughout. The result must remain explicitly
local: this slice does not establish positivity of every 31×31 block, the full
Farkas certificate, the Lean formalization, or the underlying theorem.
Floating-point eigenvalues and discovery artifacts are not proof evidence.

Write `submission.json` and digest-bind
`evidence/farkas-slice-certificate.json`. Claim at most `COMPUTED`.

The digest-bound evidence file must be a JSON object with exactly four keys:
`schema_version` (the string `"1"`), `task_id` (the task identifier),
`result` (the same result object placed in `submission.json`), and
`limitations` (the same limitations list placed in `submission.json`).

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** one of `LOCAL_SLICE_CERTIFIED`, `INSUFFICIENT_EVIDENCE`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** one of `COMPLETE`, `PARTIAL`, `UNKNOWN`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/farkas-slice-certificate.json`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `application/json`.
- **Required artifact filenames:** `evidence/farkas-slice-certificate.json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
