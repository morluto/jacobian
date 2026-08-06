# Audit a minimum-rank completion argument

The frozen proof makes two claims:

1. support in `|i-j|<3` makes the pattern symmetric; and
2. because zero positions are free, choosing a full-rank submatrix forces a
   lower bound on the minimum completion rank.

Submit an exact GF(2) certificate that refutes both steps. Choose a dimension
from 8 through 14, give an asymmetric binary 3-pattern with at least `n+1`
forced ones, and provide two complete binary matrices satisfying `A ∘ M = M`:
one of rank exactly 1 and one of rank exactly `n`.

Write the certificate to `/app/submission.json` and bind an evidence copy at
`evidence/matrix-completion.json`. The evidence file must be a JSON object with
exactly the fields `schema_version` (the string `"1"`), `task_id` (the same
`task_id` string placed in the submission), `result` (the same
`result` object placed in the submission), and `limitations` (the same
`limitations` array placed in the submission). It is not the full submission
envelope. Assurance is capped at `COMPUTED`.

The evidence descriptor in `submission.json` must use `{"path": "evidence/matrix-completion.json", "sha256": "sha256:<hex>"}` where `<hex>` is the SHA-256 digest of the exact on-disk file bytes (not canonicalized JSON).

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** one of `MIN_RANK_ARGUMENT_REFUTED`, `INSUFFICIENT_EVIDENCE`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** one of `COMPLETE`, `PARTIAL`, `UNKNOWN`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/matrix-completion.json`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `application/json`.
- **Required artifact filenames:** `evidence/matrix-completion.json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
