# Audit a minimum-rank completion argument

The frozen proof makes two claims:

1. support in `|i-j|<3` makes the pattern symmetric; and
2. because zero positions are free, choosing a full-rank submatrix forces a
   lower bound on the minimum completion rank.

Submit an exact GF(2) certificate that refutes both steps. Choose a dimension
from 8 through 14, give an asymmetric binary 3-pattern with at least `n+1`
forced ones, and provide two complete binary matrices satisfying `A ∘ M = M`:
one of rank exactly 1 and one of rank exactly `n`.

Write the certificate to `/app/submission.json` and bind an task-specific witness copy at
`evidence/matrix-completion.json`. The task-specific witness file must be a JSON object with
exactly the fields `schema_version` (the string `"1"`), `task_id` (the same
`task_id` string placed in the submission), `result` (the same

The witness descriptor in `submission.json` must use `{"path": "evidence/matrix-completion.json", "sha256": "sha256:<hex>"}` where `<hex>` is the SHA-256 digest of the exact on-disk file bytes (not canonicalized JSON).

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates the declared task-specific witness.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/matrix-completion.json`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `application/json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
