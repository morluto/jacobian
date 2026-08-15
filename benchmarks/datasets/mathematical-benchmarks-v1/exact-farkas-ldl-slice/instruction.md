# Audit an exact rational Farkas-certificate slice

The frozen input contains scalar values and a 4×4 principal submatrix extracted
from the canonical exact rational Farkas certificate in the Bandeira 0.2a
repository. Replay the scalar identity `m00 = y0 + c00_y`, establish the two
sign checks `m00 < 0` and `objective > 0`, and certify that this matrix is
positive definite using either:

- `LDL`: an exact unit-lower-triangular `L` and positive diagonal `D`
  satisfying `Q=L D L^T`; or
- `SYLVESTER`: all four exact positive leading principal determinants.

Use reduced rational objects throughout. The result must remain explicitly
local: this slice does not establish positivity of every 31×31 block, the full
Farkas certificate, the Lean formalization, or the underlying theorem.
Floating-point eigenvalues and discovery artifacts are not proof evidence.

Write `submission.json` and digest-bind
`evidence/farkas-slice-certificate.json`.

The digest-bound task-specific witness file must be a JSON object with exactly three keys:
`schema_version` (the string `"1"`), `task_id` (the task identifier),
`result` (the same result object placed in `submission.json`).

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates the declared task-specific witness.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/farkas-slice-certificate.json`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `application/json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
