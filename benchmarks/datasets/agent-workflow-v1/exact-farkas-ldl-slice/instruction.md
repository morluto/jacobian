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
