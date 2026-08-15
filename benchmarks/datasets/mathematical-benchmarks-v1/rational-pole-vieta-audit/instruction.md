# Audit a rational-equation root-sum proof

The frozen trace claims that the poles of
`sum_{k=1}^4 k/(x^2-k) = 2010x-4` are `1,2,3,4`. Diagnose that step and repair
the computation of the sum of all complex solutions.

Submit `/app/submission.json` following `/app/submission_schema.json` and a
digest-bound JSON envelope at `/app/evidence/pole-vieta-certificate.json`.
The envelope must contain exactly `schema_version` (value `"1"`), `task_id`,
`result`, and `limitations`, with the latter three matching the submission.
Coefficient arrays are low-to-high. Provide the common denominator, combined
numerator, cleared polynomial, the value of the surviving numerator at each
denominator square value `k=1,2,3,4`, and the resulting root sum. The verifier
reconstructs the rational equation and checks that clearing denominators
introduced no pole roots.

Do not claim proof-assistant verification. Claim `COMPUTED` assurance and
complete scope.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier independently reconstructs and checks the exact polynomial and pole-domain certificate.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/pole-vieta-certificate.json`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `application/json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
