# Derive and certify the Euler-line relation

Use the normalized triangle and restricted sparse rational-function format in
the offline input. Return exact coordinates for `O`, `G`, and `H`, together
with nonzero relation coefficients in the declared point order. The
coordinates must satisfy every defining identity and the submitted relation
as rational-function identities under the declared nonzero assumption.

Write `submission.json` to the exact agent-visible `submission_schema.json`.
Put a concise symbolic derivation in `evidence/answer.txt`, include a
`RESULT_JSON:` line containing the submitted result as JSON, and bind the file
with its SHA-256 digest.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `TRUE`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
