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

The verifier replays the task-specific mathematical predicate from the submitted result and validates a witness only where the task needs one.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `text/plain`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
