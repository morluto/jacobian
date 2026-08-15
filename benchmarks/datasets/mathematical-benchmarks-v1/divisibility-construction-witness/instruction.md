# Construct a divisibility witness

Find any positive integers `a` and `b` within the offline input's search scope
that satisfy both divisibility conditions. Return the witness and the exact
arithmetic certificate requested by the schema.

Write `submission.json` to the exact agent-visible `submission_schema.json`.
Put a concise exact derivation in `evidence/answer.txt`, include a
`RESULT_JSON:` line containing the submitted result as JSON, and bind that file
with its SHA-256 digest.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates a witness only where the task needs one.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `text/plain`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
