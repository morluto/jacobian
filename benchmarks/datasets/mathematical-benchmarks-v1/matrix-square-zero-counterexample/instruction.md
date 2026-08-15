# Square-zero matrix counterexample

Decide the universal matrix claim in the offline input. If it is false, return
one integer 2 by 2 matrix that satisfies every required counterexample
condition.

Write `submission.json` to the exact agent-visible `submission_schema.json`.
Show the complete matrix multiplication in `evidence/answer.txt`, include a
`RESULT_JSON:` line containing the submitted result as JSON, and include that
file's SHA-256 digest in the submission.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates a witness only where the task needs one.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `text/plain`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
