# Graph counterexample discovery

The offline input describes a universal claim about finite simple graphs.
Decide whether the claim holds. If it is false, return one explicit graph that
satisfies every stated hypothesis and violates the conclusion.

Write `submission.json` to the exact agent-visible `submission_schema.json`.
Put a short calculation or independently replayable witness in
`evidence/answer.txt`, and include that file's SHA-256 digest in the submission.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates the declared task-specific witness.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `text/plain`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
