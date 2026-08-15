# Four-subspace direct-sum counterexample

Decide the claim in the offline input. If false, give one nonzero integer
generator for each of four one-dimensional subspaces of `Q^3` and nonzero
integer coefficients witnessing a linear dependence among the four
generators.

The verifier will check every ordered choice of three distinct indices and
will independently replay the dependence. Write `submission.json` to the exact
agent-visible `submission_schema.json`, include a `RESULT_JSON:` line containing
the submitted result as JSON in `evidence/answer.txt`, and bind that file by
SHA-256.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates a witness only where the task needs one.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `text/plain`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
