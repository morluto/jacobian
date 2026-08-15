# Construct an optimal distinct-sum pairing

For the frozen ground set, construct as many disjoint unordered pairs as
possible so that all pair sums are distinct and at most `n`.

Submit the pairs in canonical increasing order, their sums, and the claimed
optimum. The verifier independently checks the witness and exhaustively solves
the finite optimization problem; it accepts any optimal pairing, not one
expected arrangement. Write `submission.json` to `submission_schema.json`,
explain the five-pair construction, its distinct sums, and the exhaustive
exclusion of a six-pair solution in `evidence/answer.txt`, and bind its SHA-256
digest.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates a witness only where the task needs one.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `text/plain`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
