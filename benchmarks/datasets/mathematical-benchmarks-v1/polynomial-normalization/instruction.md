# Polynomial normalization

Combine like terms in the exact sparse polynomial from `input.json`. Return
canonical rational coefficients as integer `numerator`/positive integer
`denominator` objects and exponent vectors, omitting zero terms.
Record the cancellation and resulting terms in `evidence/answer.txt`, and use
its SHA-256 digest in the evidence list. Write `submission.json` to the exact
agent-visible `submission_schema.json`.
<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates the declared task-specific witness.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `text/plain`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
