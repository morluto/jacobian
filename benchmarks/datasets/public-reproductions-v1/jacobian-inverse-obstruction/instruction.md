# Polynomial-map collision witness

For the polynomial map and two rational points in `input.json`, verify that the points are distinct but share the same image, witnessing non-invertibility. Write `submission.json` to the exact agent-visible `submission_schema.json`, record the calculation in `evidence/answer.txt`, and include that file's SHA-256 digest in the evidence list. Claim `COMPUTED` assurance; do not claim `VERIFIED`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates the declared task-specific witness.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `text/plain`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
