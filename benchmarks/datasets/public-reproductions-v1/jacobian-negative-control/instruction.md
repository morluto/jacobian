# Negative control for a polynomial-map collision

The `claimed_image` in `input.json` has been mutated away from the true common image. Determine whether the two points still share that claimed image. If they do not, return `UNKNOWN` with a short boundary note. Write `submission.json` to the exact agent-visible `submission_schema.json`, record the calculation in `evidence/answer.txt`, and include that file's SHA-256 digest in the witness list.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates the declared task-specific witness.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `text/plain`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
