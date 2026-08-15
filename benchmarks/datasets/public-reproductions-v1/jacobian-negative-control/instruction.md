# Negative control for a polynomial-map collision

The `claimed_image` in `input.json` has been mutated away from the true common image. Determine those two conclusions independently: whether both points map to that claimed image, and whether the two distinct points have the same actual image. A collision of actual images verifies non-invertibility even when the claimed image is wrong. Return those two conclusions as booleans. Write `submission.json` to the exact agent-visible `submission_schema.json`, record the calculation and the bounded `UNKNOWN` conclusion in `evidence/answer.txt`, and include that file's SHA-256 digest in the witness list.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates the declared task-specific witness.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `text/plain`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
