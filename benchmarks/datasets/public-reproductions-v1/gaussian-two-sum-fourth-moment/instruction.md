# Gaussian polynomial moment

Compute the exact fixed-order complex Gaussian moment of the polynomial in `input.json` with respect to independent standard complex Gaussian variables. Return the rational real and imaginary parts of the moment. Write `submission.json` to the exact agent-visible `submission_schema.json`. A finite list of checked moments is not an all-order identity; claim `COMPUTED` assurance only and do not claim `VERIFIED`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates the declared task-specific witness.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `text/plain`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
