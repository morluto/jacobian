# Derive and certify the Euler-line relation

Use the normalized triangle and restricted sparse rational-function format in
the offline input. Return exact coordinates for `O`, `G`, and `H`, together
with nonzero relation coefficients in the declared point order. The
coordinates must satisfy every defining identity and the submitted relation
as rational-function identities under the declared nonzero assumption.
Represent every submitted coefficient as an integer `numerator` and positive
integer `denominator`.

Write `submission.json` to the exact agent-visible `submission_schema.json`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
