# Exact rational linear solution

Solve the exact linear system in `input.json` over the rationals. Return one structured `{numerator, denominator}` object for each declared variable. Equivalent encodings such as `34/14` and `17/7` are accepted after exact `Fraction` normalization. Write `submission.json` to the exact agent-visible schema.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
