# Audit a rational-equation root-sum proof

The frozen trace claims that the poles of
`sum_{k=1}^4 k/(x^2-k) = 2010x-4` are `1,2,3,4`. Diagnose that step and repair
the computation of the sum of all complex solutions.

Submit `/app/submission.json` following `/app/submission_schema.json` and a
Coefficient arrays are low-to-high. Provide the common denominator, combined
numerator, cleared polynomial, the value of the surviving numerator at each
denominator square value `k=1,2,3,4`, and the resulting root sum. The verifier
reconstructs the rational equation and checks that clearing denominators
introduced no pole roots.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
