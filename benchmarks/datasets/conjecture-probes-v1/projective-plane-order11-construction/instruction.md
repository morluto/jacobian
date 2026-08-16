# Construct a projective plane of order 11

Label the 133 points by integers 0 through 132 and submit 133 lines. Each line must contain exactly 12 distinct point labels. Every unordered pair of distinct points must occur on exactly one submitted line.

Line order and point order within a line are irrelevant. The verifier checks the complete incidence predicate and accepts every labeled projective plane satisfying it; no finite-field presentation is required.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
