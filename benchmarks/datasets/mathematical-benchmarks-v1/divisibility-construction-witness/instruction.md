# Construct a divisibility witness

Find any positive integers `a` and `b` within the offline input's search scope
that satisfy both divisibility conditions. Return the witness and the exact
arithmetic certificate requested by the schema.

Write `submission.json` to the exact agent-visible `submission_schema.json`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
