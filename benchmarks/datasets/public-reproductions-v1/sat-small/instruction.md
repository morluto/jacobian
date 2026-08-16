# SAT decision with witness

Decide the exact CNF formula in `input.json`. For a satisfiable formula, return `SATISFIABLE` and a complete Boolean assignment satisfying every clause. For an unsatisfiable formula, return `UNSATISFIABLE`. Write `submission.json` to the exact agent-visible `submission_schema.json`. Claim `COMPUTED` assurance; do not claim `VERIFIED`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
