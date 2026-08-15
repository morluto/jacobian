# Audit shallow Lean axiom reports

The frozen input contains two declaration-dependency graphs and the axiom sets
reported by the affected Lean collector. For each case, reconstruct the full
transitive dependency closure from the declared roots, compare it with the
observed report, and list the missing dependencies in sorted order.

Classify each report as `COMPLETE` or `INCOMPLETE`. Classify every missing
dependency by its frozen role and write `/app/submission.json`. The checker audits only the frozen graphs

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
