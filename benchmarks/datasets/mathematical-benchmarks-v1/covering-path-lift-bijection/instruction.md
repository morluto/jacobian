# Construct a fiber bijection by unique path lifting

The frozen input gives a finite graph covering and two base vertices. Choose
any simple base path satisfying the declared length bounds. Lift that path from
every point in the source fiber, then lift the reversed path from every
resulting endpoint. Submit the complete forward and reverse lift traces and
the induced fiber bijection.

Your certificate must show that every trace projects to its declared base
path, every step is a cover edge, each lift is the unique available continuation,
and reverse lifting returns every source point. Merely reporting equal fiber
cardinalities is incomplete.

The verifier checks this finite covering exactly but does not certify the
general topological theorem.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
