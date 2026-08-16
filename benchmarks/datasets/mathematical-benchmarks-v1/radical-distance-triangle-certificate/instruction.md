# Recover an exact geometric lower-bound certificate

The frozen expression is a sum of two radicals in arbitrary real `a,b`. Select a declared proof method and submit an exact distance-model certificate: both fixed centers as rational coefficients of `sqrt(2)`, the independently expanded radicand coefficient records, the squared distance between the centers, the universal lower bound obtained from the triangle inequality, and an exact equality witness proving sharpness.

Each `scaled_centers` entry is an integer pair `[p, q]` encoding the center coordinate `(p/2) * sqrt(2), (q/2) * sqrt(2)`; the two centers may be listed in either order. The verifier derives each radicand coefficient record from the submitted center and compares the resulting multiset against the frozen expression, so the expanded radicands must be paired with their corresponding centers but need not follow a fixed ordering.

Sampling is not a universal proof.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
