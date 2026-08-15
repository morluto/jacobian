# Construct an optimal distinct-sum pairing

For the frozen ground set, construct as many disjoint unordered pairs as
possible so that all pair sums are distinct and at most `n`.

Submit the pairs in canonical increasing order, their sums, and the claimed
optimum. The verifier independently checks the submitted pairing and
exhaustively solves the finite optimization problem; it accepts any optimal
pairing, not one expected arrangement.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
