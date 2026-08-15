# Audit an autoformalized orthogonality statement

Compare the frozen informal theorem, its reference formalization, and a proposed
Lean formalization. Determine whether the proposal preserves the dimension
premise and the meaning of the dot product.

Supply exact integer-vector certificates for both semantic defects:

1. a one-dimensional instance showing why omitting `k ≥ 2` changes the claim;
2. a two-dimensional instance with a nonzero vector orthogonal in the genuine
   dot-product sense, but not coordinatewise annihilated. Write `submission.json` to the exact agent-visible
`submission_schema.json`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
