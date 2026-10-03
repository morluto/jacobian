# Audit a minimum-rank completion argument

The frozen proof makes two claims:

1. support in `|i-j|<3` makes the pattern symmetric; and
2. because zero positions are free, choosing a full-rank submatrix forces a
   lower bound on the minimum completion rank.

Submit an exact GF(2) certificate that refutes both steps. Choose a dimension
from 8 through 14, give an asymmetric binary 3-pattern with at least `n+1`
forced ones, and provide two complete binary matrices satisfying `A ∘ M = M`:
one of rank exactly 1 and one of rank exactly `n`.

`task_id` string placed in the submission), `result` (the same

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

Choose an integer `dimension` from 8 through 14. Each of `pattern`, `low_rank_completion`, and `full_rank_completion` must have exactly `dimension` rows and exactly `dimension` entries in every row. All entries must be binary integers (0 or 1). The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
