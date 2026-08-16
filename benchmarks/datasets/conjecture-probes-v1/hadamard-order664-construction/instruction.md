# Construct a Hadamard matrix of order 664

Submit 664 binary strings of length 664. Decode each 1 as +1 and each 0 as -1; the resulting rows must form a Hadamard matrix, so every two distinct rows have integer dot product zero.

Row and column order, row and column signs, and normalization are unrestricted. The verifier parses the entire represented matrix and checks every row pair exactly using bit counts.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
