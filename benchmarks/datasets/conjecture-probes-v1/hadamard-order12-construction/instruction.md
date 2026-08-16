# Construct and certify a normalized Hadamard matrix of order 12

Construct a `12 × 12` matrix with entries in `{−1, 1}` whose first row and
first column are all `1`. Supply the complete matrix, its complete integer Gram
matrix `H H^T`, and the exact signed determinant.

The verifier independently recomputes matrix dimensions, entries,
normalization, every Gram entry, and the determinant using exact integer
arithmetic. Any normalized order-12 Hadamard matrix is accepted; the matrix is
not required to match a particular Paley presentation.

This finite construction is evidence for one admissible order only. It does
not prove the Hadamard matrix conjecture for every positive multiple of four.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
