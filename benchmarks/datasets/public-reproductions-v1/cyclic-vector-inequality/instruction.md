# Certify a sharp cyclic vector inequality

For real `a_1,...,a_n`, `n>1`, determine the largest constant `C` for which
`sum_i sqrt(a_i^2 + (1-a_{i+1})^2) >= C n`, with cyclic indices.

Choose any certificate dimension from 5 through 12. Submit the full sparse-affine vector family used in the norm-sum reduction, its exact aggregate, a completed-square polynomial certificate for the lower bound, and an equality witness proving sharpness. The verifier reconstructs all symbolic coefficients and the equality case at the chosen dimension only; state the scope as the cyclic vector inequality at that chosen dimension `n`, not as a universal claim over all lengths. A bare constant, numerical sampling, or a non-sharp lower bound fails. Do not claim proof-assistant verification.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
