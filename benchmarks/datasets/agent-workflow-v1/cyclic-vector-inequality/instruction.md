# Certify a sharp cyclic vector inequality

For real `a_1,...,a_n`, `n>1`, determine the largest constant `C` for which
`sum_i sqrt(a_i^2 + (1-a_{i+1})^2) >= C n`, with cyclic indices.

Choose any certificate dimension from 5 through 12. Submit the full sparse-affine vector family used in the norm-sum reduction, its exact aggregate, a completed-square polynomial certificate for the lower bound, and an equality witness proving sharpness. The verifier reconstructs all symbolic coefficients and the equality case at the chosen dimension only; state the scope as the cyclic vector inequality at that chosen dimension `n`, not as a universal claim over all lengths. A bare constant, numerical sampling, or a non-sharp lower bound fails. Bind the result with exactly one `RESULT_JSON:` evidence line and do not claim proof-assistant verification.
