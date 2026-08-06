# Certify a sharp cyclic vector inequality

For real `a_1,...,a_n`, `n>1`, determine the largest constant `C` for which
`sum_i sqrt(a_i^2 + (1-a_{i+1})^2) >= C n`, with cyclic indices.

Choose any certificate dimension from 5 through 12. Submit the full sparse-affine vector family used in the norm-sum reduction, its exact aggregate, a completed-square polynomial certificate for the lower bound, and an equality witness proving sharpness. The verifier reconstructs all symbolic coefficients and the equality case at the chosen dimension only; state the scope as the cyclic vector inequality at that chosen dimension `n`, not as a universal claim over all lengths. A bare constant, numerical sampling, or a non-sharp lower bound fails. Bind the result with exactly one `RESULT_JSON:` evidence line and do not claim proof-assistant verification.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `SHARP_CONSTANT_CERTIFIED`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** a string matching `^the cyclic vector inequality at dimension n = ([5-9]|1[012])$`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
