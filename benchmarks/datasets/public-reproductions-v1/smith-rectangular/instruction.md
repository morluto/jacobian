# Smith normal form of a rectangular integer matrix

Compute the exact Smith normal form of the integer matrix in `input.json`. Return the rank and the list of nonzero invariant factors in canonical order. Write `submission.json` to the exact agent-visible `submission_schema.json`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result. Integer fields denote mathematical values: 2, 2.0, and 2e0 are equivalent and parsed exactly. Numeric tokens are limited to 4300 characters, expanded integers to 4300 digits, and decimal exponents to [-4300,4300] before conversion; booleans, strings, and nonintegral numbers are invalid integers. The complete UTF-8 submission file, including whitespace, is limited to 16777216 bytes.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
