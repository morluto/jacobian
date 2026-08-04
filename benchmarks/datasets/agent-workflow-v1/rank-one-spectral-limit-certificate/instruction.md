# Certify the limiting determinant root
Read `/app/input.json`. For the displayed `(n-1)×(n-1)` matrix, derive a diagonal-minus-rank-one decomposition, an exact partial-fraction identity for `1/(k^3-k)`, the resulting closed form for the unique determinant root, and its limit.

Choose at least three distinct checkpoint values of `n` in the declared bounds and report both the reduced reciprocal sum and determinant root at each checkpoint. Supply the partial-fraction scale and integer coefficients rather than a prose label. Any equivalent exact factorization of the identity is accepted.

Write `/app/submission.json` following the schema. The `root_formula` field must carry the exact rational expression for the determinant root as a function of `n`; the verifier checks the value independently.

Bind the identical result at `evidence/spectral-certificate.json` as a JSON object with exactly these fields: `schema_version` (the string `"1"`), `task_id` (the task identifier from the input), `result` (the same result object placed in `submission.json`), and `limitations` (the same limitations array). The evidence file must not exceed 1 MiB. Floating-point samples are not accepted as reward-bearing evidence, but private numerical exploration during research is permitted. Do not claim proof-assistant verification. Assurance is `COMPUTED`.
