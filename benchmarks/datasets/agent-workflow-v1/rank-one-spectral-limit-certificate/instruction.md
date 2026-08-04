# Certify the limiting determinant root
Read `/app/input.json`. For the displayed `(n-1)×(n-1)` matrix, derive a diagonal-minus-rank-one decomposition, an exact partial-fraction identity for `1/(k^3-k)`, the resulting closed form for the unique determinant root, and its limit.

Choose at least three distinct checkpoint values of `n` in the declared bounds and report both the reduced reciprocal sum and determinant root at each checkpoint. Supply the partial-fraction coefficients rather than a prose label.

Write `/app/submission.json` following the schema and bind the identical result at `evidence/spectral-certificate.json`. Do not use floating-point eigenvalue sampling or claim proof-assistant verification. Assurance is `COMPUTED`.
