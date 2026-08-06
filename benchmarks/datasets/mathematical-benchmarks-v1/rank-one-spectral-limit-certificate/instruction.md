# Certify the limiting determinant root
Read `/app/input.json`. For the displayed `(n-1)×(n-1)` matrix, derive a diagonal-minus-rank-one decomposition, an exact partial-fraction identity for `1/(k^3-k)`, the resulting closed form for the unique determinant root, and its limit.

Choose at least three distinct checkpoint values of `n` in the declared bounds and report both the reduced reciprocal sum and determinant root at each checkpoint. Supply the partial-fraction coefficients rather than a prose label.

Write `/app/submission.json` following the schema and bind the identical result at `evidence/spectral-certificate.json`. Do not use floating-point eigenvalue sampling or claim proof-assistant verification. Assurance is `COMPUTED`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier independently replays exact rational sums and freely selected checkpoints; the determinant lemma and rational limit inference remain trusted.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `LIMIT_DERIVED_EXACTLY`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/spectral-certificate.json`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `application/json`.
- **Required artifact filenames:** `evidence/spectral-certificate.json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
