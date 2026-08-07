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
Claim `CHECKED` only for this order-12 certificate.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks one complete normalized order-12 matrix; this finite construction does not prove the general conjecture.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `HADAMARD_ORDER12_CONSTRUCTION`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED`, `CHECKED` (ceiling `CHECKED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `application/json`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
