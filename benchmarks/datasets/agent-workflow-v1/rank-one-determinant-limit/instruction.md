# Certify a rank-one determinant limit

For each `n >= 2`, the frozen problem defines an `(n-1) x (n-1)` matrix whose
diagonal entry indexed by `i=2,...,n` is `i^3-i-lambda` and whose off-diagonal
entries are `-lambda`. Let `lambda_n` be its largest real determinant root.

Submit an exact certificate that:

- identifies the three rational coefficients in a partial-fraction
  decomposition of `1/(i^3-i)`;
- supplies at least six distinct, freely chosen sample sizes `n` in `[2,30]`,
  with the diagonal product, reciprocal sum, determinant constant and linear
  coefficients, and root `lambda_n`;
- gives an exact rational formula for `lambda_n - 4` using two affine factors
  in `n`, establishing the limit and hence the limsup.

The verifier independently applies the rank-one determinant identity and exact
integer/rational arithmetic. Numerical eigenvalues and floating-point limits
are not accepted. Write `submission.json` and digest-bind
`evidence/determinant-certificate.json`. Claim at most `COMPUTED`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `LIMSUP_EQUALS_FOUR`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/determinant-certificate.json`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `application/json`.
- **Required artifact filenames:** `evidence/determinant-certificate.json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
