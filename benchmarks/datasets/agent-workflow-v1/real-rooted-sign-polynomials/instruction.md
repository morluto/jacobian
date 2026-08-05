# Classify real-rooted sign polynomials

Classify every nonconstant polynomial whose coefficients all belong to
`{-1,1}` and whose complex roots are all real, including overall sign
changes.

Submit a universal degree-bound certificate based on monic normalization,
Newton's second power-sum identity, the root product, and AM-GM. Then submit a
complete audit of every sign-coefficient polynomial in the remaining degrees
1 through 3. Coefficients are in ascending degree order. For quadratics and
cubics, include the exact discriminant and the resulting real-rooted decision.
The verifier independently enumerates the whole finite residue class and
recomputes every discriminant; a copied final list is insufficient.

Write `submission.json` and digest-bind
`evidence/classification-certificate.json`. Claim at most `COMPUTED`.

The digest-bound evidence file must be a JSON object with exactly four keys:
`schema_version` (the string `"1"`), `task_id` (the task identifier),
`result` (the same result object placed in `submission.json`), and
`limitations` (the same limitations list placed in `submission.json`).

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** one of `CLASSIFICATION_CERTIFIED`, `INSUFFICIENT_EVIDENCE`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** one of `COMPLETE`, `PARTIAL`, `UNKNOWN`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/classification-certificate.json`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `application/json`.
- **Required artifact filenames:** `evidence/classification-certificate.json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
