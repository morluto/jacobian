# Factor and certify reciprocal symmetry

The frozen input gives a degree-16 integer polynomial by coefficients in constant-to-leading order. Recover a complete factorization

`leading_coefficient * product(Phi_order(x)^multiplicity)`

using cyclotomic orders from 2 through 30. Submit the ordered factor list, the independently expanded coefficient vector, its reciprocal vector, `P(1)`, and the reciprocal scalar. Explain why excluding `Phi_1=x-1` corresponds to `P(1) != 0` and why inversion of root-of-unity orbits produces coefficient symmetry.

The verifier reconstructs cyclotomic polynomials and the entire product; coefficient-pattern recognition alone is insufficient. Do not claim `VERIFIED` because no proof assistant certifies the unrestricted source theorem.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** one of `CYCLOTOMIC_RECIPROCITY_CERTIFIED`, `INSUFFICIENT_EVIDENCE`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** a string value
- **Completeness:** one of `COMPLETE`, `PARTIAL`, `UNKNOWN`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
