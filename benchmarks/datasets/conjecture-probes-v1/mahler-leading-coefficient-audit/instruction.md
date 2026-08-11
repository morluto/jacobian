# Mahler-measure leading-coefficient audit

The frozen evaluator multiplies the moduli of roots outside the unit disk but silently assumes the polynomial is monic. Audit that formula for the supplied nonmonic reciprocal degree-eight polynomial.

Submit:

1. a factorization into exactly four primitive quadratic integer factors, ordered lexicographically by coefficient triple;
2. each factor's exact outside-root contribution in the basis `a+b*sqrt(5)`;
3. the flawed monic-normalized result;
4. the missing leading-coefficient multiplier; and
5. the corrected Mahler measure in the same radical basis.

The verifier multiplies the factors back to the frozen polynomial, classifies the two cyclotomic factors, checks the reciprocal quadratic roots algebraically, and recomputes all radical products. Equivalent factor signs are normalized by requiring positive leading coefficients and primitive coefficient gcd one.

This Assurance result concerns one exact polynomial and one formula defect only. It does not determine Lehmer's problem or compare all integer polynomials. Claim at most `CHECKED`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

Exact nonmonic Mahler-formula audit for one degree-eight polynomial only.

The evidence file must be a JSON object with exactly four fields: `schema_version` must be `"1"`, `task_id` must equal the submission task ID, `result` must be an exact JSON copy of `submission.json`'s result object, and `limitations` must be an exact JSON copy of its limitations list.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `NONMONIC_FORMULA_REPAIRED`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED`, `CHECKED` (ceiling `CHECKED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.json`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `application/json`.
- **Required artifact filenames:** `evidence/answer.json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
