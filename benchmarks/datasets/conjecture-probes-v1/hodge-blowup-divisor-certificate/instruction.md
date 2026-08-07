# Certify an algebraic divisor class on a six-point blow-up

Submit a nonzero homogeneous cubic over `ZZ`, in the frozen ten-monomial
order, that vanishes simply at all six frozen points. For each point report the
polynomial value and three first partial derivatives. At least one derivative
must be nonzero, certifying multiplicity exactly one.

For the strict-transform divisor class `D=3H-E1-...-E6`, report its class
vector, `D^2`, `D·K` for `K=-3H+E1+...+E6`, and the adjunction arithmetic
genus. The verifier recomputes all evaluations and intersection arithmetic.

The coefficients must be primitive (i.e., their GCD must be 1); scalar multiples
are rejected. Evidence is matching JSON with exactly `schema_version`, `task_id`, `result`,
and `limitations`. Lefschetz (1,1) is a declared trusted theorem.
This one divisor certificate does not address higher-codimension Hodge classes
or prove the Hodge Conjecture.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

Exact divisor-class certificate under trusted Lefschetz (1,1); no general Hodge conclusion.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `ALGEBRAIC_DIVISOR_CLASS_CERTIFICATE`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED`, `CHECKED` (ceiling `CHECKED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `application/json`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
