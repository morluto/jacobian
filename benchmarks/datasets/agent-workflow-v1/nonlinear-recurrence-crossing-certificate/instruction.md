# Certify a negative recurrence term

For the total real sequence `a₁=56`, `aₙ₊₁=aₙ-1/aₙ`, prove that some
`aₙ<0` with `n<2002`. Submit an exact certificate based on `dₙ=aₙ²`: give the
potential identity coefficients in the Laurent basis `[a², 1, a⁻²]`, a
rational threshold phase, its exact minimal step budget, and three exact
open-interval images under `a -> a-1/a` that force a negative term. The three
interval certificates may be listed in any order and are checked from their
rational endpoints rather than from prose labels. Use reduced rational
numerator/denominator objects; numerical simulation is not accepted.

Write `submission.json` and digest-bind
`evidence/nonlinear-recurrence-certificate.json`, which must copy `result` and
`limitations` exactly. Claim at most `COMPUTED`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** one of `NEGATIVE_TERM_BEFORE_2002`, `INSUFFICIENT_EVIDENCE`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** one of `COMPLETE`, `PARTIAL`, `UNKNOWN`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/nonlinear-recurrence-certificate.json`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `application/json`.
- **Required artifact filenames:** `evidence/nonlinear-recurrence-certificate.json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
