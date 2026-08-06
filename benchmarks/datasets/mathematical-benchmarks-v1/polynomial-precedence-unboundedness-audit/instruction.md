# Polynomial-precedence semantic audit

The frozen input contains an informal minimization problem and a proposed formal
translation. Audit whether the formal polynomial preserves the informal claim.

Submit `/app/submission.json` matching the visible schema. Your certificate must
give rational polynomials

`x(t) = x0 + x1*t + x2*t^2` and `y(t) = y0 + y1*t`

such that substituting them into the formal polynomial produces the submitted
coefficient list in ascending powers of `t`, has degree at least two, and has a
strictly negative leading coefficient. Give four distinct integer checkpoints
with the exact substituted values. The verifier independently performs the
symbolic substitution and checks the checkpoints; valid alternative families
are accepted.

Write one digest-bound evidence object at
`/app/evidence/precedence-audit.json`. It must contain exactly
`schema_version`, `task_id`, `result`, and `limitations`, with the latter three
equal to the submission. The only supported assurance is `COMPUTED`: this task
checks an exact countermodel family but does not elaborate Lean or prove the
informal minimum.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

Exact rational symbolic replay; Lean elaboration and the informal minimum remain outside scope.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** one of `FORMALIZATION_CHANGES_SEMANTICS`
- **Assurance:** scoreable values are `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/precedence-audit.json`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `application/json`.
- **Required artifact filenames:** `evidence/precedence-audit.json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
