# Classify an exact swapped polynomial one-form

Read `/app/input.json`. Let `f` have the ten coefficients in the declared order. Classify exactly those degree-at-most-three polynomials for which

`f(x,y) dx + f(y,x) dy`

has zero integral around every closed smooth curve in `R2`.

The frozen source solution claims that closedness is `f_y(x,y)=f_x(y,x)`. Audit the chain rule rather than trusting that claim.

Write `/app/submission.json` matching `/app/submission_schema.json` and `/app/evidence/answer.txt`. Submit two independent integer row constraints on the ten coefficients, their rank and the resulting dimension, eight integer coefficient vectors forming a basis of the solution space, and one polynomial potential for each basis vector. A potential `F` must satisfy `F_x=f(x,y)` and `F_y=f(y,x)` exactly. Rational coefficients use canonical strings such as `1/2` or `-3`; combine like terms and omit zero terms.

The evidence file must contain exactly these six labeled lines (the text after each label may use equivalent wording while preserving the stated facts): `CHAIN_RULE:` with the corrected derivative `d/dx f(y,x)=f_y(y,x)`; `CONSTRAINTS:` with `a_11-2*a_02=0` and `a_21-3*a_03=0`; `RANK: 2`; `DIMENSION: 8`; `POTENTIALS:` stating that every listed potential has both required derivatives; and `LIMITATION:` stating that the analytic Poincare lemma and arbitrary smooth forms are not checked. Do not add unrelated lines.

The verifier independently derives the correctly chained closedness constraints, checks row-space equality rather than a fixed presentation, performs exact rank and nullspace tests, and differentiates every potential. It does not assess arbitrary smooth forms or prove the Poincare lemma; assurance must remain `COMPUTED`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `SOURCE_CHAIN_RULE_REPAIRED`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** a string value
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
