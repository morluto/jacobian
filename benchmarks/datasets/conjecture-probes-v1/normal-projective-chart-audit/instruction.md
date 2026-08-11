# Ellipse-normal projective-chart audit

The frozen solver finds roots of the affine normal-incidence eliminant and concludes that the origin has exactly three normal footpoints on the ellipse. Audit the omitted projective chart boundary.

Submit the three finite roots of `t(1-t^2)`, their exact parametrized points, the missing homogeneous parameter `[T:U]=[1:0]`, its limiting ellipse point, and the complete four-point repaired footpoint list. For every point provide the exact ellipse residual and the exact normal-incidence residual

`4*y*h - x*k - 3*x*y`

at `(h,k)=(0,0)`. Rational values must be canonical strings. The verifier independently evaluates the finite parametrization, its homogeneous extension, the ellipse equation, and the normal condition. Ordering is canonical by point coordinates.

This Assurance audit certifies one chart-completeness defect on one ellipse. It does not establish any general concurrent-normals conjecture. Claim at most `CHECKED`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

Exact affine-versus-projective normal-footpoint audit for one ellipse only.

The evidence file must be a JSON object with exactly four fields: `schema_version` must be `"1"`, `task_id` must equal the submission task ID, `result` must be an exact JSON copy of `submission.json`'s result object, and `limitations` must be an exact JSON copy of its limitations list.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `AFFINE_NORMAL_COUNT_IS_INCOMPLETE_AND_REPAIRED`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED`, `CHECKED` (ceiling `CHECKED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.json`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `application/json`.
- **Required artifact filenames:** `evidence/answer.json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
