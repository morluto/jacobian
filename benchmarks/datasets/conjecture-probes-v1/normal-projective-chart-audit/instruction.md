# Ellipse-normal projective-chart audit

The frozen solver finds roots of the affine normal-incidence eliminant and concludes that the origin has exactly three normal footpoints on the ellipse. Audit the omitted projective chart boundary.

Submit the three finite roots of `t(1-t^2)`, their exact parametrized points, the missing homogeneous parameter `[T:U]=[1:0]`, its limiting ellipse point, and the complete four-point repaired footpoint list. For every point provide the exact ellipse residual and the exact normal-incidence residual

`4*y*h - x*k - 3*x*y`

at `(h,k)=(0,0)`. Rational values must be canonical strings. The verifier independently evaluates the finite parametrization, its homogeneous extension, the ellipse equation, and the normal condition. Ordering is canonical by point coordinates.

This Assurance audit certifies one chart-completeness defect on one ellipse. It does not establish any general concurrent-normals conjecture. Claim at most `COMPUTED`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates the declared task-specific witness.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/answer.json`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `application/json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
