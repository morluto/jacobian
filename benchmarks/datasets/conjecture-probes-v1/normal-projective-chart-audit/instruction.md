# Ellipse-normal projective-chart audit

The frozen solver finds roots of the affine normal-incidence eliminant and concludes that the origin has exactly three normal footpoints on the ellipse. Audit the omitted projective chart boundary.

Submit the three finite roots of `t(1-t^2)`, their exact parametrized points, the missing homogeneous parameter `[T:U]=[1:0]`, its limiting ellipse point, and the complete four-point repaired footpoint list. For every point provide the exact ellipse residual and the exact normal-incidence residual

`4*y*h - x*k - 3*x*y`

at `(h,k)=(0,0)`. Rational values are `{numerator, denominator}` objects. Equivalent encodings such as `2/2` and `1` are accepted after exact `Fraction` normalization. The verifier independently evaluates the finite parametrization, its homogeneous extension, the ellipse equation, and the normal condition. Ordering is canonical by point coordinates.

This Assurance audit certifies one chart-completeness defect on one ellipse. It does not establish any general concurrent-normals conjecture.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
