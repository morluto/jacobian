# Repair a binary fractional-ratio proof

The frozen problem maximizes `(alpha + t·x)/(beta + f·x)` over binary vectors. The public proof instead analyzes a linear benefit objective with a budget constraint and fractional decision variables.

Identify the three contract mismatches exactly. Then solve the frozen 24-item instance and submit an exact residual optimality certificate. If the attained reduced ratio is `p/q`, report every residual `q*t_i - p*f_i`, the affine constant `q*alpha - p*beta`, the indices with positive residual, and the maximum residual sum obtained by independently choosing each binary coordinate. A zero maximum residual proves no binary vector exceeds `p/q`; the submitted vector must attain equality.

Write `/app/submission.json`. The verifier replays the typed residual certificate directly; no prose explanation or duplicate artifact is required.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
