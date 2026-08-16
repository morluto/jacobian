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

checks an exact countermodel family but does not elaborate Lean or prove the
informal minimum.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
