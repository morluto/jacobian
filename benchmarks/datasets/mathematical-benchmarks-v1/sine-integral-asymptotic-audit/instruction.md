# Audit a published sine-integral asymptotic expansion

The frozen dataset answer claims

`Si(x) = PI/2 - cos(x)/x + sin(x)/x^2 + ...`.

Audit its signs without numerical sampling. Submit an exact certificate obtained from the tail
`J(x) = integral_x^infinity sin(t)/t dt`. Give five terms of `J`, the corresponding five terms of `Si = PI/2 - J`, and the single integral remainder after five integrations by parts. Your certificate must be independently checkable by differentiating the proposed tail identity.

Also give a rigorous absolute bound for the scaled remainder, using
`|integral_x^infinity cos(t)/t^6 dt| <= 1/(5*x^5)` for `x > 0`, and classify whether the published coefficient of `sin(x)/x^2` is correct.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
