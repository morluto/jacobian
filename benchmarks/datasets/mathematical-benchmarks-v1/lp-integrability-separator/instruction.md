Construct a nonnegative measurable function on `(0,+infinity)` that belongs to
`L^2` and to no other `L^p` for positive finite `p`.

Use the frozen two-tail family

`f(x) = x^(-1/2) (log(1/x))^(-beta)` for `0 < x < e^(-1)`,

`f(x) = 0` for `e^(-1) <= x <= e`, and

`f(x) = x^(-1/2) (log x)^(-beta)` for `x > e`,

but choose your own structured rational `beta > 1/2`. Submit exact transformed
`p=2` integrals and a regime certificate for the origin obstruction at every
`p>2` and infinity obstruction at every `0<p<2`. Do not use numeric sampling.

Every rational field (`beta`, `p2_log_exponent`, `p2_integral_each`) is a
`{numerator, denominator}` object. Equivalent encodings such as `2/2` and `1`
are accepted after exact `Fraction` normalization.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
