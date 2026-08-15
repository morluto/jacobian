# Certify a limit missed by every straight line

Construct a function from the family

`f_p(x,y) = x^(2p) y / (x^(4p) + y^2)`, with `1 <= p <= 5`, away from the
origin and define its value at the origin to be zero.

Submit an exact certificate showing that the limit is zero along every
straight line through the origin, including both coordinate axes, while the
two-variable limit does not exist. Your all-lines certificate must give the
orders obtained after substituting `y=m*x` for an arbitrary nonzero slope and
explain why the resulting quotient tends to zero. Also submit three distinct,
freely chosen nonzero rational parameters `c` for paths `y=c*x^(2p)`, with the
exact nonzero limit on each path.

The verifier independently checks the exponent relations and every rational
path value. Write `c` and each exact limit as a signed integer, finite decimal,
or signed fraction such as `+1`, `0.5`, or `-2/3`; fraction denominators must be
nonzero. Numerical sampling or a conclusion label alone is insufficient.
The submitted line and nonlinear-path data must show that agreement on every
straight line does not establish the multivariable limit.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
