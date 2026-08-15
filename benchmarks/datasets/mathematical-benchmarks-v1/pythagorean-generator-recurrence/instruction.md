Certify the NaturalProofs generator recurrence for almost-isosceles primitive
Pythagorean triangles.

Choose any integers `2 <= m <= 100` and `1 <= n < m` that are coprime, have
opposite parity, and satisfy `|m^2 - 2mn - n^2| = 1`. Starting from your seed,
apply `(m,n) -> (2m+n,m)` seven times. Submit all eight generators and their
triples `(2mn, m^2-n^2, m^2+n^2)`, together with the exact transformation
matrix, its determinant, and the multiplier by which it changes the quadratic
invariant.

The certificate must demonstrate every recurrence step, primitive-generator
condition, Pythagorean identity, and unit leg gap. Write `/app/submission.json`
using the supplied schema.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
