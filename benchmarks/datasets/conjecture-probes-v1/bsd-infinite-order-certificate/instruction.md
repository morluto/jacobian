# Certify an infinite-order point without overclaiming BSD

Choose integers `A,B` and a nonzero integral point `P=(x,y)` within the bounds
in `/app/input.json` on the nonsingular curve `y^2=x^3+A*x+B`. Supply the exact
discriminant `-16(4A^3+27B^2)`, `y^2`, and the exact rational coordinates of
`2P` and `3P` as structured `{numerator, denominator}` rationals. Equivalent encodings such as `2/2` and `1` are accepted after exact `Fraction` normalization.

Your certificate must satisfy the Lutz–Nagell obstruction: `y != 0` and `y^2`
does not divide the absolute discriminant. The verifier independently checks
the curve equation, nonsingularity, divisibility obstruction, and both group
law computations. Lutz–Nagell itself is a declared trusted theorem.

This proves only that one authored elliptic curve has a rational point of
infinite order. It does not compute an L-function, determine the full rank, or
prove any case of the Birch–Swinnerton-Dyer conjecture.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
