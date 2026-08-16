# Repair a natural-subtraction proof

Audit the failed rewrite in the frozen natural-number proof branch, then give
an exact algebraic repair certificate.

First report whether the failed pattern occurs as a subtree of the target AST.
Then use the declared equation basis to derive the goal: submit one rational
multiplier per basis equation and the resulting coefficient vector in the
declared variable order. Represent each rational as an integer `numerator` and
positive integer `denominator`. Equivalent encodings such as `2/8` and `1/4` are accepted after exact `Fraction` normalization. The subtraction-recovery equation is justified only
by the recorded `b<=a` side condition.

The verifier independently traverses the expression tree and recomputes the
linear combination over exact rationals. Write `submission.json` to
`submission_schema.json`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
