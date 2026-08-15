# Eliminate a complex power-sum system

Solve the frozen complex-number problem completely and provide an exact
certificate that:

1. establishes that the submitted branches are exhaustive;
2. proves every denominator used in the two hypotheses and target is nonzero;
3. returns both and only the possible target values as quadratic surds; and
4. certifies that each submitted branch is achievable.

The optional `recurrence` and `hypothesis_factorization` fields describe one
accepted exact certificate format; they are not required.

The two algebraic branches may appear in either order. Rational numbers must be
reduced with positive denominators. Write the result to `/app/submission.json`
in the published schema.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
