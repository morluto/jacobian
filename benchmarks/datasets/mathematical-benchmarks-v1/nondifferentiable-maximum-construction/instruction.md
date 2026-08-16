# Construct a nondifferentiable maximum

Construct a continuous piecewise-linear function on `[-1,1]` whose maximum is
attained at `0` but which is not differentiable there. Use the two-branch family
declared in the input and choose any rational peak and slopes satisfying the
requirements.

Return exact rational parameters as integer `numerator`/positive integer
`denominator` objects and the branch values at the join. Equivalent encodings such as `2/8` and `1/4` are accepted after exact `Fraction` normalization. The
verifier independently checks continuity at zero, monotonicity toward and away
from the peak, and the unequal one-sided derivatives. Write `submission.json`
to the exact `submission_schema.json` contract.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
