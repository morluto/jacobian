# Construct a nondifferentiable maximum

Construct a continuous piecewise-linear function on `[-1,1]` whose maximum is
attained at `0` but which is not differentiable there. Use the two-branch family
declared in the input and choose any rational peak and slopes satisfying the
requirements.

Return exact rational parameters as integer `numerator`/positive integer
`denominator` objects in lowest terms and the branch values at the join. The
verifier independently checks continuity at zero, monotonicity toward and away
from the peak, and the unequal one-sided derivatives. Write `submission.json`
to the exact `submission_schema.json` contract, Write the task-specific witness to
`evidence/answer.txt`, and bind that file with its SHA-256 digest.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates the declared task-specific witness.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `text/plain`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
