# Construct a nondifferentiable maximum

Construct a continuous piecewise-linear function on `[-1,1]` whose maximum is
attained at `0` but which is not differentiable there. Use the two-branch family
declared in the input and choose any rational peak and slopes satisfying the
requirements.

Return exact rational parameters and the branch values at the join. The
verifier independently checks continuity at zero, monotonicity toward and away
from the peak, and the unequal one-sided derivatives. Write `submission.json`
to the exact `submission_schema.json` contract, put a concise derivation in
`evidence/answer.txt`, and bind that file with its SHA-256 digest.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** one of `VALID_NONDIFFERENTIABLE_MAXIMUM`, `NO_CONSTRUCTION`, `UNSUPPORTED`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** a string value
- **Completeness:** one of `COMPLETE`, `PARTIAL`, `UNKNOWN`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
