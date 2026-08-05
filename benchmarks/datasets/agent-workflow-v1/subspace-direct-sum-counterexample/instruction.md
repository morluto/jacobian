# Four-subspace direct-sum counterexample

Decide the claim in the offline input. If false, give one nonzero integer
generator for each of four one-dimensional subspaces of `Q^3` and nonzero
integer coefficients witnessing a linear dependence among the four
generators.

The verifier will check every ordered choice of three distinct indices and
will independently replay the dependence. Write `submission.json` to the exact
agent-visible `submission_schema.json`, include a `RESULT_JSON:` line containing
the submitted result as JSON in `evidence/answer.txt`, and bind that file by
SHA-256.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `FALSE`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
