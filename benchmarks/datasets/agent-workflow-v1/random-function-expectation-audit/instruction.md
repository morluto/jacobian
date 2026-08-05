# Audit an expectation claim

Audit the claim in the offline input using exact arithmetic. Account explicitly
for the dependence in `f(f(x))` when `f(x)=x`; return the relevant exact point
probabilities, the ordered squared-difference sum, and the exact expectation.

Write `submission.json` to the exact agent-visible `submission_schema.json`.
Put a concise exact derivation in `evidence/answer.txt`, include a
`RESULT_JSON:` line containing the submitted result as JSON, and bind the file
with its SHA-256 digest.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** one of `TRUE`, `FALSE`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
