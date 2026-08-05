# Smallest finite-magma countermodel

The offline input states a universally quantified premise and target identity
for one binary operation. Decide whether the premise implies the target over
nonempty finite magmas. If not, return a smallest countermodel, a valuation
that refutes the target, and the smaller carrier orders exhaustively checked.

Write `submission.json` to the exact agent-visible `submission_schema.json`.
Put a short calculation or independently replayable witness in
`evidence/answer.txt`. It must describe the operation, the refuting assignment,
and the one-element minimality check. Include that file's SHA-256 digest in the
submission.

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
