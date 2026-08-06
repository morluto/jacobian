# Finite partition and exact coverage

Partition the exact universe in `input.json` into named cases according to the
given residue relation. Return every member exactly once. State `TRUE` only
when the cases are pairwise disjoint and cover the complete supplied universe.
Write `submission.json` to the exact schema in the agent-visible
`submission_schema.json`. Put the coverage calculation in `evidence/answer.txt`
and include that file's SHA-256 digest in the evidence list.
Claim `VERIFIED` only by writing
`evidence/verification-record.json` and binding it through the
submission descriptor's `verification_record_uri`; otherwise claim `COMPUTED`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `TRUE`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` (ceiling `VERIFIED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`, `evidence/verification-record.json`.
- **Verification record:** write `evidence/verification-record.json` and bind it through `verification_record_uri` when claiming `VERIFIED`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
