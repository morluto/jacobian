# Audit two degree-sequence claims

For each sorted sequence in `/app/input.json`, determine whether it is the
degree sequence of a finite simple undirected graph. For the graphical case,
submit any simple edge list whose exact degrees match the sequence. Edge pairs
may be oriented either way, and vertex labels may be zero-based (`0..n-1`) or
one-based (`1..n`). For the
nongraphical case, submit every violating Erdős–Gallai index `k`, with the
exact left and right sides of the inequality.

Bind a concise explanation at `/app/evidence/answer.txt` and write
`/app/submission.json`. Do not claim completeness beyond these two frozen
sequences or claim `VERIFIED`; the checker provides exact finite computation.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** one of `MIXED_REALIZATION_AND_OBSTRUCTION`, `INCOMPLETE_AUDIT`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** a string value
- **Completeness:** one of `COMPLETE`, `PARTIAL`, `UNKNOWN`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
