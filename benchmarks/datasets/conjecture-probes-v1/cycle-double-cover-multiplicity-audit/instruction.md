# Cycle-double-cover multiplicity audit

The frozen validator accepts any collection of simple cycles whose union covers every edge. This is not the cycle-double-cover contract, which requires every edge to occur exactly twice **counting cycle multiplicity**.

For the supplied Petersen graph, submit a complete certificate containing:

1. a flawed-pass collection of at least four distinct simple cycles that covers every edge but is not a double cover;
2. its complete 15-entry edge-multiplicity vector and the exact sorted indices whose multiplicity is not two; and
3. a repaired collection of distinct simple cycles whose complete multiplicity vector is exactly two on every edge.

Each cycle is a vertex list without a repeated closing vertex. The verifier canonicalizes rotations and reversals, checks simplicity and every consecutive graph edge, rejects duplicate cycles, and recomputes both multiplicity profiles. Alternative valid collections are accepted.

This is one finite Assurance audit. It does not establish the Cycle Double Cover Conjecture for any graph family. Claim at most `CHECKED`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

Exact multiplicity audit and repair for one Petersen-graph cycle system only.

The evidence file must be a JSON object with exactly four fields: `schema_version` must be `"1"`, `task_id` must equal the submission task ID, `result` must be an exact JSON copy of `submission.json`'s result object, and `limitations` must be an exact JSON copy of its limitations list.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `UNION_COVERAGE_IS_INSUFFICIENT_AND_REPAIRED`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED`, `CHECKED` (ceiling `CHECKED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.json`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `application/json`.
- **Required artifact filenames:** `evidence/answer.json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
