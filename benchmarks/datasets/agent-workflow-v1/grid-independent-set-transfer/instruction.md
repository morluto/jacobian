# Replay independent-set transfer states on square grids

For each `n` from 2 through 5, count black/white colorings of an `n × n` grid with no horizontally or vertically adjacent black cells.

Use row masks. Submit, for every `n`, the sorted list of horizontally valid masks, the number of ordered vertically compatible mask pairs, the total number of partial colorings after each successive row, and the final count `x_n`. Also submit `x_2+x_3+x_4+x_5`. The four cases may appear in any order in the `cases` array; each case is matched by its `n` value.

Write `submission.json` and digest-bind `evidence/answer.txt`, which must be a JSON object that copies `result` and `limitations` exactly and includes `schema_version` set to `"1"` and `task_id` set to the `task_id` value from `/app/input.json`. The verifier reconstructs every state and transition independently. A final scalar without the intermediate transfer certificate is incomplete. Claim only `COMPUTED`; the scope is the four frozen finite grids.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `TRANSFER_COUNTS_REPLAYED`
- **Assurance:** scoreable values are `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
