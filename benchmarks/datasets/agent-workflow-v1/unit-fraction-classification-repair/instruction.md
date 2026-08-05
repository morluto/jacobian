# Repair a flawed unit-fraction classification

The frozen solution derives `(x-n)(y-n)=n^2` correctly, but then makes invalid divisibility claims and reports 2022 qualifying integers through 2025. The flawed derivation is provided in the frozen input. Diagnose each defect and recompute the exact classification from the equation and constraints.

Submit the complete membership vector for `n=1..2025` as a 508-character little-endian bit-packed hexadecimal string (bit `n-1`; unused final bits zero), the corrected count, at least ten distinct valid `(n,d)` witnesses, three required nonmember counterexamples, and all diagnosed defects. The evidence artifact must be a JSON object with exactly `schema_version`, `task_id`, `result`, and `limitations`, using `schema_version: "1"`, the submitted task identifier, and exact copies of the submitted `result` and `limitations`; keep it at or below 16 MiB. Claim only `COMPUTED`. The verifier independently enumerates every divisor condition and reconstructs `(x,y)` for each witness.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `PUBLISHED_CLASSIFICATION_AND_COUNT_FALSE`
- **Assurance:** scoreable values are `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/unit-fraction-repair.json`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `application/json`.
- **Required artifact filenames:** `evidence/unit-fraction-repair.json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
