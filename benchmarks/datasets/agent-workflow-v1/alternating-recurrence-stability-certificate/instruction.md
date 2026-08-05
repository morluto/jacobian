# Certify recurrence stability
For `a_(n+1)=2^n-7a_n`, derive the exact closed form in terms of `delta=a_0-c`, where `c` is the particular-solution coefficient. Explain with a structured parity certificate why every positive and every negative nonzero `delta` eventually violates `a_(n+1)>a_n`.

Choose at least four distinct checkpoint indices in the declared bounds and give exact values and forward differences for the surviving initial value. Report the requested reciprocal.

Write `/app/submission.json` and bind an evidence object at `evidence/stability-certificate.json`. The evidence object must be JSON with exactly these fields: `schema_version` set to `"1"`, `task_id` set to the `task_id` value from `/app/input.json`, `result` containing an exact copy of the submitted `result`, and `limitations` containing an exact copy of the submitted `limitations`. The evidence file must be no larger than 16 MiB. Finite simulation alone is insufficient. Assurance is `COMPUTED`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `UNIQUE_STABLE_INITIAL_VALUE`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/stability-certificate.json`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `application/json`.
- **Required artifact filenames:** `evidence/stability-certificate.json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
