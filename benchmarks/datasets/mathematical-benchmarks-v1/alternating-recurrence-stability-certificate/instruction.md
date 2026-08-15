# Certify recurrence stability
For `a_(n+1)=2^n-7a_n`, derive the exact closed form in terms of `delta=a_0-c`, where `c` is the particular-solution coefficient. Explain with a structured parity certificate why every positive and every negative nonzero `delta` eventually violates `a_(n+1)>a_n`.

Choose at least four distinct checkpoint indices in the declared bounds and give exact values and forward differences for the surviving initial value. Report the requested reciprocal.

Write `/app/submission.json` and bind an evidence object at `evidence/stability-certificate.json`. The evidence object must be JSON with exactly these fields: `schema_version` set to `"1"`, `task_id` set to the `task_id` value from `/app/input.json`, `result` containing an exact copy of the submitted `result`, and `limitations` containing an exact copy of the submitted `limitations`. The evidence file must be no larger than 16 MiB. Finite simulation alone is insufficient. Assurance is `COMPUTED`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates a witness only where the task needs one.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/stability-certificate.json`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `application/json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
