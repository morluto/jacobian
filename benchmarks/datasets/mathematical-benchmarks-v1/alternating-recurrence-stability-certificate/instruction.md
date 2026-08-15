# Certify recurrence stability
For `a_(n+1)=2^n-7a_n`, derive the exact closed form in terms of `delta=a_0-c`, where `c` is the particular-solution coefficient.

Choose at least four distinct checkpoint indices in the declared bounds and give exact values and forward differences for the surviving initial value. Report the requested reciprocal.


<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates the declared task-specific witness.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/stability-certificate.json`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `application/json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
