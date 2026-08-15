# Audit a Chebotarev density solution

Audit the frozen solution for `f(x)=x^4-4x+1`. Produce the exact mod-2 factorization certificate, the actual integer discriminant, and a complete `S4` cycle-type table with class size, fixed-point count, and whether the class contributes to the root-mod-p density. Compute the corrected fixed-point proportion and encoded answer.

Write `/app/submission.json` and `/app/evidence/chebotarev-audit.json` according to the schema. The density calculation is explicitly conditional on the frozen premise `Gal(f)=S4`;

The digest-bound witness file `evidence/chebotarev-audit.json` must be a JSON object with exactly three keys: `schema_version` (the string `"1"`), `task_id` (the task identifier), and `result` (the same result object placed in `submission.json`).

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates the declared task-specific witness.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/chebotarev-audit.json`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `application/json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
