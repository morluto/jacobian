# Audit a Chebotarev density solution

Audit the frozen solution for `f(x)=x^4-4x+1`. Produce the exact mod-2 factorization certificate, the actual integer discriminant, and a complete `S4` cycle-type table with class size, fixed-point count, and whether the class contributes to the root-mod-p density. Compute the corrected fixed-point proportion and encoded answer.

Write `/app/submission.json` and `/app/evidence/chebotarev-audit.json` according to the schema. The density calculation is explicitly conditional on the frozen premise `Gal(f)=S4`; do not claim to prove that classification or Chebotarev. Claim only `COMPUTED`.

The digest-bound evidence file `evidence/chebotarev-audit.json` must be a JSON object with exactly four keys: `schema_version` (the string `"1"`), `task_id` (the task identifier), `result` (the same result object placed in `submission.json`), and `limitations` (the same limitations list placed in `submission.json`). The evidence file must not exceed 16 MiB.
