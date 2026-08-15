# Audit the primitive norm criterion

Read `/app/input.json`. The frozen proof claims that a primitive value of
`Q(x,y)=x^2+xy+y^2` is never divisible by 3, while allowing even powers of primes congruent to 2 modulo 3.

Submit two local certificates:

1. choose coprime nonzero integers `x,y` in the declared bounds whose norm is divisible by 3, and report its exact 3-adic valuation;
2. choose one allowed inert prime, report all residue pairs modulo that prime for which `Q` vanishes, and use the result to classify whether its square can have a primitive representation.

Then state the repaired prime-factor criterion through the maximum allowed
exponent of 3 and the modulus/residue class of forbidden inert primes. Write
`/app/submission.json` following the schema and bind a matching copy of the
result at `evidence/local-audit.json`. The task-specific witness file must be a
JSON object with exactly the fields `schema_version` (the string `"1"`),
`task_id` (the task identifier), and `result` (matching the submission's
`result`). Do not solve or certify the source's cubic-form counting problem.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates the declared task-specific witness.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/local-audit.json`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `application/json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
