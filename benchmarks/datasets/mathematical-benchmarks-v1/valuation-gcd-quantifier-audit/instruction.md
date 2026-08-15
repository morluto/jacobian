# Repair a valuation quantifier

The frozen proof translates `gcd(a,b,c,d)=1` into: “the minimum of the four
valuations is zero for at least one prime dividing `n`.” Audit this translation.

Write `/app/submission.json` following the supplied schema and one bound JSON
task-specific witness file at `evidence/valuation-audit.json`.

The task-specific witness file must be a JSON object with exactly these keys:
`schema_version` (the string `"1"`), `task_id` (matching the submission's
`task_id`), `result` (the same result object as the submission), and

Submit:

1. a valuation countermodel on at least three distinct primes where the weak
   existential condition holds but the represented four integers have gcd
   greater than one; and
2. a repaired valuation system on at least three primes where every prime row
   has minimum zero, maximum `k`, and row sum `3k`, so the represented integers
   have gcd `1`, lcm `n`, and product `n^3`.

Prime rows must be strictly increasing and exponents are ordered as `(a,b,c,d)`.
Use at least two different zero-coordinate positions in the repaired system.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates the declared task-specific witness.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/valuation-audit.json`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `application/json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
