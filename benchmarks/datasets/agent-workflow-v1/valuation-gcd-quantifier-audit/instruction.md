# Repair a valuation quantifier

The frozen proof translates `gcd(a,b,c,d)=1` into: “the minimum of the four
valuations is zero for at least one prime dividing `n`.” Audit this translation.

Write `/app/submission.json` following the supplied schema and one bound JSON
evidence file at `evidence/valuation-audit.json`.

The evidence file must be a JSON object with exactly these keys:
`schema_version` (the string `"1"`), `task_id` (matching the submission's
`task_id`), `result` (the same result object as the submission), and
`limitations` (the same limitations array as the submission).

Submit:

1. a valuation countermodel on at least three distinct primes where the weak
   existential condition holds but the represented four integers have gcd
   greater than one; and
2. a repaired valuation system on at least three primes where every prime row
   has minimum zero, maximum `k`, and row sum `3k`, so the represented integers
   have gcd `1`, lcm `n`, and product `n^3`.

Prime rows must be strictly increasing and exponents are ordered as `(a,b,c,d)`.
Use at least two different zero-coordinate positions in the repaired system.
The maximum permitted assurance is `COMPUTED`; do not claim `VERIFIED`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** one of `QUANTIFIER_DEFECT_REPAIRED`, `INSUFFICIENT_EVIDENCE`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** one of `COMPLETE`, `PARTIAL`, `UNKNOWN`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/valuation-audit.json`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `application/json`.
- **Required artifact filenames:** `evidence/valuation-audit.json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
