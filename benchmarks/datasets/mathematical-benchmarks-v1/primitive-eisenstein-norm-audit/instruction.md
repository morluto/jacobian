# Audit the primitive norm criterion

Read `/app/input.json`. The frozen proof claims that a primitive value of
`Q(x,y)=x^2+xy+y^2` is never divisible by 3, while allowing even powers of primes congruent to 2 modulo 3.

Submit two local certificates:

1. choose coprime nonzero integers `x,y` in the declared bounds whose norm is divisible by 3, and report its exact 3-adic valuation;
2. choose one allowed inert prime, report all residue pairs modulo that prime for which `Q` vanishes, and use the result to classify whether its square can have a primitive representation.

Then state the repaired prime-factor criterion. Write `/app/submission.json` following the schema and bind a matching copy of the result at `evidence/local-audit.json`. The evidence file must be a JSON object with exactly the fields `schema_version` (the string `"1"`), `task_id` (matching the submission), `result` (matching the submission's `result`), and `limitations` (matching the submission's `limitations`). Do not solve or certify the source's cubic-form counting problem. Assurance is `COMPUTED` only.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `SOURCE_PRIMITIVE_NORM_CRITERION_FALSE`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/local-audit.json`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `application/json`.
- **Required artifact filenames:** `evidence/local-audit.json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
