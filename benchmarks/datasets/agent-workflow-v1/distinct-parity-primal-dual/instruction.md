# Certify a parity-constrained distinct-sum optimum

Choose distinct positive even integers and distinct positive odd integers whose combined sum is exactly 2025. If their counts are `m` and `n`, maximize `5m+7n`.

Submit both sides of an optimality certificate: any valid optimal construction, and the complete frontier for every possible odd count `n=1,3,...,45`. Each frontier row must give the largest feasible even count under the unavoidable minimum sum `m(m+1)+n^2`, that minimum sum, and its objective. Frontier rows may appear in any order; the verifier matches them by `odd_count`. Write `/app/submission.json` and `/app/evidence/distinct-parity-certificate.json` according to the schema. Claim only `COMPUTED`; the verifier independently checks the construction and reconstructs the entire upper-bound frontier.

The digest-bound evidence file must be a JSON object with exactly four keys:
`schema_version` (the string `"1"`), `task_id` (the task identifier),
`result` (the same result object placed in `submission.json`), and
`limitations` (the same limitations list placed in `submission.json`).
The evidence file must be a regular file of at most 16 MiB; larger or
non-regular files are rejected before hashing.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `OPTIMUM_CERTIFIED`
- **Assurance:** scoreable values are `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/distinct-parity-certificate.json`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `application/json`.
- **Required artifact filenames:** `evidence/distinct-parity-certificate.json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
