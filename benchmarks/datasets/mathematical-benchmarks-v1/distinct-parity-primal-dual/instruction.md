# Certify a parity-constrained distinct-sum optimum

Choose distinct positive even integers and distinct positive odd integers whose combined sum is exactly 2025. If their counts are `m` and `n`, maximize `5m+7n`.

Submit both sides of an optimality certificate: any valid optimal construction, and the complete frontier for every possible odd count `n=1,3,...,45`. Each frontier row must give the largest feasible even count under the unavoidable minimum sum `m(m+1)+n^2`, that minimum sum, and its objective. Frontier rows may appear in any order; the verifier matches them by `odd_count`. Write `/app/submission.json` and `/app/evidence/distinct-parity-certificate.json` according to the schema.

The digest-bound witness file must be a JSON object with exactly three keys:
`schema_version` (the string `"1"`), `task_id` (the task identifier), and
`result` (the same result object placed in `submission.json`).

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates the declared task-specific witness.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/distinct-parity-certificate.json`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `application/json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
