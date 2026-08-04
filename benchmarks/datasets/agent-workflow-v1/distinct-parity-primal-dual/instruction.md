# Certify a parity-constrained distinct-sum optimum

Choose distinct positive even integers and distinct positive odd integers whose combined sum is exactly 2025. If their counts are `m` and `n`, maximize `5m+7n`.

Submit both sides of an optimality certificate: any valid optimal construction, and the complete frontier for every possible odd count `n=1,3,...,45`. Each frontier row must give the largest feasible even count under the unavoidable minimum sum `m(m+1)+n^2`, that minimum sum, and its objective. Frontier rows may appear in any order; the verifier matches them by `odd_count`. Write `/app/submission.json` and `/app/evidence/distinct-parity-certificate.json` according to the schema. Claim only `COMPUTED`; the verifier independently checks the construction and reconstructs the entire upper-bound frontier.

The digest-bound evidence file must be a JSON object with exactly four keys:
`schema_version` (the string `"1"`), `task_id` (the task identifier),
`result` (the same result object placed in `submission.json`), and
`limitations` (the same limitations list placed in `submission.json`).
The evidence file must be a regular file of at most 16 MiB; larger or
non-regular files are rejected before hashing.
