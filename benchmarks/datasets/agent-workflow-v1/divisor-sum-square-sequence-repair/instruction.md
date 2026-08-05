# Construct infinitely many square divisor sums

The frozen solution claims no sequence exists, relying on probabilistic
language that does not apply to a deterministic existence problem. Repair it.

Submit a deterministic piecewise formula for positive integers `a_n` with
`a_1=1`, a default power-of-two branch for every index that is not an odd prime
(including `n=2`), and a separate odd-prime branch. Certify that for each fixed
positive `k`, every `a_n` with `n>=max(2,k)` is divisible by `2^k`, and
therefore only finitely many can equal `k mod 2^k`. Also submit at least four
freely chosen distinct odd-prime probes where `b_p = sum_{d|p} d*a_d` is an
exact square.

Use `/app/submission.json` and bind an identical certificate at
`evidence/sequence-construction.json`. The certificate must be a JSON object
with exactly the fields `schema_version` (the string `"1"`), `task_id`
(the task identifier), `result` (an object equal to
the submission's `result`), and `limitations` (an array equal to the
submission's `limitations`). The `evidence[0].sha256` field must be the
SHA-256 digest of the certificate's exact on-disk bytes, prefixed with
`sha256:`. Use completeness `COMPLETE` and maximum assurance `COMPUTED`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** one of `INFINITE_SQUARE_SUBSEQUENCE_CONSTRUCTED`, `INSUFFICIENT_EVIDENCE`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/sequence-construction.json`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `application/json`.
- **Required artifact filenames:** `evidence/sequence-construction.json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
