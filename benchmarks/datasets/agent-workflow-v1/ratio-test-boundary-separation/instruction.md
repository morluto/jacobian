# Separate the two outcomes at the ratio-test boundary

Use the harmonic series `1/n` as the divergent witness and the telescoping
series `1/(n*(n+1))` as the convergent witness. Both are positive rational
sequences indexed by `n >= 1` whose consecutive-term ratios tend to one, but
the harmonic series diverges while the telescoping series converges.

Supply the exact ratio and ratio-error identities for each witness, nine
dyadic lower-bound blocks for the divergent witness, and at least four freely
chosen partial-sum checkpoints for the convergent witness.

The verifier independently evaluates every rational checkpoint and replays the
submitted symbolic identities. A conclusion label or numerical sampling alone
is insufficient. The evidence must contain exactly one `RESULT_JSON:` line
equal to the submitted `result` and explain why a ratio limit equal to one is
inconclusive — both outcomes must be discussed. Do not claim proof-assistant
verification.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `RATIO_BOUNDARY_INCONCLUSIVE`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
