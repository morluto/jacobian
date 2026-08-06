# Audit edge-pair ordering in a graph sum

Read `/app/input.json`, including its complete definition of the edge-pair polynomials, the summation domain, and the source derivation being audited. Determine the coefficient of `x` in their sum over all labeled simple graphs, respecting the declared pair semantics. Diagnose the ordered-versus-unordered counting mismatch from the supplied source derivation; do not introduce a different polynomial family.

Write `/app/submission.json` using `/app/submission_schema.json` and `/app/evidence/answer.txt`. Give the symbolic incident-pair factor, the number of free graph-edge bits, the corrected formula, and exact coefficients for every requested `n`. The symbolic formula applies for `n>=3`, where at least one distinct incident edge pair exists; for `n<3` the coefficient is zero. Set `scope` exactly to `ordered edge pairs over all labeled simple graphs` and set `limitations` exactly to `Finite exhaustive replay covers n=3,4,5,6 only; it does not machine-prove the symbolic formula for every n.`

The evidence file must contain exactly one `RESULT_JSON:` line whose compact JSON is exactly the submitted `result` object. Outside that line, explain the ordered-pair count, the factor-of-two diagnosis, the free-edge contribution, and why the finite probes do not establish the all-`n` formula by themselves. Bind this file by its SHA-256 digest. Do not claim `VERIFIED`.

The verifier independently enumerates every labeled graph for `n=3,4,5,6` and its ordered edge pairs, and separately checks the symbolic factors. It does not prove the formula for arbitrary `n`; assurance must remain `COMPUTED`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `SOURCE_FACTOR_OF_TWO_REPAIRED`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
