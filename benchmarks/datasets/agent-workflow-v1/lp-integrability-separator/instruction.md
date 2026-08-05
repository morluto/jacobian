Construct a nonnegative measurable function on `(0,+infinity)` that belongs to
`L^2` and to no other `L^p` for positive finite `p`.

Use the frozen two-tail family

`f(x) = x^(-1/2) (log(1/x))^(-beta)` for `0 < x < e^(-1)`,

`f(x) = 0` for `e^(-1) <= x <= e`, and

`f(x) = x^(-1/2) (log x)^(-beta)` for `x > e`,

but choose your own canonical rational `beta > 1/2`. Submit exact transformed
`p=2` integrals and a regime certificate explaining why the origin obstructs
every `p>2` and infinity obstructs every `0<p<2`. Do not use numeric sampling.

Every rational field (`beta`, `p2_log_exponent`, `p2_integral_each`) accepts any
mathematically equivalent rational written in canonical or non-canonical form
(for example `1`, `1/1`, `2/1`, `-2`, `-2/1` are all accepted); the verifier
compares exact values, not lexical strings.

Write `/app/submission.json` and `/app/evidence/answer.txt` using the provided
schema. The evidence file must be at most 64 KiB and must contain exactly one
`RESULT_JSON:` line whose JSON equals the submitted `result` object. Each
rational field is at most 80 characters. Claim no assurance above `COMPUTED`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `VALID_LP_SEPARATOR`
- **Assurance:** scoreable values are `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE_FOR_DECLARED_FAMILY`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
