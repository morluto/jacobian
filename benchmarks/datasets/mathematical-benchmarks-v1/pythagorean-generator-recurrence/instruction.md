Certify the NaturalProofs generator recurrence for almost-isosceles primitive
Pythagorean triangles.

Choose any integers `2 <= m <= 100` and `1 <= n < m` that are coprime, have
opposite parity, and satisfy `|m^2 - 2mn - n^2| = 1`. Starting from your seed,
apply `(m,n) -> (2m+n,m)` seven times. Submit all eight generators and their
triples `(2mn, m^2-n^2, m^2+n^2)`, together with the exact transformation
matrix, its determinant, and the multiplier by which it changes the quadratic
invariant.

The certificate must demonstrate every recurrence step, primitive-generator
condition, Pythagorean identity, and unit leg gap. Write `/app/submission.json`
and digest-bound `/app/evidence/answer.txt`.

The evidence file must contain exactly one `RESULT_JSON:` line whose JSON
equals the submitted `result` object. Do not claim `VERIFIED`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `VALID_RECURRENCE_CERTIFICATE`
- **Assurance:** scoreable values are `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
