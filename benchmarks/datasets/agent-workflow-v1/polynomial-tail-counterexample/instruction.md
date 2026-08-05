# Polynomial-tail monotonicity counterexample

Decide the polynomial claim in the offline input. If it is false, return
rational coefficient lists for `P` and `Q`, their distinct real roots, and an
exact rational pair `x1 < x2` at or beyond every root for which
`(P-Q)(x1) >= (P-Q)(x2)`.

Represent every rational, including every listed root, as a canonical rational
string such as `"5/3"` or `"-1"`. This benchmark's witness format intentionally
restricts roots to rational values; algebraic irrational roots are outside the
advertised artifact contract. Write `submission.json` to the exact
agent-visible `submission_schema.json`. Put the exact evaluations in
`evidence/answer.txt`, include a `RESULT_JSON:` line containing the submitted
result as JSON, and bind its SHA-256 digest.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `FALSE`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
