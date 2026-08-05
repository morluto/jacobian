# Eliminate a complex power-sum system

Solve the frozen complex-number problem completely and provide an exact
certificate that:

1. establishes that the submitted branches are exhaustive;
2. proves every denominator used in the two hypotheses and target is nonzero;
3. returns both and only the possible target values as quadratic surds; and
4. certifies that each submitted branch is achievable.

The optional `recurrence` and `hypothesis_factorization` fields describe one
accepted exact certificate format; they are not required, and any derivation
strategy may be used.

The two algebraic branches may appear in either order. Rational numbers must be
reduced with positive denominators. Write the result to
`/app/submission.json`, place a concise derivation in
`/app/evidence/answer.txt`, and bind the evidence by SHA-256. The derivation
must state the power-sum recurrence, both eliminated polynomials, why both
branches occur, and the denominator-norm checks.

Do not claim `VERIFIED`: the clean-room checker replays exact polynomial and
quadratic-field arithmetic, but no external proof assistant checks the general
complex-algebra argument.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** one of `COMPLETE_TARGET_SET_DERIVED`, `INSUFFICIENT_EVIDENCE`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** a string value
- **Completeness:** one of `COMPLETE`, `PARTIAL`, `UNKNOWN`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
