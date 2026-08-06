# Audit the summation domain

For fixed `n`, the source defines `a_n` by summing over every binary function
on the positive integers with finite support, then silently replaces that set
by subsets of `{1,...,n}`.

Choose `4 <= n <= 12` and provide at least six distinct singleton supports
strictly beyond `n`. Compute each summand and the resulting finite partial-sum
lower bound. Then repair the definition by restricting supports to
`{1,...,n}`: provide at least three exact rational checkpoints for

`c_n = product_{k=1}^n (2+1/k^2) / n!`

and a uniform ratio certificate showing `c_{n+1}/c_n <= 3/4` for every `n>=2`.
Bind an evidence object at `evidence/scope-audit.json`. The object must have exactly `schema_version`, `task_id`, `result`, and `limitations`; use `schema_version: "1"`, the task identifier from `/app/input.json`, and exact copies of the submitted `result` and `limitations`. Assurance is `COMPUTED` only.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** one of `ORIGINAL_SUM_DIVERGES_TRUNCATED_LIMIT_ZERO`, `INSUFFICIENT_EVIDENCE`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** one of `COMPLETE`, `PARTIAL`, `UNKNOWN`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/scope-audit.json`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `application/json`.
- **Required artifact filenames:** `evidence/scope-audit.json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
