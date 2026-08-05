# Repair a natural-subtraction proof

Audit the failed rewrite in the frozen natural-number proof branch, then give
an exact algebraic repair certificate.

First report whether the failed pattern occurs as a subtree of the target AST.
Then use the declared equation basis to derive the goal: submit one rational
multiplier per basis equation and the resulting coefficient vector in the
declared variable order. The subtraction-recovery equation is justified only
by the recorded `b<=a` side condition.
Use scope exactly: `the frozen natural-subtraction branch and declared equation basis`.

The verifier independently traverses the expression tree and recomputes the
linear combination over exact rationals. It does not run Lean or accept a
`VERIFIED` claim. Write `submission.json` to `submission_schema.json`, put a
concise diagnosis and derivation in `evidence/answer.txt`, and bind its SHA-256
digest.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** one of `REPAIR_CERTIFIED`, `NO_REPAIR`, `UNSUPPORTED`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** one of `COMPLETE`, `PARTIAL`, `UNKNOWN`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
