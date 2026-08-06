# Construct an optimal distinct-sum pairing

For the frozen ground set, construct as many disjoint unordered pairs as
possible so that all pair sums are distinct and at most `n`.

Submit the pairs in canonical increasing order, their sums, and the claimed
optimum. The verifier independently checks the witness and exhaustively solves
the finite optimization problem; it accepts any optimal pairing, not one
expected arrangement. Write `submission.json` to `submission_schema.json`,
explain the five-pair construction, its distinct sums, and the exhaustive
exclusion of a six-pair solution in `evidence/answer.txt`, and bind its SHA-256
digest.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** one of `OPTIMAL_PAIRING`, `NO_PAIRING`, `UNSUPPORTED`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** a string value
- **Completeness:** one of `COMPLETE`, `PARTIAL`, `UNKNOWN`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
