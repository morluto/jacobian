# Exhaustively audit convex-position subsets of a frozen point set

The thirteen labelled integer points in `/app/input.json` are claimed to be in
general position. Verify that claim, count exactly how many subsets of each
size `3` through `13` are in convex position, determine the maximum such size,
and provide one maximum-size witness as point IDs in cyclic hull order.

A subset is in convex position when every selected point is a vertex of its
convex hull. The verifier independently checks every orientation determinant,
enumerates every subset, reconstructs the exact convex hulls, and validates the
witness. The order of the count rows is irrelevant, and any valid cyclic order
of any maximum witness is accepted.

This is a finite exact-geometry probe motivated by the Happy Ending
conjecture. It does not determine the general Erdős–Szekeres number. Claim
`CHECKED` only for the frozen thirteen-point scope.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier exhaustively checks the frozen point set; the finite optimum does not establish the general Happy Ending formula.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `HAPPY_ENDING_FINITE_CONVEX_POSITION`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED`, `CHECKED` (ceiling `CHECKED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `application/json`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
