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
`COMPUTED` only for the frozen thirteen-point scope.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates the declared task-specific witness.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `application/json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
