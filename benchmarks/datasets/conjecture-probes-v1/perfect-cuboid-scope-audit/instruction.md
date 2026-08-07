# Audit finite evidence around the perfect-cuboid conjecture

For every frozen integer-edge cuboid in `/app/input.json`, compute the three
face-diagonal radicands and the space-diagonal radicand. Report an integer
square root when the radicand is a perfect square and `null` otherwise, then
classify the cuboid as exactly one of:

- `PERFECT_CUBOID`: all three face diagonals and the space diagonal are integers;
- `EULER_BRICK_ONLY`: all three face diagonals are integers but the space diagonal is not;
- `SPACE_AND_TWO_FACES`: the space diagonal and exactly two face diagonals are integers;
- `OTHER`: every remaining case.

Submit all twelve cases exactly once, but their order is irrelevant. Report the
counts of all four classes and whether the frozen family contains a perfect
cuboid. The verifier recomputes every square predicate from the frozen edges.
The three `face_radicands` entries and their aligned `face_roots` entries may
be reported in any common order; the verifier compares the three aligned
radicand/root pairs as an unordered set.

`evidence/answer.txt` must be a JSON object with exactly `schema_version`,
`task_id`, `result`, and `limitations`. Use schema version `1`, the task ID and
limitations from the submission contract, and the same `result` object as in
`submission.json`. The file must be no larger than 2 MiB.

This is a finite semantic-scope audit. An Euler brick is not a perfect cuboid,
and finding no perfect cuboid in these twelve cases is not evidence of global
nonexistence. Claim `CHECKED` only for the frozen case set.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks all finite square predicates independently; no bounded outcome resolves the open perfect-cuboid problem.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `PERFECT_CUBOID_FINITE_SCOPE_AUDIT`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED`, `CHECKED` (ceiling `CHECKED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `application/json`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
