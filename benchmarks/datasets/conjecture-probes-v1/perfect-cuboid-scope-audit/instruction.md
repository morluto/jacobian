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

This is a finite semantic-scope audit. An Euler brick is not a perfect cuboid,
and finding no perfect cuboid in these twelve cases is not evidence of global
nonexistence. Claim `COMPUTED` only for the frozen case set.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
