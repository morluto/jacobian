# Exact Zarankiewicz certificate from the projective plane over F3

Construct the point-line incidence graph of the projective plane over
`F3`. Submit the 13 normalized projective points, the 13 normalized projective
lines, all 52 incidence edges, and exact pair-count data proving that the graph
is 4-regular and contains no `K_{2,2}`. Normalization means that the first
nonzero coordinate of every triple is `1` modulo 3.

The verifier reconstructs projective equivalence classes, recomputes every
dot-product incidence, checks degrees and duplicate-free coverage, exhaustively
checks all pairs on both sides, and replays the extremal upper bound. For a
`K_{2,2}`-free bipartite graph with 13 vertices on each side and left degrees
`d_i`, the 78 unordered right-vertex pairs can be used at most once, so
`sum_i binom(d_i,2) <= binom(13,2)`. Convexity then rules out 53 edges, while
the submitted 52-edge construction attains the bound.

Evidence is matching JSON with exactly `schema_version`, `task_id`, `result`,
and `limitations`. This finite certificate establishes only
`z(13,13;2,2)=52`; it does not settle the general Zarankiewicz problem.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

Exact PG(2,3) incidence and finite extremal replay only; no general Zarankiewicz conclusion.

The evidence file must be a JSON object with exactly four fields: `schema_version` must be `"1"`, `task_id` must equal the submission task ID, `result` must be an exact JSON copy of `submission.json`'s result object, and `limitations` must be an exact JSON copy of its limitations list.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `PG2_F3_ZARANKIEWICZ_EXTREMAL_CERTIFICATE`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED`, `CHECKED` (ceiling `CHECKED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.json`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `application/json`.
- **Required artifact filenames:** `evidence/answer.json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
