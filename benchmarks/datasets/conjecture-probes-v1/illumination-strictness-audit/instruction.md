# Cube illumination strictness audit

The frozen evaluator treats a direction as illuminating a cube vertex when it is merely non-outward on every active facet. True illumination requires strict inward motion on every active facet.

Submit:

1. exactly four nonzero directions in `{-1,0,1}^3`, each with exactly one zero coordinate, whose weak nonpositive test covers all eight vertices but whose strict test does not;
2. the complete sorted list of weakly accepted `(vertex_index,direction_index)` pairs that fail strict illumination; and
3. exactly eight nonzero directions in `{-1,1}^3` that strictly illuminate all eight vertices, together with the unique illuminating direction index for each vertex.

For a vertex `v` and direction `d`, the weak test is `v_i*d_i <= 0` for all coordinates; strict illumination is `v_i*d_i < 0` for all coordinates. The verifier reconstructs every pair. It also certifies minimality for the cube: one strict sign direction can illuminate at most one vertex.

This Assurance result is exact for one cube only and does not establish the Illumination Conjecture in general. Claim at most `CHECKED`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

Exact weak-versus-strict illumination audit for the three-dimensional cube only.

The evidence file must be a JSON object with exactly four fields: `schema_version` must be `"1"`, `task_id` must equal the submission task ID, `result` must be an exact JSON copy of `submission.json`'s result object, and `limitations` must be an exact JSON copy of its limitations list.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `WEAK_ILLUMINATION_IS_UNSOUND_AND_REPAIRED`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED`, `CHECKED` (ceiling `CHECKED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.json`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `application/json`.
- **Required artifact filenames:** `evidence/answer.json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
