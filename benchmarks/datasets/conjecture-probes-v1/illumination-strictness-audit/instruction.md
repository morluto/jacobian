# Cube illumination strictness audit

The frozen evaluator treats a direction as illuminating a cube vertex when it is merely non-outward on every active facet. True illumination requires strict inward motion on every active facet.

Submit:

1. exactly four nonzero directions in `{-1,0,1}^3`, each with exactly one zero coordinate, whose weak nonpositive test covers all eight vertices but whose strict test does not;
2. the complete sorted list of weakly accepted `(vertex_index,direction_index)` pairs that fail strict illumination; and
3. exactly eight nonzero directions in `{-1,1}^3` that strictly illuminate all eight vertices, together with the unique illuminating direction index for each vertex.

For a vertex `v` and direction `d`, the weak test is `v_i*d_i <= 0` for all coordinates; strict illumination is `v_i*d_i < 0` for all coordinates. The verifier reconstructs every pair. It also certifies minimality for the cube: one strict sign direction can illuminate at most one vertex.

This Assurance result is exact for one cube only and does not establish the Illumination Conjecture in general.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
