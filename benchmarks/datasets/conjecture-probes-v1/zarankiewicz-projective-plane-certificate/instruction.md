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

This finite certificate establishes only
`z(13,13;2,2)=52`; it does not settle the general Zarankiewicz problem.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
