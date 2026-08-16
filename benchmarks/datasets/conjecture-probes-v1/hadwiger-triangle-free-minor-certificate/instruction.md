# Construct a nontrivial finite Hadwiger certificate

Submit a simple connected triangle-free graph on vertices `0..10` with exactly
20 edges and minimum degree at least 3. Supply a proper four-coloring and four
pairwise-disjoint nonempty branch sets witnessing a `K4` minor: every branch
set must induce a connected subgraph and every pair must have a crossing edge.

The verifier independently rejects every possible three-coloring by exact
backtracking, checks the submitted four-coloring, and validates the complete
minor model. Complete graphs, triangles, isolated padding, and label-only
chromatic claims are rejected.

equal the task id, `result` must exactly copy the submitted `result` object
(including JSON types), and `limitations` must equal the declared limitations.
This checks one finite graph only and does not prove Hadwiger's conjecture.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
