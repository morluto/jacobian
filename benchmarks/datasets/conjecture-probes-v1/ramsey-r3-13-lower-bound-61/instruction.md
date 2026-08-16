# Construct a graph certifying R(3,13) at least 61

Submit the complete edge set of a simple undirected graph on vertices 0 through 59. The graph must contain no triangle and no independent set of size 13.

Each edge is a two-element integer array. Endpoint order and edge-list order are irrelevant; self-loops and duplicate undirected edges are invalid. The verifier checks triangle-freeness directly and independently runs an exact bitset search for a 13-vertex independent set.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
