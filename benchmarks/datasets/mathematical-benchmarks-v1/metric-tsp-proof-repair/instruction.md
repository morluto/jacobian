# Repair a Metric TSP proof

Audit and repair the flawed claim in the offline input. Identify the invalid
inference and state the strongest generic guarantee as the finite claim
`DOUBLE_TREE_TWO_APPROXIMATION`. For the supplied metric, give a minimum spanning tree,
an Euler circuit of its doubled edges, the Hamiltonian cycle obtained by
retaining first visits, and an optimal Hamiltonian cycle. Report all exact
weights so the concrete trace demonstrates why exactness is unsupported while
the repaired guarantee holds.

Write `submission.json` to the exact agent-visible `submission_schema.json`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
